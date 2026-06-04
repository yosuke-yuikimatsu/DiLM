import copy
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from dataset_attrs import DATASET_ATTRS
from utils import tqdm_disabled

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Config for Generator Model"""

    model_name: str = "gpt2"
    pretrained_model_dir: str | Path | None = None
    checkpoint_name: str | None = "last-ckpt"
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float = 1.0
    generate_batch_size: int = 32
    # number of GPUs to shard data generation across.
    #   1  -> single GPU (default, unchanged behavior)
    #   N  -> use N GPUs;  -1 -> use all visible GPUs
    generate_num_gpus: int = 1
    generate_max_length: int | None = None
    generate_bf16: bool = False
    generate_fp16: bool = False

    gradient_checkpointing: bool = False


class GeneratorModel(nn.Module):
    def __init__(self, config: GeneratorConfig, task_name: str):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.problem_type = DATASET_ATTRS[self.task_name]["problem_type"]
        self.num_labels = DATASET_ATTRS[task_name]["num_labels"]
        self.generate_max_length = self.config.generate_max_length
        if self.generate_max_length is None:
            self.generate_max_length = DATASET_ATTRS[self.task_name]["max_length"]

        assert self.problem_type != "single_label_classification" or self.num_labels > 1

        # setup model
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, from_tf=bool(".ckpt" in config.model_name)
        )

        if self.config.pretrained_model_dir is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            assert os.path.isdir(self.config.pretrained_model_dir)
            pretrained_model_path = os.path.join(
                self.config.pretrained_model_dir, self.config.checkpoint_name
            )
            assert os.path.exists(pretrained_model_path)
            self.load_model(save_path=pretrained_model_path)

            pretrained_tokenizer_path = os.path.join(
                self.config.pretrained_model_dir, "tokenizer"
            )
            assert os.path.exists(pretrained_tokenizer_path)
            self.load_tokenizer(save_path=pretrained_tokenizer_path)

            assert len(self.tokenizer) == len(self.model.get_input_embeddings().weight)

        # set pad token
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # add sep token (don't set as a special token)
        self.sep_token = "<sep>"
        self.tokenizer.add_tokens(self.sep_token)
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)

        self.model.resize_token_embeddings(len(self.tokenizer))

        # setup bos token for each label
        self.bos_tokens_map = {
            label_id: f"<bos_{label_id}>" for label_id in range(self.num_labels)
        }
        for bos_token in self.bos_tokens_map.values():
            if bos_token not in self.tokenizer.vocab:
                self.tokenizer.add_tokens(bos_token)
                self.model.resize_token_embeddings(len(self.tokenizer))
                bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)
                bos_token_weight = self.model.get_input_embeddings().weight[
                    self.tokenizer.bos_token_id
                ]
                with torch.no_grad():
                    self.model.get_input_embeddings().weight[bos_token_id].copy_(
                        bos_token_weight
                    )
        self.bos_token_ids_map = {
            label_id: self.tokenizer.convert_tokens_to_ids(bos_token)
            for label_id, bos_token in self.bos_tokens_map.items()
        }

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, *args, **kwargs) -> CausalLMOutputWithCrossAttentions:
        return self.model(*args, **kwargs)

    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        assert "labels" in kwargs
        labels: torch.LongTensor = kwargs.pop("labels")

        outputs: CausalLMOutputWithCrossAttentions = self.model(*args, **kwargs)

        # Shift so that tokens < n predict n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        return loss.reshape(shift_labels.shape).mean(-1)  # (batch_size,)

    def generate_dataset(self, dpc: int, n: int = 1) -> list[Dataset]:
        self.eval()
        self.cuda()
        dataset_list = []

        data_size_per_dataset = dpc * self.num_labels
        data_size = data_size_per_dataset * n
        generate_list = [
            {
                "sample_id": sample_id,
                "bos_token": [self.bos_token_ids_map[sample_id % self.num_labels]],
                "text": None,
                "labels": sample_id % self.num_labels,
            }
            for sample_id in range(data_size)
        ]

        devices = self.generate_devices()

        # generate data (optionally sharded across multiple GPUs)
        with trange(
            data_size,
            leave=False,
            dynamic_ncols=True,
            desc="Generating data",
            disable=tqdm_disabled(),
        ) as pbar:
            if len(devices) <= 1:
                generated_data = self.generate_on_device(
                    generate_list, self.model, self.device, data_size, pbar
                )
            else:
                generated_data = self.generate_multi_gpu(
                    generate_list, devices, data_size, pbar
                )

        # sort by sample id
        generated_data = [sample for _, sample in sorted(generated_data.items())]

        for i in range(n):
            dataset_list.append(
                Dataset.from_list(
                    generated_data[
                        data_size_per_dataset * i : data_size_per_dataset * (i + 1)
                    ]
                )
            )

        return dataset_list

    def generate_devices(self) -> list[torch.device]:
        """List of CUDA devices to shard generation across."""
        if self.device.type != "cuda":
            return [self.device]
        available = torch.cuda.device_count()
        # tolerate configs that predate this field (defaults to single GPU)
        num_gpus = getattr(self.config, "generate_num_gpus", 1)
        if num_gpus is None or num_gpus < 0:
            num_gpus = available
        num_gpus = min(max(num_gpus, 1), available)
        if num_gpus <= 1:
            return [self.device]
        return [torch.device("cuda", i) for i in range(num_gpus)]

    def generate_on_device(
        self,
        samples: list[dict],
        model: PreTrainedModel,
        device: torch.device,
        retry_budget: int,
        pbar=None,
        pbar_lock=None,
    ) -> dict[int, dict]:
        """Generate (with retry) for a list of samples on a single device."""
        pending = {sample["sample_id"]: sample for sample in samples}
        results = {}
        retry_count = 0
        while len(pending) > 0:
            batch = []
            for sample in pending.values():
                batch.append(sample)
                if len(batch) == self.config.generate_batch_size:
                    break
            generated_samples = self.batch_generate(batch, model=model, device=device)
            num_error = len(batch) - len(generated_samples)
            if num_error > 0:
                logger.warning(f"Number of failed samples is {num_error} (retry)")
                retry_count += num_error
                assert retry_count < retry_budget, "Too many samples failed to generate!!"

            for sample_id, generated_sample in generated_samples.items():
                pending.pop(sample_id)
                results[sample_id] = generated_sample

            if pbar is not None:
                if pbar_lock is not None:
                    with pbar_lock:
                        pbar.update(len(generated_samples))
                else:
                    pbar.update(len(generated_samples))

        return results

    def generate_multi_gpu(
        self,
        samples: list[dict],
        devices: list[torch.device],
        retry_budget: int,
        pbar=None,
    ) -> dict[int, dict]:
        """Shard generation across GPUs with one model replica per device.

        Each device runs in its own thread on an independent (frozen, eval) copy
        of the current generator weights; results are merged at the end. On any
        replication failure this falls back to single-GPU generation.
        """
        try:
            # device 0 reuses the live model; the rest get fresh weight copies
            replicas = {devices[0]: self.model}
            for device in devices[1:]:
                replica = copy.deepcopy(self.model).to(device)
                replica.eval()
                replicas[device] = replica
        except Exception as error:  # noqa: BLE001 - any failure -> safe fallback
            logger.warning(
                f"Failed to replicate generator across GPUs ({error}); "
                "falling back to single-GPU generation."
            )
            return self.generate_on_device(
                samples, self.model, self.device, retry_budget, pbar
            )

        # shard samples round-robin across devices
        shards = {device: [] for device in devices}
        for i, sample in enumerate(samples):
            shards[devices[i % len(devices)]].append(sample)

        results = {}
        results_lock = threading.Lock()
        pbar_lock = threading.Lock()

        def worker(device):
            with torch.cuda.device(device):
                shard_results = self.generate_on_device(
                    shards[device],
                    replicas[device],
                    device,
                    retry_budget,
                    pbar,
                    pbar_lock,
                )
            with results_lock:
                results.update(shard_results)

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(worker, device) for device in devices]
            for future in futures:
                future.result()  # propagate any exception from the worker threads

        # free replicas (keep self.model on device 0)
        for device in devices[1:]:
            replicas.pop(device, None)
        torch.cuda.empty_cache()

        return results

    @torch.inference_mode()
    def batch_generate(
        self,
        batch: list[dict],
        model: PreTrainedModel | None = None,
        device: torch.device | None = None,
    ) -> dict[int, dict]:
        model = self.model if model is None else model
        device = self.device if device is None else device
        inputs = [sample["bos_token"] for sample in batch]

        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = model.generate(
                torch.as_tensor(inputs, dtype=int, device=device),
                do_sample=True,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                max_length=self.generate_max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[bos] for bos in self.bos_token_ids_map.values()],
            )

        batch_generated_text = self.tokenizer.batch_decode(
            outputs[:, 1:], skip_special_tokens=True
        )
        sentence_keys = DATASET_ATTRS[self.task_name]["sentence_keys"]
        good_samples = {}
        for sample, generated_text in zip(batch, batch_generated_text):
            sentences = generated_text.split(self.sep_token)
            if len(sentences) >= len(sentence_keys):
                sentences = sentences[: len(sentence_keys)]
                if "" in sentences:
                    logger.warning(f"Empty sentence was generated: {generated_text}")
                generated_sample = {
                    key: sentence.strip()
                    for key, sentence in zip(sentence_keys, sentences)
                }
                generated_sample["labels"] = sample["labels"]
                good_samples[sample["sample_id"]] = generated_sample

        return good_samples

    def save_model(self, save_path):
        logger.info(f"Save generator model in `{save_path}`.")
        return self.model.save_pretrained(save_path)

    def load_model(self, save_path):
        logger.info(f"Load generator model from `{save_path}`.")
        self.model = self.model.from_pretrained(save_path)

    def save_tokenizer(self, save_path):
        logger.info(f"Save generator tokenizer in `{save_path}`.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return self.tokenizer.save_pretrained(save_path)

    def load_tokenizer(self, save_path):
        logger.info(f"Load generator tokenizer from `{save_path}`")
        self.tokenizer = AutoTokenizer.from_pretrained(save_path)

    @property
    def device(self):
        return self.model.device

    @property
    def use_amp(self):
        return self.config.generate_fp16 or self.config.generate_bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.generate_fp16 else torch.bfloat16
