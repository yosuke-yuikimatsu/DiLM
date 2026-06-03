# DiLM: Distilling Dataset into Language Model

Implementaiton of "DiLM: Distilling Dataset into Language Model for Text-level Dataset Distillation" (accepted by NAACL2024 Findings)".

**Abstract**: Dataset distillation aims to compress a training dataset by creating a small number of informative synthetic samples such that neural networks trained on them perform as well as those trained on the original training dataset. Current text dataset distillation methods create each synthetic sample as a sequence of word embeddings instead of a text to apply gradient-based optimization; however, such embedding-level distilled datasets cannot be used for training other models whose word embedding weights are different from the model used for distillation. To address this issue, we propose a novel text dataset distillation approach, called Distilling dataset into Language Model (DiLM), which trains a language model to generate informative synthetic training samples as text data, instead of directly optimizing synthetic samples. We evaluated DiLM on various text classification datasets and showed that distilled synthetic datasets from DiLM outperform those from current coreset selection methods. DiLM achieved remarkable generalization performance in training different types of models and in-context learning of large language models. Our code will be available at https://github.com/arumaekawa/DiLM.

**Paper**: [[arXiv](https://arxiv.org/abs/2404.00264)], [[NAACL2024 Findings](https://aclanthology.org/2023.acl-short.12/)]

## Contents

This repository utilizes [PyTorch](https://pytorch.org/) and modern experiment manager tools, [Hydra](https://hydra.cc/) and [MLflow](https://www.mlflow.org/).

Datasets and pre-trained models are downloaded and used with [Hugging Face](https://huggingface.co/).

### Directory structure

```
.
├── configs
│  ├── test
│  │  ├── coreset.yaml
│  │  ├── dc.yaml
│  │  └── lm.yaml
│  └── train
│     ├── generator
│     │  ├── pretrained_mnli.yaml
│     │  ├── pretrained_qqp.yaml
│     │  └── pretrained_sst2.yaml
│     ├── dc.yaml
│     └── lm.yaml
├── src
│  ├── coreset
│  │  ├── __init__.py
│  │  ├── coreset_base.py
│  │  ├── coreset_utils.py
│  │  ├── herding.py
│  │  ├── k_centers.py
│  │  ├── random.py
│  │  └── rank_dilm.py
│  ├── distillation
│  │  ├── __init__.py
│  │  ├── distilled_data.py
│  │  ├── trainer_base.py
│  │  ├── trainer_dc.py
│  │  └── trainer_lm.py
│  ├── data.py
│  ├── dataset_attrs.py
│  ├── evaluator.py
│  ├── generator.py
│  ├── learner.py
│  ├── test.py
│  ├── train.py
│  └── utils.py
├── README.md
└── requirements.txt
```

## Run Scripts

1. Install packages (Python 3.10)

   ```bash
   $ pip install -r requirements.txt
   ```

2. Run pre-training (LM)

   ```bash
    $ python src/train.py --config-name=lm data.task_name=sst2
   ```

3. Run dataset fine-tuning (Gradient Matching)

   ```bash
    $ python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2
   ```

4. Run evaluation

   ```bash
    $ python src/test.py --config-name=dc data.task_name=sst2 generator.pretrained_model_dir=path/to/pretrained_model_dir
   ```

5. Check the results with MLFlow (http://localhost:5000)

   ```bash
    $ mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
   ```

## Distribution Matching (DM)

In addition to the original **gradient matching** objective, DiLM can train the
generator with a **Distribution Matching (DM)** objective. DM feeds both the real
and the generated (synthetic) samples through the learner (BERT) used as a
*frozen feature extractor*, concatenates the hidden states of the selected
layers, averages them over the batch for each class, and minimizes the squared
distance between the real and the synthetic feature means:

```
L_DM = (1 / C) * Σ_c || mean_i f(x_real^c_i) − Σ_i w_i · f(x_syn^c_i) ||²
```

where `f(·)` is the concatenated hidden-layer representation and `C` is the
number of classes. As with gradient matching, the synthetic text is discrete and
therefore not directly differentiable, so the gradient reaches the generator
through the same generation-probability weights `w_i = softmax(−L_gen_i / τ)`
that DiLM already uses for gradient matching. In other words, **DM updates the
generator** (the learner is only a feature extractor and is *never* updated by
the DM loss).

DM can be used in two ways, selected with `train.dm_mode`:

| `train.dm_mode`     | Behavior                                                                                      |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `none` (default)    | Gradient matching only (original DiLM).                                                        |
| `regularizer`       | Gradient matching **+** `dm_lambda` × DM loss.                                                 |
| `standalone`        | DM loss only (gradient matching is disabled — faster and lighter, no per-parameter gradients). |

Relevant options (see the `train` section of `configs/train/dc.yaml`):

| Option                    | Default | Description                                                                                                                                                            |
| ------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train.dm_mode`           | `none`  | One of `none` / `regularizer` / `standalone`.                                                                                                                          |
| `train.dm_lambda`         | `1.0`   | Weight of the DM loss (used in both `regularizer` and `standalone`).                                                                                                   |
| `train.dm_hidden_layers`  | `[-1]`  | BERT hidden states to use. The `hidden_states` tuple has `num_layers + 1` entries (`0` = embedding output, `-1` = last layer). Multiple indices are concatenated, e.g. `[-4,-3,-2,-1]`. |
| `train.dm_pooling`        | `cls`   | `cls` uses the `[CLS]` token, `mean` uses the attention-masked token mean.                                                                                             |
| `train.use_projectors`    | `False` | Match **random projections** of the features (original DM) instead of the raw features. See below.                                                                     |
| `train.num_projectors`    | `1`     | Number of random projectors averaged per step (only used when `use_projectors=True`).                                                                                  |
| `train.projector_output_dim` | `128` | Output dimension of each random projector (only used when `use_projectors=True`).                                                                                      |

### Feature matching vs. random projections

By default (`train.use_projectors=False`) DM matches the concatenated hidden
features **directly**, i.e. it matches the first moment (mean) of the features.

Setting `train.use_projectors=True` reproduces the original Distribution Matching
more closely: at **every step** a fresh set of `num_projectors` random **frozen**
MLPs (`Linear -> ReLU -> Linear`, output size `projector_output_dim`) is sampled,
the real and synthetic features are pushed through each projector, and the squared
distance between the projected means is averaged over the projectors. Because of
the `ReLU` nonlinearity, `mean(ψ(f))` captures higher-order statistics of the
feature distribution (not just the mean), and re-sampling the projectors every
step matches the distribution over many random projections. The projectors are
never trained — as with direct matching, the gradient reaches only the generator
through `loss_weights`.

DM is implemented for the dataset fine-tuning trainer (`--config-name=dc`) and is
logged to MLflow as `train.loss_dm` alongside `train.loss_dc`/`train.grad_sim`.

Example overrides (local run):

```bash
# gradient matching + DM as a regularizer
$ python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2 \
    train.dm_mode=regularizer train.dm_lambda=0.5 'train.dm_hidden_layers=[-1]'

# DM as a standalone loss (no gradient matching)
$ python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2 \
    train.dm_mode=standalone 'train.dm_hidden_layers=[-4,-3,-2,-1]' train.dm_pooling=mean

# DM with random projectors (original DM), as a regularizer
$ python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2 \
    train.dm_mode=regularizer train.use_projectors=True \
    train.num_projectors=8 train.projector_output_dim=128
```

> Quote the layer list (`'train.dm_hidden_layers=[-1]'`) so the shell does not try
> to glob the square brackets.

## Training DiLM with DM on Kaggle

The steps below reproduce the two-stage DiLM training (generator pre-training →
dataset fine-tuning) inside a [Kaggle](https://www.kaggle.com/code) notebook,
using DM either as a regularizer or as a standalone loss.

> **Kaggle settings**: in the notebook **Settings** panel, set
> **Accelerator → GPU** (T4 ×2 or P100) and **Internet → On** (needed to install
> packages and download the GPT-2 / BERT weights and the GLUE dataset).

**1. Clone the repository and install dependencies**

```python
!git clone https://github.com/arumaekawa/DiLM.git
%cd DiLM
!pip install -r requirements.txt
```

> Kaggle already ships with a recent PyTorch build. If the pinned
> `torch==2.0.0+cu118` clashes with the pre-installed CUDA runtime, install the
> remaining requirements and keep Kaggle's PyTorch instead.

**2. Pre-train the generator (LM step)**

```python
!python src/train.py --config-name=lm data.task_name=sst2
```

This saves the generator to
`save/train.gpt2.bert-base-uncased.sst2/dilm.lm/step_80000/generator`, the path
referenced by `configs/train/generator/pretrained_sst2.yaml`.

> The defaults run for many steps. For a quick session you can shorten it with
> `train.total_train_step=8000` (mind Kaggle's GPU time limit). If you change the
> step count, the checkpoint folder becomes `.../dilm.lm/step_<N>/generator`, so
> pass that path explicitly in step 3 via `generator.pretrained_model_dir=...`.
> Re-running the LM step with the same step count fails because the output
> directory already exists — delete it or change the step count first.

**3. Dataset fine-tuning with DM**

DM as a **regularizer** (gradient matching + DM):

```python
!python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2 \
    train.dm_mode=regularizer train.dm_lambda=0.5 'train.dm_hidden_layers=[-1]'
```

DM as a **standalone loss** (gradient matching disabled):

```python
!python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2 \
    train.dm_mode=standalone 'train.dm_hidden_layers=[-4,-3,-2,-1]' train.dm_pooling=mean
```

To match **random projections** instead of the raw features (original DM), add
`train.use_projectors=True` to either command, e.g. a standalone projector run:

```python
!python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2 \
    train.dm_mode=standalone train.use_projectors=True \
    train.num_projectors=8 train.projector_output_dim=128
```

> The `dc` run uses a timestamped sub-run name, so it can be launched repeatedly
> without clearing the output directory.

**4. Evaluate the distilled dataset**

```python
!python src/test.py --config-name=dc data.task_name=sst2 \
    generator.pretrained_model_dir=path/to/pretrained_model_dir
```

**5. (optional) Inspect the metrics with MLflow**

DiLM logs `train.loss_dc`, `train.loss_dm`, `train.loss_lm` and `train.grad_sim`
to `./mlruns`. Download the `mlruns/` folder from the notebook output and open it
locally:

```bash
$ mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

## Performance: multi-GPU generation & quiet logging

### Multi-GPU generation

Synthetic-data generation (autoregressive GPT-2) is the dominant cost of the `dc`
run and is embarrassingly parallel, so it can be sharded across GPUs. Set
`generator.generate_num_gpus`:

```python
!python src/train.py --config-name=dc data.task_name=sst2 +generator=pretrained_sst2 \
    generator.generate_num_gpus=2 \
    train.dm_mode=standalone train.use_projectors=False ...
```

- `generate_num_gpus=1` (default) — single GPU, unchanged behavior.
- `generate_num_gpus=N` — shard generation across `N` GPUs (e.g. `2` for a Kaggle
  T4 ×2). `-1` uses all visible GPUs.

Each GPU runs an independent (frozen, eval) copy of the current generator weights
in its own thread; results are merged. On any replication failure it falls back
to single-GPU generation. Only **generation** is parallelized — the gradient /
distribution-matching step relies on `torch.func` and is left on the main GPU, so
the speed-up applies to the generation portion of each loop.

### Reducing progress-bar spam

On Kaggle the nested tqdm bars (the per-step learner-training bar during
validation, the data-generation bar, the inner-loop bar) are not collapsed and
flood the output. Silence them while keeping the periodic metric logs:

```python
import os
os.environ["DILM_DISABLE_TQDM"] = "1"   # propagates to the !python subprocess
```

`DILM_DISABLE_TQDM=1` disables the noisy nested bars but keeps the outer-loop
heartbeat. To disable **every** bar (including the heartbeat), use tqdm's built-in
`os.environ["TQDM_DISABLE"] = "1"` instead. The `train.*` metric lines logged
every `train.log_interval` steps are unaffected either way.

> Validation cost is dominated by `evaluate.train_step` (the learner is retrained
> from scratch on the distilled data at every `train.val_interval`). If validation
> feels too frequent/slow, increase `train.val_interval` and/or lower
> `evaluate.train_step` / `evaluate.n_eval_per_dataset`.

## Citation

```
@inproceedings{maekawa-etal-2023-dataset,
    title = "Dataset Distillation with Attention Labels for Fine-tuning {BERT}",
    author = "Maekawa, Aru  and
      Kobayashi, Naoki  and
      Funakoshi, Kotaro  and
      Okumura, Manabu",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.12",
    doi = "10.18653/v1/2023.acl-short.12",
    pages = "119--127",
}
```
