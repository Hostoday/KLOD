# KLOD: Locality-Preserving Knowledge Editing via Non-Target Distribution Preservation

KLOD is a batch knowledge editing method for large language models. It fine-tunes a single target MLP layer using a three-term objective: a one-sided logit-odds hinge loss to inject the new fact, plus two KL-divergence regularizers that prevent the edit from disrupting unrelated outputs.

---

## Method

The loss for a batch of edits is:

```
L = L_edit  +  λ_rw · L_kl_prefix  +  λ_nt · L_kl_non_target
```

| Term | Description |
|---|---|
| `L_edit` | One-sided hinge on target-token logit odds: `mean(relu(β − logit_odds(y*)))` where `β = logit(target_alpha)` |
| `L_kl_prefix` | KL(original/pre-edit ‖ edited) on prefix (non-target) token positions |
| `L_kl_non_target` | KL on the non-target vocabulary distribution at target positions |

The logit odds of the target token y\* is defined as `log p(y*) − log(1 − p(y*))`, computed from the shifted logits. The hinge pushes p(y\*) above `target_alpha` without over-optimizing.

Default hyperparameters (from `scripts/run_klod.sh`):

| Hyperparameter | Default |
|---|---|
| `target_alpha` | 0.85 |
| `rewrite_kl_lambda` | 1.2 |
| `non_target_kl_lambda` | 0.6 |
| `early_stop_loss` | 0.1 |
| `num_steps` | 25 |
| `batch_size` | 100 |
| `lr` | 5e-4 |

---

## Repository Layout

```
KLOD/
├── klod/                          # KLOD training entry points
│   ├── train_klod.py              # Main KLOD trainer (logit-odds hinge + KL)
│   ├── train_klod_locft_150k.py   # Staged 150 k runner used by the method wrappers
│   └── train_locft_overtone.py    # LocFT-BF baseline trainer
├── evaluate/                      # Evaluators
│   ├── eval_hf_easyedit.py        # EasyEdit-style reliability/generalization/locality eval
│   ├── eval_hf_easyedit_kl.py     # KL-divergence analysis between base and edited model
│   ├── aggregate_easyedit_eval_results.py
│   ├── aggregate_kl_analysis_results.py
│   └── aggregate_lm_eval_results.py
├── hparams/                       # YAML hyperparameter files per method and model
│   ├── KLOD/                      # llama3-8b.yaml, qwen2.5-7b.yaml
│   ├── LocFT-BF/
│   ├── ROME/
│   ├── MEMIT/
│   ├── AlphaEdit/
│   ├── FT/
│   ├── UltraEdit/
│   └── TRAINING/
├── scripts/                       # Reproduction and aggregation scripts
│   ├── run_klod.sh                # Train KLOD (all model × dataset combos)
│   ├── run_klod_150k.sh           # Pure KLOD scaling run through 150 k edits
│   ├── run_locft_bf_150k.sh       # Pure LocFT-BF scaling run through 150 k edits
│   ├── run_eval_kl.sh             # KL-divergence analysis on trained models
│   ├── eval.sh                    # EasyEdit evaluation pipeline
│   ├── eval_150k.sh               # Method-selectable evaluation for 150 k-edit runs
│   ├── capa_eval.sh               # Capability (lm-eval) evaluation
│   ├── run_easyedit_batch.py      # Batch runner for ROME/MEMIT/AlphaEdit/UltraEdit baselines
│   ├── run_aggregate_*.sh         # Result aggregation wrappers
├── EasyEdit/                      # In-tree EasyEdit with KLOD/LocFT-BF extensions
├── data/                          # Datasets (see data/README.md)
│   ├── counterfact/               # counterfact_3k.json
│   ├── zsre/                      # zsre_3k.json
│   └── stats/                     # Precomputed covariance stats for ROME/MEMIT/AlphaEdit (gitignored)
└── outputs/                       # Generated models and results (gitignored)
    ├── Models/
    └── evaluation/
```

`EasyEdit/` is kept in-tree because the experiments depend on local extensions for KLOD, LocFT-BF, and batch editing.

---

## Supported Models and Datasets

**Models**

| Label | HuggingFace ID | Edited Layer |
|---|---|---|
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | Layer 22 (`down_proj`) |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | Layer 6 (`down_proj`) |

**Training datasets** (3 000 samples each)

| Label | File |
|---|---|
| `counterfact` | `data/counterfact/counterfact_3k.json` |
| `zsre` | `data/zsre/zsre_3k.json` |


---

## Setup

Experiments were run with **Python 3.9.7**

```bash
conda create -n KLOD python=3.9.7 -y
conda activate KLOD

python -m pip install --no-deps -r requirements.txt

# Set cache location (optional)
export HUGGINGFACE_HUB_CACHE=/path/to/hf/cache
```

For ROME, MEMIT, and AlphaEdit, precomputed covariance statistics must be placed under `data/stats/` (not tracked by git).

---

## Training

### KLOD

Run all 4 combinations (2 models × 2 datasets):

```bash
bash scripts/run_klod.sh
```

Override hyperparameters via environment variables:

```bash
REWRITE_KL_LAMBDA=1.5 NON_TARGET_KL_LAMBDA=0.8 \
EARLY_STOP_LOSS=0.05 TARGET_ALPHA=0.9 \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_klod.sh
```

Run a single configuration directly:

```bash
python klod/train_klod.py \
  --klod_config_path hparams/KLOD/llama3-8b.yaml \
  --data_path data/counterfact/counterfact_3k.json \
  --rewrite_kl_lambda 1.2 \
  --non_target_kl_lambda 0.6 \
  --early_stop_loss 0.1 \
  --target_alpha 0.85
```

Edited models are saved to:

```
outputs/Models/KLOD/<model-slug>_<data-stem>_rewritekl<λ>_ntkl<λ>_early_stop_loss<v>_target_alpha<α>/
```

Each run saves `training_requests.json` and `training_manifest.json` alongside the model weights.

### LocFT-BF baseline

The LocFT-BF baseline is invoked through the same infrastructure via `klod/train_locft_overtone.py` and the hparams in `hparams/LocFT-BF/`.

### Separate 150K scaling runs

The scaling experiment trains KLOD and LocFT-BF separately. The wrappers below invoke the shared staged runner with mutually exclusive objective phases and distinct output roots:

```bash
# Pure KLOD: 25 KLOD objective epochs and no LocFT-BF phase
bash scripts/run_klod_150k.sh

# Pure LocFT-BF: standard CE training and no KLOD objective epoch
bash scripts/run_locft_bf_150k.sh
```

| Method | Fixed phase settings | Default output root |
|---|---|---|
| KLOD | `--use_locft 0 --klod_epochs 25` | `outputs/Models/KLOD_150k/` |
| LocFT-BF | `--use_locft 1 --klod_epochs 0` | `outputs/Models/LocFT-BF_150k/` |

Both wrappers use the ZsRE training stream, save checkpoints at `3k,5k,10k,20k,50k,100k,150k`, and accept environment overrides such as `CONFIG_PATH`, `MODEL_NAME`, `DEVICE`, `PYTHON_BIN`, and `SAVE_COUNTS`. For example, to run Qwen2.5-7B-Instruct:

```bash
CONFIG_PATH=hparams/KLOD/qwen2.5-7b.yaml bash scripts/run_klod_150k.sh
CONFIG_PATH=hparams/LocFT-BF/qwen2.5-7b.yaml bash scripts/run_locft_bf_150k.sh
```

---

## Evaluation

### EasyEdit metrics (reliability / generalization / locality)

```bash
bash scripts/eval.sh
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ROOT` | `outputs/Models/` | Root directory of saved models |
| `METHOD_FILTER` | `KLOD` | Filter to specific method folder |
| `NUM_SAMPLES` | `3000` | Samples per dataset |
| `PARALLEL_EVALS` | `auto` | Parallel eval jobs (one per GPU) |

Results are written to `outputs/evaluation/eval_results_easyedit/`.

Aggregate across runs:

```bash
bash scripts/run_aggregate_easyedit_eval_results.sh
```

### 150K checkpoint evaluation

Evaluate the separately trained scaling checkpoints with the matching method selector:

```bash
METHOD=KLOD bash scripts/eval_150k.sh
METHOD=LocFT-BF bash scripts/eval_150k.sh
```

Results are separated under `outputs/evaluation/eval_results_easyedit_150k/KLOD_150k/` and `outputs/evaluation/eval_results_easyedit_150k/LocFT-BF_150k/`.

### KL-divergence analysis

After training, measure the KL divergence between the base and edited model on the training prompts:

```bash
bash scripts/run_eval_kl.sh
```

Results are written to `outputs/evaluation/Analysis/kl_analysis/KLOD/`.

The analysis measures pre-edit-to-edited KL divergence at evaluated
next-token positions. Prompt KL and Target KL are reported as token-position
averages over prompt-side and target-side positions, respectively. For
rewrite and rephrase contexts, Non-target KL removes the gold target token,
renormalizes the remaining vocabulary, and averages the resulting divergence
over target prediction positions. Locality contexts report full-distribution
Prompt KL and Target KL only.

Aggregate:

```bash
bash scripts/run_aggregate_kl_analysis_results.sh
```

### Capability evaluation (lm-eval)

```bash
bash scripts/capa_eval.sh
```

Aggregate:

```bash
bash scripts/run_aggregate_lm_eval_results.sh
```

---

## Baseline Methods

The following methods are run via `scripts/run_easyedit_batch.py` using the hparams in `hparams/<method>/`:

| Method | Type |
|---|---|
| ROME | Rank-one weight update |
| MEMIT | Multi-layer weight update |
| AlphaEdit | Null-space projected update |
| FT | Standard fine-tuning |
| UltraEdit | Training-free lifelong parameter update |

---

## Data Format

Training data (CounterFact / ZsRE) uses the following fields:

```json
{
  "case_id": 0,
  "prompt": "The mother tongue of Danielle Darrieux is",
  "target_new": "English",
  "subject": "Danielle Darrieux",
  "ground_truth": "French",
  "rephrase_prompt": "...",
  "locality_prompt": "...",
  "locality_ground_truth": "..."
}
```

Both `prompt`/`target_new` (CounterFact) and `src`/`alt` (ZsRE) field names are handled transparently.

---

## Notes

- `data/stats/` (covariance matrices for ROME/MEMIT/AlphaEdit) must be restored separately.
- `outputs/` (saved models, evaluation results, logs) is gitignored.
- Training requires CUDA; `bf16=true` is set by default and falls back to fp16 on devices that do not support bfloat16.
- Multi-GPU inference uses `model_parallel: true` via HuggingFace device maps.
