# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DerivaML Model Template - a template for creating ML models integrated with DerivaML. This specific instance implements a CIFAR-10 CNN classifier with 7 model variants and an ROC analysis notebook.

## Common Commands

See [`../CLAUDE.md`](../CLAUDE.md) for shared `uv`, `pytest`, `ruff`,
and `bump-version` conventions. Repo-specific commands:

> **CWD:** every command below assumes you are in
> `/Users/carl/GitHub/DerivaML/deriva-ml-model-template`. The Bash tool's
> cwd is **not** reliably persistent across turns — always chain `cd` into
> a single call, e.g.
> `cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template && uv run deriva-ml-run ...`.
> See the workspace-level `CLAUDE.md` ("CWD discipline") for the rule.

```bash
# Environment setup (extra dependency groups)
uv sync --group=jupyter                   # Add Jupyter support
uv sync --group=torch                     # Add PyTorch support

# Running models
uv run deriva-ml-run +experiment=cifar10_quick       # Quick training (3 epochs)
uv run deriva-ml-run +experiment=cifar10_extended     # Extended training (50 epochs)
uv run deriva-ml-run +multirun=quick_vs_extended      # Compare quick vs extended
uv run deriva-ml-run dry_run=true                     # Dry run (no catalog writes)
uv run deriva-ml-run --info                           # Show available configs

# Notebook execution
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb deriva_ml=localhost_1407 assets=roc_e2e_localhost
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb deriva_ml=localhost_1407 assets=roc_lr_sweep_localhost
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --inspect  # show available papermill parameters

# Tests (config smoke tests; no catalog needed)
uv run python -m pytest tests/

# Data loading
uv run python src/scripts/load_cifar10.py \
    --hostname <hostname> --catalog-id <id> --num-images 500
# (or --create-catalog <name> instead of --catalog-id for a fresh catalog)
```

## Models

### CIFAR-10 2-Layer CNN (`src/models/cifar10_cnn.py`)

A PyTorch convolutional neural network for CIFAR-10 image classification.

**Architecture:** Conv2d(3, C1) → ReLU → MaxPool → Conv2d(C1, C2) → ReLU → MaxPool → Linear(C2×8×8, hidden) → ReLU → Linear(hidden, 10)

**Model configs** (in `src/configs/cifar10_cnn.py`):

| Config | Channels | Hidden | Epochs | LR | Notes |
|--------|----------|--------|--------|------|-------|
| `default_model` | 32→64 | 128 | 10 | 1e-3 | Standard training |
| `cifar10_quick` | 32→64 | 128 | 3 | 1e-3 | Fast validation |
| `cifar10_large` | 64→128 | 256 | 20 | 1e-3 | More capacity |
| `cifar10_regularized` | 32→64 | 128 | 20 | 1e-3 | Dropout 0.25, weight decay 1e-4 |
| `cifar10_fast_lr` | 32→64 | 128 | 15 | 1e-2 | Fast convergence |
| `cifar10_slow_lr` | 32→64 | 128 | 30 | 1e-4 | Stable convergence |
| `cifar10_extended` | 64→128 | 256 | 50 | 1e-3 | Best accuracy, full regularization |
| `cifar10_test_only` | 32→64 | 128 | — | — | Load weights, evaluate only |

**Experiments** (in `src/configs/experiments.py`):

| Experiment | Model Config | Dataset | Purpose |
|-----------|-------------|---------|---------|
| `cifar10_quick` | quick | small labeled split | Fast pipeline validation |
| `cifar10_default` | default | small training | Standard training |
| `cifar10_extended` | extended | small labeled split | Best accuracy on small set |
| `cifar10_quick_full` | quick | full labeled split | Baseline on full data |
| `cifar10_extended_full` | extended | full labeled split | Production run |
| `cifar10_test_only` | test_only | small labeled testing | Evaluate pretrained weights |

**Data flow:** Downloads dataset as BDBag → `restructure_assets()` creates ImageFolder layout → torchvision DataLoader → training/evaluation → saves weights + prediction CSV as execution assets.

### ROC Analysis Notebook (`notebooks/roc_analysis.ipynb`)

Compares model predictions across experiments by generating ROC curves. Configured via `src/configs/roc_analysis.py`. Takes asset RIDs (prediction CSVs) as input.

## Source Layout

- `src/configs/` — Hydra-zen configuration (Python, no YAML)
  - `__init__.py` — Re-exports `load_configs`; has deprecated `load_all_configs` alias
  - `base.py` — `BaseConfig` dataclass (shared config fields)
  - `cifar10_cnn.py` — Model configs (architectures, hyperparameters)
  - `datasets.py` — `DatasetSpecConfig` entries for each dataset
  - `deriva.py` — Deriva connection configs (`default_deriva`)
  - `workflow.py` — Workflow definitions
  - `assets.py` — Asset RID configs for model weights and predictions
  - `experiments.py` — Experiment configs (model + dataset combinations)
  - `multiruns.py` — Multirun sweep configs (parameter combinations)
  - `multirun_descriptions.py` — Rich markdown descriptions for multirun parent executions
  - `dataset_generation.py` — Dataset-generation script configs
  - `roc_analysis.py` — ROC notebook asset configs
  - `dev/` — Per-environment overrides (deriva, datasets, assets, experiments)
- `src/models/` — Model implementations
  - `cifar10_cnn.py` — CNN model, training loop, prediction recording
  - `model_protocol.py` — Protocol/interface for model functions
- `src/scripts/` — Data loading
  - `load_cifar10.py` — Downloads CIFAR-10 from Kaggle, sets up catalog schema, loads images
- `notebooks/` — Analysis notebooks (roc_analysis.ipynb)

## Key Rules

- **Commit before running** (also in README §8) — DerivaML tracks git commit hash for provenance; dirty-tree warnings appear when bouncing between branches. Use `DERIVA_ML_ALLOW_DIRTY=true uv run ...` for iterative test runs (see workspace `CLAUDE.md` for the rationale).
- **Update configs for your catalog before running** (also in README §7) — the checked-in dataset RIDs are stale demo RIDs; `Catalog Environments` below has the agent-only details
- **Use labeled datasets for evaluation** — `cifar10_small_labeled_split` or `cifar10_labeled_split` (unlabeled splits have no test ground truth)
- **`Execution_Asset`** for model outputs (weights, predictions, plots); `Execution_Metadata` is auto-managed
- **Test with `dry_run=true`** before production runs
- **`--config` on `deriva-ml-run-notebook` does NOT override `run_notebook()` config name** — use positional Hydra overrides instead (e.g., `assets=my_assets_prod`)
- **`--host` / `--catalog` on `deriva-ml-run-notebook` are papermill parameters, NOT Hydra overrides** — the notebook connects to whichever `deriva_ml` group the in-notebook `run_notebook(...)` call resolves. To target a non-default catalog, pass `deriva_ml=<config_name>` as a Hydra override (and register the connection in `src/configs/dev/deriva_<env>.py`).

## Catalog Environments

| Config Name | Hostname | Usage |
|------------|----------|-------|
| `default_deriva` | localhost | Local development and testing |

Production configs use a `*_prod` suffix convention. Add alternate configs in `src/configs/dev/`.

### Pointing the configs at a new catalog

See README §7 ("Update configs for your catalog") for the user-facing
procedure and the config-name → loader-output RID mapping table.

**Agent-only notes** (not in README):

- Versions are not predictable — `0.21.0`/`0.22.0`/`0.4.0` are typical for a
  freshly loaded catalog. Get actual values via `ml.find_datasets()` →
  `current_version` per Dataset.
- For multi-env setups, the `dev/datasets_<env>.py` pattern registers
  parallel `<name>_<env>` configs in the same `datasets` group; select at
  CLI with `datasets=cifar10_small_labeled_split_<env>`. Working example
  exists at `src/configs/dev/datasets_localhost.py` (catalog 1248).

## Gotchas

- **Kaggle API key required** for `load-cifar10` — must have `~/.kaggle/kaggle.json` configured
- **Use `uv run python -m pytest`, not `uv run pytest`** — the venv's `pytest` shim has a stale shebang that points at system Python 3.10. `uv sync --reinstall` fixes it.
- **Two `scripts/` dirs** — `src/scripts/` (Python package, importable) vs `scripts/` (standalone shell/CLI utilities, not a package)
- **`num_workers=0`** in DataLoaders — required on macOS because `fork()` + MPS/GPU threads deadlock

## Related Docs

- `CIFAR10.md` — End-to-end guide for CIFAR-10 workflow (catalog setup, data loading, training, analysis)
- `Experiments.md` — Experiment configuration reference
- `experiment-decisions.md` — Design rationale and decision log
