# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DerivaML Model Template - a template for creating ML models integrated with DerivaML, a Python library for reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs for reproducibility.

## Common Commands

```bash
# Environment setup
uv sync                                    # Initialize environment
uv sync --group=jupyter                   # Add Jupyter support
uv sync --group=pytorch                   # Add PyTorch support

# Running models (uses Hydra config defaults for host/catalog)
uv run deriva-ml-run                                  # Run with defaults
uv run deriva-ml-run model_config=cifar10_quick      # Override model config
uv run deriva-ml-run +experiment=cifar10_quick       # Use experiment preset
uv run deriva-ml-run dry_run=true                    # Dry run (no catalog writes)
uv run deriva-ml-run +multirun=quick_vs_extended     # Named multirun
uv run deriva-ml-run --info                          # Show available configs

# Override host/catalog from command line
uv run deriva-ml-run --host localhost --catalog 45 +experiment=cifar10_quick

# Notebook execution (uses Hydra config defaults for host/catalog)
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb assets=my_assets
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --info

# Override host/catalog from command line
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \
  --host www.example.org --catalog 2

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Testing
uv run pytest

# Version management
uv run bump-version major|minor|patch
uv run python -m setuptools_scm

# Authentication
uv run deriva-globus-auth-utils login --host <hostname>

# Asset upload
uv run python scripts/upload_assets.py --dry-run     # Preview
uv run python scripts/upload_assets.py                # Upload from manifest

# Data loading (CIFAR-10 example)
uv run load-cifar10 --host <hostname> --catalog_id <id> --num_images 500
```

## Architecture

### Source Layout (`src/`)

- `src/configs/` — Hydra-zen configuration modules (Python, no YAML)
  - `base.py` — Base `DerivaModelConfig`
  - `deriva.py` — Catalog connection settings
  - `datasets.py` — Dataset specifications
  - `assets.py` — Asset RID configurations
  - `workflow.py` — Workflow definitions
  - `cifar10_cnn.py` — Model variant configs (7 variants)
  - `experiments.py` — Experiment presets
  - `multiruns.py` — Named multirun configurations
  - `multirun_descriptions.py` — Rich markdown descriptions for multiruns
  - `roc_analysis.py` — ROC analysis notebook config
  - `dev/` — Alternate catalog configs (connection, datasets, assets, experiments)
- `src/models/` — DerivaML model implementations
  - `cifar10_cnn.py` — CIFAR-10 CNN model
- `src/scripts/` — Data loading scripts
  - `load_cifar10.py` — CIFAR-10 dataset loader

### Configuration System (Hydra-Zen)

All configuration is Python-first using hydra-zen, no YAML files. Configs are in `src/configs/`. Config modules are auto-discovered via `pkgutil.iter_modules()` in `load_configs()`.

### Experiments (`Experiments.md`)

`Experiments.md` is the canonical registry of all defined experiments. It documents each experiment's config group references (workflow, model_config, datasets, assets), model parameters, and outputs. When adding or modifying experiments:
- **Prefer creating new experiments** over modifying existing ones to keep the history clear
- Always document new experiments in `Experiments.md` alongside the code in `experiments.py`
- Use `dry_run=true` to test before committing and running for real

### Model Pattern

Models follow this signature pattern:
```python
def model_name(
    param1: type = default,
    param2: type = default,
    # Always present - injected by framework
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
```

Wrap with `builds(..., zen_partial=True)` for deferred execution:
```python
from hydra_zen import builds, store

ModelConfig = builds(model_function, param1=val1, zen_partial=True)
store(ModelConfig, group="model_config", name="default")
store(ModelConfig, param1=val2, group="model_config", name="variant")
```

### Entry Point

`deriva-ml-run` - CLI provided by deriva-ml. Loads config modules from `src/configs/` automatically.

### Notebook Configuration Pattern

Notebooks use the simplified `run_notebook()` API for initialization:

1. **Define a config module** in `src/configs/` (e.g., `my_analysis.py`):

   Simple notebook (only standard fields):
   ```python
   from deriva_ml.execution import notebook_config

   notebook_config(
       "my_analysis",
       defaults={"assets": "my_assets", "datasets": "my_dataset"},
   )
   ```

   Notebook with custom parameters:
   ```python
   from dataclasses import dataclass
   from deriva_ml.execution import BaseConfig, notebook_config

   @dataclass
   class MyAnalysisConfig(BaseConfig):
       threshold: float = 0.5
       num_iterations: int = 100

   notebook_config(
       "my_analysis",
       config_class=MyAnalysisConfig,
       defaults={"assets": "my_assets"},
   )
   ```

2. **Initialize the notebook** (single call does everything):
   ```python
   from deriva_ml.execution import run_notebook

   ml, execution, config = run_notebook("my_analysis")

   # Ready to use:
   # - ml: Connected DerivaML instance
   # - execution: Execution context with downloaded inputs
   # - config: Resolved configuration (config.assets, config.threshold, etc.)

   # At the end of notebook:
   execution.upload_execution_outputs()
   ```

3. **Run notebook with overrides** (command line):
   ```bash
   # Show available configuration options
   uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info

   # Run with overrides (positional Hydra overrides, NOT --config)
   uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
     --host localhost --catalog 45 \
     assets=different_assets
   ```

## Catalog Environments

When using multiple catalogs (e.g., dev + production), add configs in `src/configs/dev/`:

| Config Name | Hostname | Usage |
|------------|----------|-------|
| `default_deriva` | localhost | Local development and testing |
| *(add more)* | *(your server)* | *(production, staging, etc.)* |

Production configs use a `*_prod` suffix convention (e.g., `my_dataset_prod`, `my_weights_prod`).

## Execution_Asset vs Execution_Metadata

DerivaML has two asset tables for execution files:

- **`Execution_Asset`** — Files **produced by** the execution (model outputs): checkpoints, metrics CSVs, predictions, plots, notebook outputs. Everything the model or notebook generates goes here.
- **`Execution_Metadata`** — Files **about** the execution environment: hydra configs, uv.lock, environment snapshots, configuration.json. These are auto-uploaded by DerivaML during initialization.

When writing model code, always use `execution.asset_file_path("Execution_Asset", filename)` for outputs. Never use `Execution_Metadata` for model-produced files.

## Tool Preferences

**IMPORTANT: Always start with DerivaML MCP tools for catalog operations.**

When interacting with DerivaML catalogs, **always prefer MCP tools over writing Python scripts**:
- **First**: Use `mcp__deriva-ml__connect_catalog` to establish a connection before any other catalog operation
- Use `mcp__deriva-ml__*` tools for catalog operations (connect, list datasets, download, query, etc.)
- Only write Python scripts when MCP tools are insufficient or when creating production code
- MCP tools provide cleaner, more direct interaction with the catalog

## Key Workflow Rules

- **MUST** commit changes before running models (DerivaML tracks code provenance)
- Use `dry_run=true` during debugging (downloads inputs without creating execution records)
- Tag versions with `bump-version` before significant model runs
- Commit `uv.lock` to repository
- Never commit notebooks with output cells (use `uv run nbstripout --install`)
- Use Google docstring format and type hints
- **Always check function/class signatures before modifying calls** - use `inspect.signature()` or check the source to verify required parameters before editing code that instantiates classes or calls functions
- **`--config` on `deriva-ml-run-notebook` does NOT override the `run_notebook()` config name** — use positional Hydra overrides instead (e.g., `assets=my_assets_prod`)

## CIFAR-10 Dataset Requirements

**IMPORTANT: Always use labeled datasets for experiments that require evaluation or analysis.**

The CIFAR-10 catalog contains two types of split datasets:
- **Unlabeled splits** (`cifar10_split`, `cifar10_small_split`): Test partition has no ground truth labels
- **Labeled splits** (`cifar10_labeled_split`, `cifar10_small_labeled_split`): Both train and test partitions have ground truth labels

For any experiment where you need to:
- Compute accuracy on the test set
- Generate ROC curves or other evaluation metrics
- Compare model predictions to ground truth

**You MUST use the labeled split datasets:**
- `cifar10_small_labeled_split` for small dataset experiments
- `cifar10_labeled_split` for full dataset experiments

The unlabeled splits are only appropriate for training-only runs where test evaluation is not needed.

## Overriding Configs at Runtime

```bash
# Choose different configs (no + prefix for groups with defaults)
uv run deriva-ml-run datasets=cifar10_small_training model_config=cifar10_quick

# Override specific fields (use + for adding new fields)
uv run deriva-ml-run model_config.epochs=50 model_config.learning_rate=0.01

# Use experiment presets
uv run deriva-ml-run +experiment=cifar10_extended

# Named multiruns (no --multirun flag needed)
uv run deriva-ml-run +multirun=lr_sweep
```
