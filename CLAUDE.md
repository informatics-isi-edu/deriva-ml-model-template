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

# Running the model
uv run src/deriva_run.py                           # Run with defaults
uv run src/deriva_run.py model_config=cifar10_quick datasets=cifar10_small_training
uv run src/deriva_run.py dry_run=true              # Dry run (no catalog writes)
uv run src/deriva_run.py --multirun experiment=run1,run2  # Multiple experiments

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Testing
uv run pytest

# Notebook execution (reproducible, uploads to catalog)
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb \
  --host www.eye-ai.org --catalog 2 --kernel <repo-name>

# Version management
uv run bump-version major|minor|patch
uv run python -m setuptools_scm

# Authentication
uv run deriva-globus-auth-utils login --host www.eye-ai.org

# Data loading (CIFAR-10 example)
uv run load-cifar10 --host <hostname> --catalog_id <id> --num_images 500
```

## Architecture

### Configuration System (Hydra-Zen)

All configuration is Python-first using hydra-zen, no YAML files. Configs are in `src/configs/`:
- `deriva.py` - DerivaML connection configs (local, eye-ai)
- `datasets.py` - Dataset specifications (test1, test2, test3)
- `assets.py` - Asset RID configurations (weights_1, weights_2)
- `workflow.py` - Workflow definitions
- `cifar10_cnn.py` - Model variant configs (7 variants)
- `experiments.py` - Experiment presets

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

`src/deriva_run.py` - Main CLI using Hydra. Dynamically loads all config modules via `src/configs/__init__.py`.

## Key Workflow Rules

- **MUST** commit changes before running models (DerivaML tracks code provenance)
- Use `dry_run=true` during debugging (downloads inputs without creating execution records)
- Tag versions with `bump-version` before significant model runs
- Commit `uv.lock` to repository
- Never commit notebooks with output cells (use `uv run nbstripout --install`)
- Use Google docstring format and type hints

## Overriding Configs at Runtime

```bash
# Choose different configs (no + prefix for groups with defaults)
uv run src/deriva_run.py datasets=cifar10_small_training model_config=cifar10_quick

# Override specific fields (use + for adding new fields)
uv run src/deriva_run.py model_config.epochs=50 model_config.learning_rate=0.01

# Use experiment presets
uv run src/deriva_run.py experiment=cifar10_extended
```
