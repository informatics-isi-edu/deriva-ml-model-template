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
uv run deriva-ml-run --multirun +experiment=cifar10_quick,cifar10_extended  # Multiple experiments
uv run deriva-ml-run --info                          # Show available configs

# Override host/catalog from command line
uv run deriva-ml-run --host localhost --catalog 45 +experiment=cifar10_quick

# Notebook execution (uses Hydra config defaults for host/catalog)
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb assets=my_assets
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb --info

# Override host/catalog from command line
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb \
  --host www.eye-ai.org --catalog 2

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Testing
uv run pytest

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

   # Run with overrides
   uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
     --host localhost --catalog 45 \
     assets=different_assets
   ```

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
```

