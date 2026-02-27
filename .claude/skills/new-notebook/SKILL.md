---
name: new-notebook
description: Create a new DerivaML notebook with configuration. Use when user asks to add a notebook, create an analysis notebook, or scaffold notebook config.
argument-hint: "[notebook-name]"
disable-model-invocation: true
---

# Create a New DerivaML Notebook

Create a new notebook named `$ARGUMENTS` with its hydra-zen configuration.

## Step 1: Define a Config Module

Create `src/configs/<notebook_name>.py`:

**Simple notebook** (only standard fields — assets, datasets, workflow):
```python
from deriva_ml.execution import notebook_config

notebook_config(
    "<notebook_name>",
    defaults={"assets": "my_assets", "datasets": "my_dataset"},
)
```

**Notebook with custom parameters:**
```python
from dataclasses import dataclass
from deriva_ml.execution import BaseConfig, notebook_config

@dataclass
class MyAnalysisConfig(BaseConfig):
    threshold: float = 0.5
    num_iterations: int = 100

notebook_config(
    "<notebook_name>",
    config_class=MyAnalysisConfig,
    defaults={"assets": "my_assets"},
)
```

Multiple named configs can share one file:
```python
notebook_config(
    "<notebook_name>",
    defaults={"assets": "my_assets"},
)

notebook_config(
    "<notebook_name>_variant",
    defaults={"assets": "other_assets", "datasets": "other_dataset"},
)
```

## Step 2: Create the Notebook

Create `notebooks/<notebook_name>.ipynb` with an initialization cell:

```python
from deriva_ml.execution import run_notebook

ml, execution, config = run_notebook("<notebook_name>")

# Ready to use:
# - ml: Connected DerivaML instance
# - execution: Execution context with downloaded inputs
# - config: Resolved configuration (config.assets, config.threshold, etc.)
```

And a final cell to upload outputs:
```python
execution.upload_execution_outputs()
```

## Step 3: Run with Overrides

```bash
# Show available configuration options
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb --info

# Run with defaults
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb

# Override assets or datasets (positional Hydra overrides, NOT --config)
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb \
    assets=different_assets

# Override host/catalog
uv run deriva-ml-run-notebook notebooks/<notebook_name>.ipynb \
    --host www.example.org --catalog 2
```

**Important:** `--config` does NOT override the `run_notebook()` config name in the notebook cell. Use positional Hydra overrides instead.

## Steps

1. **Create the config module** in `src/configs/` (one file per notebook)
2. **Create the notebook** in `notebooks/` with `run_notebook()` initialization
3. **Test**: `uv run deriva-ml-run-notebook notebooks/<name>.ipynb --info`
4. Use `execution.asset_file_path("Execution_Asset", filename)` for outputs
