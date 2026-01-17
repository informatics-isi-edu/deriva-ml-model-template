# Configuration Overview

DerivaML uses [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for configuration management. This provides a Python-first approach to configuration - no YAML files needed.

## Why Hydra-Zen?

- **Python-first**: Configurations are Python code with type hints and IDE support
- **Composable**: Mix and match configuration groups at runtime
- **Reproducible**: Configurations are serialized and tracked with your results
- **Flexible**: Override any parameter from the command line

## Configuration Architecture

```
src/configs/
├─ __init__.py          # Loads all config modules
├─ deriva.py            # DerivaML connection settings
├─ datasets.py          # Dataset specifications
├─ assets.py            # Asset RID configurations
├─ workflow.py          # Workflow metadata
├─ simple_model.py      # Model hyperparameters
├─ experiments.py       # Experiment presets
└─ my_notebook.py       # Notebook configurations
```

## Key Concepts

### Configuration Groups

Configuration groups organize related settings. Each group has multiple named configurations:

| Group | Purpose | Example Configs |
|-------|---------|-----------------|
| `deriva_ml` | Catalog connection | `local`, `eye_ai`, `dev` |
| `datasets` | Input data specification | `training`, `testing`, `full` |
| `assets` | Input assets (weights, etc.) | `weights_v1`, `pretrained` |
| `model_config` | Model hyperparameters | `quick`, `extended`, `regularized` |
| `workflow` | Workflow metadata | `training_workflow`, `analysis_workflow` |
| `experiment` | Preset combinations | `run1`, `baseline`, `ablation` |

### The `builds()` Function

The `builds()` function creates a structured configuration from a function or class:

```python
from hydra_zen import builds
from models.my_model import my_model

# Create a configuration that captures the function signature
MyModelConfig = builds(
    my_model,
    learning_rate=1e-3,
    epochs=10,
    populate_full_signature=True,  # Include all parameters
    zen_partial=True,              # Create partial function
)
```

### The `store()` Function

The `store()` function registers configurations with Hydra:

```python
from hydra_zen import store

# Create a store for a specific group
model_store = store(group="model_config")

# Register configurations
model_store(MyModelConfig, name="default")
model_store(MyModelConfig, epochs=50, name="extended")
```

### Defaults and Overrides

Each configuration file specifies defaults using `hydra_defaults`:

```python
hydra_defaults = [
    "_self_",
    {"deriva_ml": "default_deriva"},
    {"datasets": "default_dataset"},
]
```

Override at runtime:

```bash
# Use a different config from a group
uv run src/deriva_run.py datasets=testing

# Override a specific field
uv run src/deriva_run.py model_config.epochs=100

# Combine multiple overrides
uv run src/deriva_run.py datasets=testing model_config=extended
```

## Next Steps

- [Configuration Groups](groups.md) - Detailed guide to each group
- [Notebook Configuration](notebooks.md) - Simplified API for notebooks
- [Experiments](experiments.md) - Creating experiment presets
