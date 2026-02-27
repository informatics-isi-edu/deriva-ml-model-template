---
name: new-model
description: Create a new DerivaML model. Use when user asks to add a model, create a model, or scaffold a new pipeline.
argument-hint: "[model-name]"
disable-model-invocation: true
---

# Create a New DerivaML Model

Create a new model named `$ARGUMENTS` following the DerivaML pattern.

## Model Function Signature

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

- `ml_instance` and `execution` are injected by the framework at runtime
- All other parameters become configurable via hydra-zen
- The function should contain the full training/inference logic

## Configuration with hydra-zen

Wrap with `builds(..., zen_partial=True)` for deferred execution:
```python
from hydra_zen import builds, store

ModelConfig = builds(model_function, param1=val1, zen_partial=True)
model_store = store(group="model_config")
model_store(ModelConfig, name="default_model")
model_store(ModelConfig, param1=val2, name="variant_name")
```

## Steps

1. **Create the model file** in `src/models/`:
   - Implement the model function with the signature above
   - Use `execution.asset_file_path("Execution_Asset", filename)` for outputs
   - Never use `Execution_Metadata` for model-produced files

2. **Create the config file** in `src/configs/`:
   - Use `builds(model_function, ..., zen_partial=True)` for each variant
   - Register with `store(group="model_config")`
   - The REQUIRED `default_model` config must exist in `src/configs/cifar10_cnn.py` (or update `base.py`)

3. **Create a workflow** in `src/configs/workflow.py`:
   - Use `builds(Workflow, name=..., workflow_type=[...], description=...)`
   - `workflow_type` can be a string or list of strings
   - Register with `workflow_store(MyWorkflow, name="my_workflow")`

4. **Add experiments** in `src/configs/experiments.py`:
   - Use `make_config(hydra_defaults=[...], bases=(DerivaModelConfig,))`
   - Override `/model_config`, `/datasets`, `/workflow` as needed

5. **Document in `Experiments.md`** alongside the code

6. **Test with dry run**: `uv run deriva-ml-run +experiment=<name> dry_run=true`

## Example: Complete Model Scaffold

```python
# src/models/my_model.py
from deriva_ml import DerivaML
from deriva_ml.execution import Execution

def my_model(
    learning_rate: float = 1e-3,
    epochs: int = 10,
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
    # Load data from execution datasets
    for table, assets in execution.asset_paths.items():
        for asset_path in assets:
            print(f"Processing {asset_path} (RID: {asset_path.asset_rid})")

    # Save outputs
    output_path = execution.asset_file_path("Execution_Asset", "results.csv")
    # ... write results to output_path ...
```

```python
# src/configs/my_model.py
from hydra_zen import builds, store
from models.my_model import my_model

MyModelConfig = builds(my_model, zen_partial=True)
model_store = store(group="model_config")
model_store(MyModelConfig, name="my_model")
model_store(MyModelConfig, epochs=50, name="my_model_extended")
```
