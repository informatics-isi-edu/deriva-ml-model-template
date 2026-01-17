# Experiment Presets

Experiments allow you to define reusable combinations of configuration choices.

## Creating Experiments

Define experiments in `src/configs/experiments.py`:

```python
from hydra_zen import store

experiment_store = store(group="experiment")

# A quick test experiment
experiment_store(
    {
        "model_config": "quick",
        "datasets": "small_test",
    },
    name="quick_test",
)

# Full training run
experiment_store(
    {
        "model_config": "extended",
        "datasets": "full_training",
        "assets": "pretrained_weights",
    },
    name="full_training",
)

# Ablation study - no pretrained weights
experiment_store(
    {
        "model_config": "extended",
        "datasets": "full_training",
        "assets": "no_pretrain",
    },
    name="ablation_no_pretrain",
)
```

## Using Experiments

Run a single experiment:

```bash
uv run src/deriva_run.py experiment=full_training
```

Run multiple experiments in sequence (multirun):

```bash
uv run src/deriva_run.py --multirun experiment=quick_test,full_training
```

Override within an experiment:

```bash
uv run src/deriva_run.py experiment=full_training model_config.epochs=100
```

## Experiment Design Patterns

### Hyperparameter Sweep

```python
# Define model variants first
model_store(MyModelConfig, learning_rate=1e-2, name="lr_high")
model_store(MyModelConfig, learning_rate=1e-3, name="lr_medium")
model_store(MyModelConfig, learning_rate=1e-4, name="lr_low")

# Create experiments for each
experiment_store({"model_config": "lr_high"}, name="sweep_lr_high")
experiment_store({"model_config": "lr_medium"}, name="sweep_lr_medium")
experiment_store({"model_config": "lr_low"}, name="sweep_lr_low")
```

Run the sweep:
```bash
uv run src/deriva_run.py --multirun experiment=sweep_lr_high,sweep_lr_medium,sweep_lr_low
```

### Dataset Comparison

```python
experiment_store(
    {"datasets": "dataset_v1", "model_config": "default"},
    name="compare_v1",
)
experiment_store(
    {"datasets": "dataset_v2", "model_config": "default"},
    name="compare_v2",
)
```

### Ablation Study

```python
# Full model
experiment_store(
    {
        "model_config": "full_model",
        "assets": "all_features",
    },
    name="ablation_full",
)

# Without feature A
experiment_store(
    {
        "model_config": "no_feature_a",
        "assets": "no_feature_a",
    },
    name="ablation_no_a",
)

# Without feature B
experiment_store(
    {
        "model_config": "no_feature_b",
        "assets": "no_feature_b",
    },
    name="ablation_no_b",
)
```

## Best Practices

1. **Name experiments descriptively**: Use names that describe what's different
2. **Document your experiments**: Add comments explaining the purpose
3. **Version your data**: Use specific dataset versions in experiments
4. **Track results**: Each run creates an Execution record in the catalog

## Multirun Output

When using `--multirun`, Hydra creates separate output directories:

```
outputs/
├─ 2024-01-15/
│  ├─ 10-30-00/          # First experiment
│  │  ├─ .hydra/
│  │  └─ output.log
│  └─ 10-35-00/          # Second experiment
│     ├─ .hydra/
│     └─ output.log
```

Each run gets its own Execution record in the DerivaML catalog with full provenance tracking.
