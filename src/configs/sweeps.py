"""Sweep configurations for multi-experiment runs.

Sweeps define pre-configured combinations of experiments to run together,
with rich documentation explaining the purpose and expected outcomes.

Unlike ad-hoc parameter sweeps (e.g., `--multirun model_config.epochs=10,20,50`),
sweep configs let you:
- Document why experiments are being compared
- Use full markdown formatting in descriptions
- Define reusable, named experiment combinations
- Track sweep metadata in the parent execution

Usage:
    # Run a pre-defined sweep
    uv run deriva-ml-run --multirun +sweep=quick_vs_extended

    # Override parameters within a sweep
    uv run deriva-ml-run --multirun +sweep=quick_vs_extended model_config.epochs=5

    # Ad-hoc sweeps still work (no sweep_description)
    uv run deriva-ml-run --multirun model_config.epochs=10,20,50
"""

from hydra_zen import make_config, store

from configs.base import DerivaModelConfig

# Sweep configurations use _global_ package to allow root-level overrides
sweep_store = store(group="sweep", package="_global_")

# =============================================================================
# CIFAR-10 Quick vs Extended Comparison
# =============================================================================
# This sweep compares a fast baseline model against a fully-trained model
# to evaluate the trade-off between training time and accuracy.

sweep_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_split"},
        ],
        description="Quick CIFAR-10 training: 3 epochs for fast validation",
        sweep_description="""## CIFAR-10 CNN Multi-Experiment Comparison

**Objective:** Compare model performance across two training configurations to evaluate
the trade-off between training speed and model accuracy.

### Experiments

| Experiment | Epochs | Architecture | Regularization | Dataset |
|------------|--------|--------------|----------------|---------|
| `cifar10_quick` | 3 | 32→64 channels, 128 hidden | None | Small Split (1,000 images) |
| `cifar10_extended` | 50 | 64→128 channels, 256 hidden | Dropout 0.25, Weight Decay 1e-4 | Small Split (1,000 images) |

### Configuration Details

**cifar10_quick** - Fast validation baseline
- Conv1: 32 channels → Conv2: 64 channels
- Hidden layer: 128 units
- Batch size: 128
- Learning rate: 1e-3
- No regularization

**cifar10_extended** - Production-quality training
- Conv1: 64 channels → Conv2: 128 channels
- Hidden layer: 256 units
- Batch size: 64
- Learning rate: 1e-3
- Dropout: 0.25
- Weight decay: 1e-4

### Expected Outcomes

- The quick model should train in under 1 minute but have low accuracy
- The extended model should achieve higher accuracy but may overfit on the small dataset
- This comparison helps validate the training pipeline before running on full data
""",
        bases=(DerivaModelConfig,),
    ),
    name="quick_vs_extended",
)

# =============================================================================
# Full Dataset Comparison
# =============================================================================
# Same model comparison but on the full CIFAR-10 dataset

sweep_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_split"},
        ],
        description="Quick CIFAR-10 on full dataset",
        sweep_description="""## CIFAR-10 Full Dataset Comparison

**Objective:** Evaluate model architectures on the complete CIFAR-10 dataset
(10,000 images) to get realistic performance estimates.

### Experiments

| Experiment | Epochs | Architecture | Dataset |
|------------|--------|--------------|---------|
| `cifar10_quick_full` | 3 | 32→64 channels | Full Split (10,000 images) |
| `cifar10_extended_full` | 50 | 64→128 channels | Full Split (10,000 images) |

### Notes

- Full dataset runs take significantly longer
- Extended model should show less overfitting with more training data
- Use this sweep for final model selection before production
""",
        bases=(DerivaModelConfig,),
    ),
    name="quick_vs_extended_full",
)
