"""Sweep configurations for multi-experiment runs.

Sweeps provide rich markdown descriptions for the parent execution when running
multiple experiments together. The sweep config documents why experiments are
being compared and what outcomes are expected.

Unlike ad-hoc parameter sweeps (e.g., `--multirun model_config.epochs=10,20,50`),
sweep configs let you:
- Document why experiments are being compared
- Use full markdown formatting in descriptions (tables, headers, bold, etc.)
- Track sweep metadata in the parent execution
- Keep the rationale alongside the code

Usage:
    # Run a pre-defined sweep (specify experiments on command line)
    uv run deriva-ml-run --multirun \\
        +experiment=cifar10_quick,cifar10_extended \\
        sweep_description="$(<src/configs/sweep_descriptions/quick_vs_extended.md)"

    # Or set sweep_description in your experiment config directly
"""

# =============================================================================
# Sweep Descriptions
# =============================================================================
# These are standalone markdown descriptions that can be used with any multirun.
# They are defined here as Python strings for easy reference and documentation.

QUICK_VS_EXTENDED_DESCRIPTION = """## CIFAR-10 CNN Multi-Experiment Comparison

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
"""

FULL_DATASET_DESCRIPTION = """## CIFAR-10 Full Dataset Comparison

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
"""
