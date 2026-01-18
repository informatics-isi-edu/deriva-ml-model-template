"""Define experiments.

Experiments are pre-configured combinations of model, dataset, and asset settings.
They use Hydra's defaults list to override specific config groups and inherit from
the main DerivaModelConfig.

Usage:
    # Run a single experiment
    uv run deriva-ml-run +experiment=cifar10_quick

    # Run multiple experiments using a multirun config
    uv run deriva-ml-run +multirun=quick_vs_extended

    # Override experiment settings
    uv run deriva-ml-run +experiment=cifar10_quick datasets=cifar10_small_training

For hyperparameter sweeps and grid searches, use multirun configs defined in
configs/multiruns.py - they are self-contained and don't require separate
experiment definitions.

Reference:
    https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html
"""

from hydra_zen import make_config, store

from configs.base import DerivaModelConfig

# Use _global_ package to allow overrides at the root level
experiment_store = store(group="experiment", package="_global_")

# =============================================================================
# CIFAR-10 CNN Experiments
# =============================================================================
# These experiments use the CIFAR-10 CNN model with different configurations.
# Each experiment inherits from DerivaModelConfig (a builds() of run_model)
# and overrides specific config groups.

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Quick CIFAR-10 training: 3 epochs, 32->64 channels, batch size 128 for fast validation",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_quick",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "default_model"},
            {"override /datasets": "cifar10_small_training"},
        ],
        description="Default CIFAR-10 training: 10 epochs, 32->64 channels, standard hyperparameters",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_default",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_small_labeled_split"},
        ],
        description="Extended CIFAR-10 training: 50 epochs, 64->128 channels, dropout 0.25, weight decay 1e-4",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended",
)

# =============================================================================
# Full Dataset Experiments
# =============================================================================

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_labeled_split"},
        ],
        description="Quick CIFAR-10 on full dataset: 3 epochs, 32->64 channels for baseline validation",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_quick_full",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_labeled_split"},
        ],
        description="Extended CIFAR-10 on full dataset: 50 epochs, 64->128 channels, full regularization",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended_full",
)

# =============================================================================
# Test-Only Experiment
# =============================================================================
# Evaluate pre-trained model on test data without training

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_test_only"},
            {"override /datasets": "cifar10_small_labeled_testing"},
            {"override /assets": "multirun_quick_weights"},
        ],
        description="CIFAR-10 evaluation only: load pre-trained weights and evaluate on labeled test set",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_test_only",
)

