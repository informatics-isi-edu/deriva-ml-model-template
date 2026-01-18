"""Define experiments.

Experiments are pre-configured combinations of model, dataset, and asset settings.
They use Hydra's defaults list to override specific config groups and inherit from
the main DerivaModelConfig.

Usage:
    # Run a single experiment
    uv run deriva-ml-run +experiment=cifar10_quick

    # Run multiple experiments (multirun)
    uv run deriva-ml-run --multirun +experiment=cifar10_quick,cifar10_extended

    # Override experiment settings
    uv run deriva-ml-run +experiment=cifar10_quick datasets=cifar10_small_training

Multirun with sweep_description:
    When running multiple experiments together, the `sweep_description` from the
    FIRST experiment (Job 0) is used for the parent execution. This allows you to
    document the purpose of the comparison with full markdown formatting.

Reference:
    https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html
"""

from hydra_zen import make_config, store

from configs.base import DerivaModelConfig
from configs.sweeps import (
    QUICK_VS_EXTENDED_DESCRIPTION,
    FULL_DATASET_DESCRIPTION,
    LEARNING_RATE_SWEEP_DESCRIPTION,
    EPOCH_SWEEP_DESCRIPTION,
)

# Use _global_ package to allow overrides at the root level
experiment_store = store(group="experiment", package="_global_")

# CIFAR-10 CNN experiments
# These experiments use the CIFAR-10 CNN model with different configurations.
# Each experiment inherits from DerivaModelConfig (a builds() of run_model)
# and overrides specific config groups.

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_split"},
        ],
        description="Quick CIFAR-10 training: 3 epochs, 32→64 channels, batch size 128 for fast validation",
        sweep_description=QUICK_VS_EXTENDED_DESCRIPTION,
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
        description="Default CIFAR-10 training: 10 epochs, 32→64 channels, standard hyperparameters",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_default",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_small_split"},
        ],
        description="Extended CIFAR-10 training: 50 epochs, 64→128 channels, dropout 0.25, weight decay 1e-4",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended",
)

# Full dataset experiments
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_split"},
        ],
        description="Quick CIFAR-10 on full dataset: 3 epochs, 32→64 channels for baseline validation",
        sweep_description=FULL_DATASET_DESCRIPTION,
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_quick_full",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
            {"override /datasets": "cifar10_split"},
        ],
        description="Extended CIFAR-10 on full dataset: 50 epochs, 64→128 channels, full regularization",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_extended_full",
)

# Test-only experiment - evaluate pre-trained model on test data
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_test_only"},
            {"override /datasets": "cifar10_small_testing"},
            {"override /assets": "multirun_quick_weights"},
        ],
        description="CIFAR-10 evaluation only: load pre-trained weights and evaluate on test set",
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_test_only",
)

# =============================================================================
# Hyperparameter Sweep Experiments
# =============================================================================
# These experiments are designed for parameter sweeps using --multirun

# Learning rate sweep base experiment
# Usage: uv run deriva-ml-run --multirun +experiment=cifar10_lr_sweep \
#            model_config.learning_rate=0.0001,0.001,0.01,0.1
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_lr_sweep"},
            {"override /datasets": "cifar10_small_split"},
        ],
        description="Learning rate sweep: exploring convergence across learning rates",
        sweep_description=LEARNING_RATE_SWEEP_DESCRIPTION,
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_lr_sweep",
)

# Epoch sweep base experiment
# Usage: uv run deriva-ml-run --multirun +experiment=cifar10_epoch_sweep \
#            model_config.epochs=5,10,25,50
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_epoch_sweep"},
            {"override /datasets": "cifar10_small_split"},
        ],
        description="Epoch sweep: analyzing training duration effects",
        sweep_description=EPOCH_SWEEP_DESCRIPTION,
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_epoch_sweep",
)
