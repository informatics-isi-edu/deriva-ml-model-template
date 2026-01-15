"""Define experiments.

Experiments are pre-configured combinations of model, dataset, and asset settings.
They use Hydra's defaults list to override specific config groups.

Usage:
    # Run a single experiment
    uv run src/deriva_run.py +experiment=cifar10_quick

    # Run multiple experiments
    uv run src/deriva_run.py --multirun +experiment=cifar10_quick,cifar10_extended

    # Override experiment settings
    uv run src/deriva_run.py +experiment=cifar10_quick datasets=cifar10_small_training

Reference:
    https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html
"""

from hydra_zen import make_config, store

# Get the base configuration from the store.
# Experiments must inherit from this base config.
# The key is a tuple (package, name) where package is None for root configs.
Config = store[None][(None, "deriva_model")]

# Use _global_ package to allow overrides at the root level
experiment_store = store(group="experiment", package="_global_")

# CIFAR-10 CNN experiments
# These experiments use the CIFAR-10 CNN model with different configurations.

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_split"},
        ],
        bases=(Config,),
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
        bases=(Config,),
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
        bases=(Config,),
    ),
    name="cifar10_extended",
)

# Full dataset experiments
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_small_split"},
        ],
        bases=(Config,),
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
        bases=(Config,),
    ),
    name="cifar10_extended_full",
)
