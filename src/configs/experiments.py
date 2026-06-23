"""Experiment configurations.

Configuration Group: ``experiment`` (registered at the ``_global_`` package
so its defaults override config groups at the root).

An *experiment* is a named, pre-wired combination of config groups — typically
a model config paired with a dataset config — built on the top-level
``DerivaModelConfig``. It lets you capture a meaningful run as a single name
instead of a long string of CLI overrides.

Usage:
    # Run a registered experiment
    uv run deriva-ml-run +experiment=<name>

    # Override an experiment's settings on top of its defaults
    uv run deriva-ml-run +experiment=<name> datasets=<other_dataset>

For hyperparameter sweeps and grid searches, see ``configs/multiruns.py`` —
those are self-contained and do not require an experiment definition.

IMPORTANT: When overriding optional fields (e.g. ``script_config``), set them
to MISSING in ``make_config`` so Hydra fills them from the defaults list rather
than shadowing the resolved value with the base's ``None`` default.

Reference:
    https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html
"""

from hydra_zen import make_config, store

from configs.base import DerivaModelConfig

# Use the _global_ package so an experiment's defaults override config groups
# at the root level.
experiment_store = store(group="experiment", package="_global_")

# A single live ``default`` experiment so ``--list-configs`` always shows at
# least one experiment. It pairs the required ``default_model`` and
# ``default_dataset`` configs — a no-op placeholder run until you supply a real
# model and dataset.
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "default_model"},
            {"override /datasets": "default_dataset"},
        ],
        description="Default experiment: the placeholder model on the default dataset.",
        bases=(DerivaModelConfig,),
    ),
    name="default",
)

# -----------------------------------------------------------------------------
# Add your own experiments below by pairing a model config with a dataset
# config. Example (replace the placeholders with names you registered in
# ``model.py`` and ``datasets.py``):
# -----------------------------------------------------------------------------
#
#   experiment_store(
#       make_config(
#           hydra_defaults=[
#               "_self_",
#               {"override /model_config": "<your_model_config>"},
#               {"override /datasets": "<your_dataset_group>"},
#           ],
#           description="<what this experiment evaluates>",
#           bases=(DerivaModelConfig,),
#       ),
#       name="<your_experiment>",
#   )
