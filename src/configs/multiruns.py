"""Multirun configurations.

A *multirun* bundles a set of Hydra overrides (the sweep axes) together with a
rich markdown description, registered under a single name. Running
``+multirun=<name>`` then launches the whole sweep — no ``--multirun`` flag and
no need to remember the override syntax each time.

Each ``multirun_config(...)`` call pairs override strings with a description
imported from ``configs/multirun_descriptions.py``. Comma-separated values in an
override (e.g. ``model_config.learning_rate=0.001,0.01``) become the sweep axes.

Usage:
    # Run a registered multirun (no --multirun flag needed)
    uv run deriva-ml-run +multirun=<name>

    # Override parameters on top of a registered multirun
    uv run deriva-ml-run +multirun=<name> model_config.epochs=5

    # Show available multiruns
    uv run deriva-ml-run --info
"""

from deriva_ml.execution import multirun_config

from configs.multirun_descriptions import EXAMPLE_SWEEP_DESCRIPTION

# A single live example sweep over the placeholder model's learning rate,
# built on the ``default`` experiment. Replace the override axes (and point at
# your own experiment) once you have a real model and dataset.
multirun_config(
    "example_sweep",
    overrides=[
        "+experiment=default",
        "model_config.learning_rate=0.001,0.01",
    ],
    description=EXAMPLE_SWEEP_DESCRIPTION,
)

# -----------------------------------------------------------------------------
# Add your own sweeps below. Example (a 2x2 grid over two parameters):
# -----------------------------------------------------------------------------
#
#   multirun_config(
#       "<your_sweep>",
#       overrides=[
#           "+experiment=<your_experiment>",
#           "model_config.<param_a>=<v1,v2>",
#           "model_config.<param_b>=<w1,w2>",
#       ],
#       description=MY_SWEEP_DESCRIPTION,  # from multirun_descriptions.py
#   )
