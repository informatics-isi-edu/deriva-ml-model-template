"""
This file is a template for how to set up a stand-alone script to execute a model.
"""
from hydra_zen import store, zen, make_config, builds

from model_runner import run_model

# Load the predefined configurations for this script.  These configurations will be stored into the hydra-zen store.
import configs.datasets  # noqa: F401
import configs.deriva  # noqa: F401
import configs.assets  # noqa: F401
import configs.simple_model  # noqa: F401

deriva_model = builds(
    run_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "local"},
        {"datasets": "test1"},
        {"assets": "weights_1"},
        {"model_config": "default_model"},
    ],
)
store(deriva_model, name="deriva_model")

experiment_store = store(group="experiment", package="_global_")

# This is an overload, so we would apply this like:
#    >>> python deriva_run.py --multirun +experiment=run1, run2
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "test1"},
            {"override /assets": "weights_2"},
            {"override /model_config": "epochs_20"},
        ],
        bases=(deriva_model,)
   ),
    name="run1"
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "test2"},
            {"override /assets": "weights_1"},
            {"override /model_config": "epochs_20"},
        ],
        bases=(deriva_model,)
   ),
    name="run2"
)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(run_model).hydra_main(
        config_name="deriva_model",
        version_base="1.3",
        config_path=None,
    )
