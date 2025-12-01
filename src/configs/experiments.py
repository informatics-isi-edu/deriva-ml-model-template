""" Define experiments

These will be stored in the experiment store and can be run with the --multirun +experiment=experiment_name.
    python deriva_run.py --multirun +experiment=run1, run2
"""

from hydra_zen import make_config, store

# Experiment extends the base configuration, so we need to get it from the store.
app_config = store[None]
app_name = next(iter(app_config))
deriva_model_config = store[None][app_name]

experiment_store = store(group="experiments")
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "test1"},
            {"override /assets": "weights_2"},
            {"override /model_config": "epochs_20"},
        ],
        bases=(deriva_model_config,)
   ),
    name="run1",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "test2"},
            {"override /assets": "weights_1"},
            {"override /model_config": "epochs_20"},
        ],
        bases=(deriva_model_config,)
   ),
    name="run2",
)
