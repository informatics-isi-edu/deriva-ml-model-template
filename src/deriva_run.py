"""
This file is a template for how to set up a stand-alone script to execute a model.
"""
from hydra_zen import store, zen, builds

from model_runner import run_model

deriva_model = builds(
    run_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "eye-ai"},
        {"datasets": "test1"},
        {"assets": "weights_1"},
        {"model_config": "default_model"},
    ],
)
store(deriva_model, name="deriva_model")

# Load the predefined configurations for this script.  These configurations will be stored into the hydra-zen store.
import configs.datasets  # noqa: F401, E402
import configs.deriva  # noqa: F401, E402
import configs.assets  # noqa: F401, E402
import configs.simple_model  # noqa: F401, E402
import configs.experiments  #noqa: F401, E402

if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(run_model).hydra_main(
        config_name="deriva_model",
        version_base="1.3",
        config_path=None,
    )
