"""
This file is a  stand alone driver to execute a model using hydra-zen to configure the execution.

You can run this script directly from the command line, specifying the configuration to use.

"""

from hydra_zen import store, zen, builds

from model_runner import run_model

# Create a configuration for this application.
deriva_model = builds(
    run_model,
    description="Simple model run",
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "local"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
    ],
)
store(deriva_model, name="deriva_model")

# Load the predefined configurations for this script.  These configurations will be stored into the hydra-zen store.
import configs.datasets  # noqa: F401, E402
import configs.deriva  # noqa: F401, E402
import configs.assets  # noqa: F401, E402
import configs.workflow # noqa: F401, E402
import configs.model  # noqa: F401, E402
import configs.experiments  # noqa: F401, E402

if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(run_model).hydra_main(
        config_name="deriva_model",
        version_base="1.3",
        config_path=None,
    )
