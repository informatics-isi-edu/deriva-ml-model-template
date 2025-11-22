"""
This file is a template for how to set up a stand-alone script to execute a model.
"""

from typing import Any

from deriva_ml import (
    RID,
    DerivaML,
    DerivaMLConfig,
    MLVocab,
)
from deriva_ml.dataset import DatasetConfigList
from deriva_ml.execution import Execution, ExecutionConfiguration
from hydra_zen import builds, store, zen

# Load the predefined configurations for this script.  These configurations will be stored into the
# hydra-zen store.
# In this example, we have configurations for datasets, deriva_ml, and model assets.  Additional configurations
# can be added.
import configs.datasets
import configs.deriva
import configs.models
import configs.experiments


def model(learning_rate: float, epochs: int, execution: Execution):
    """A  simple model function.  This should be replaced with the proper top level model for the script.
    Args:
        learning_rate: Sample model parament
        epochs:  Sample model parameter
        execution: DerivaML execution object that will contain datasets and assets.

    Returns:

    """
    print(f"Training with learning rate: {learning_rate} and epochs: {epochs} and dataset")
    print(execution.datasets)
    print(execution.assets)

# Build a configuration interface for our model, providing default values. The execution value will be populated later
# at runtime, not configuration time.
# We use hydra-zens introspection capaiblity so that we don't have to build our own configuration object.
ModelConfig = builds(model, learning_rate=1e-3, epochs=10,
                     populate_full_signature=True,
                     zen_partial=True)
model_store = store(group="model_config")
model_store(ModelConfig, learning_rate=1e-3, epochs=10, name="model1")
model_store(ModelConfig, name="model2", learning_rate=23, epochs=20)


# Default configuration values are defined in configure.  Sets up the default confguration using hydra configurations
# defined in config directory and above.
@store(name="app_config",
           populate_full_signature=True,
           hydra_defaults=["_self_", {"deriva_ml": "local"},
                           {"datasets": "test1"},
    #                       {"assets": "asset1"},
                           {"model_config": "model1"}],
       )
def main(
    deriva_ml: DerivaMLConfig,
    datasets: DatasetConfigList,
    model_config: Any,
  #  assets: list[RID] = None,
    dry_run: bool = False,
) ->None:
    """
    Main entry point for this script.  Argument names must correspond to configuration groups in hydra configuration.
    Args:
        deriva_ml: A configuration for DeriaML with values corrisponding to parameters DerivaML()
        datasets: A list of datasets to use in creating an ExecutionConfig
        model_config: Configuration for the ML model.
        assets:  A list of assets to be used in creating an ExecutionConfig
        dry_run: Optional dryrun parameter for Execution.  Other configuration arguments could be added here.

    Returns:

    """
#    assets = assets or []
    datasets = datasets or []
    print(deriva_ml.model_dump())
    # deriva_ml is a pydantic dataclass, so get the dictionary representation of it.
    ml_instance = DerivaML(**deriva_ml.model_dump())  # This should be changed to the domain specific class.

    # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
    ml_instance.add_term(MLVocab.workflow_type, "Template Model Script", description="Initial setup of Model Notebook")
    workflow = ml_instance.create_workflow('demo-workflow', 'Template Model Script')

    # Create an execution instance.
    config = ExecutionConfiguration(
        datasets=datasets,
   #     assets=assets,
        workflow=workflow)

    execution = ml_instance.create_execution(config, dry_run=dry_run)
    with execution as e:
        # The model function has been partially configured, so we need to instantiate it with the execution object.
        model_config(execution=e)
    print("Uploading outputs...")
    execution.upload_execution_outputs()


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="app_config",
        version_base="1.3",
        config_path=".",
    )