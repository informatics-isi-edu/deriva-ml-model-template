"""
This file is a template for how to set up a stand-alone script to execute a model.
"""

from hydra_zen import zen, builds
from typing import Any

from deriva_ml import (
    DerivaML,
    DatasetConfigList,
    RID,
    DerivaMLConfig,
    MLVocab,
    ExecutionConfiguration,
    Execution
)

import configure

# Load our predefined configurations and initialize them.
store = configure.init_config()

# This is our simple model function.
def model(learning_rate: float, epochs: int, execution: Execution):
    print(f"Training with learning rate: {learning_rate} and epochs: {epochs} and dataset")
    print(execution.datasets)


# Build a configuration interface for our model, providing default values. The execution value will be populated later
# at runtime, not configuration time.
ModelConfig = builds(model, learning_rate=1e-3, epochs=10,
                     populate_full_signature=True,
                     zen_partial=True)
model_store = store(group="model_config")
model_store(ModelConfig, learning_rate=1e-3, epochs=10, name="model1")
model_store(ModelConfig, name="model2", learning_rate=23, epochs=20)

# Default configuration values are defined in configure.
@store(name="app_config",
           populate_full_signature=True,
           hydra_defaults=["_self_", {"deriva_ml": "local"}, {"datasets": "test1"}, {"assets": "asset1"},
                           {"model_config": "model1"}],
       )
def main(
    deriva_ml: DerivaMLConfig,
    datasets: DatasetConfigList,
    model_config: Any,
    assets: list[RID] = None,
    dry_run: bool = False,
):
    assets = assets or []
    datasets = datasets or []

    print("Datasets", datasets)
    print("Assets", assets)

    ml_instance = DerivaML(**deriva_ml.model_dump())  # This should be changed to the domain specific class.

    # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
    ml_instance.add_term(MLVocab.workflow_type, "Template Model Script", description="Initial setup of Model Notebook")
    workflow = ml_instance.create_workflow('demo-workflow', 'Template Model Script')

    # Create an execution instance that will work with the latest version of the input datasets.
    config = ExecutionConfiguration(
        datasets=datasets,
        assets=assets,
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