"""
This file is a template for how to set up a stand-alone script to execute a model.
"""

from hydra_zen import zen, builds, instantiate

from deriva_ml import (
    DerivaML,
    DatasetConfigList,
    RID,
    DerivaMLConfig,
    MLVocab,
    ExecutionConfiguration,
)

import configure

store = configure.init_config()

def model(learning_rate: float, epochs: int):
    print(f"Training with learning rate: {learning_rate} and epochs: {epochs} and dataset")

ModelConfig = builds(model, learning_rate=1e-3, epochs=10, populate_full_signature=True)

store(ModelConfig, name="basemodel", group="model_config")
store(ModelConfig, learning_rate=23, epochs=20, name="model2", group="model_config")


# Default configuration values are defined in configure.
@store(name="app_config",
           populate_full_signature=True,
           hydra_defaults=["_self_", {"deriva_ml": "local"}, {"datasets": "test1"}, {"assets": "asset1"},
                           {"model_config": "basemodel"}],
       )
def main(
    deriva_ml: DerivaMLConfig,
    datasets: DatasetConfigList,
    model_config: ModelConfig,
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
    with execution as _:
        instantiate(model_config)
    print("Uploading outputs...")
    execution.upload_execution_outputs()


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="app_config",
        version_base="1.3",
        config_path=".",
    )