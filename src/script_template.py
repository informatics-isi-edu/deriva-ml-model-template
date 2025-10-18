"""
This file is a template for how to set up a stand-alone script to execute a model.
"""

from hydra_zen import zen, ZenStore
from hydra.core.hydra_config import HydraConfig

from deriva_ml import (
    DerivaML,
    Execution,
    DatasetConfigList,
    RID,
    DerivaMLConfig,
    MLVocab,
    ExecutionConfiguration
)

import configure

store = configure.init_config()

print("Initialized configurations:")
for conf in store:
    print(f"\t{conf['group']}.{conf['name']}")

# Default configuration values are defined in configure.
@store(name="app_config",
           populate_full_signature=True,
           hydra_defaults=["_self_", {"deriva_ml": "local"}, {"datasets": "test1"}, {"assets": "asset1"}],
       )
def main(
    deriva_ml: DerivaMLConfig,
    datasets: DatasetConfigList | None = None,
    assets: list[RID] = None,
    dry_run: bool = False,
):
    assets = assets or []
    datasets = datasets or []

    print("Datasets", datasets)
    print("Assets", assets)
    print("working directory:", deriva_ml.working_dir)
    print("hydra output dir", HydraConfig.get().runtime.output_dir)
    print("hydra output subdir", HydraConfig.get().output_subdir)

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
        do_stuff(e)
   # self.execution.upload_execution_outputs()


def do_stuff(execution: Execution):
    print(f" Execution with input assets: {execution.asset_paths}")
    print(f"Execution datasets: {execution.datasets}")


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="app_config",
        version_base="1.3",
        config_path=".",
    )