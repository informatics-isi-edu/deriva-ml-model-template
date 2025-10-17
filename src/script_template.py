"""
This file is a template for how to set up a stand-alone script to execute a model.
"""

from hydra_zen import zen, ZenStore
from hydra.conf import RunDir, HydraConf
import hydra
from deriva_ml import Execution, DatasetConfig, DatasetConfigList, RID, DerivaMLConfig

# These should be set to be the RIDs of input datasets and assets that are downloaded prior to execution.

store = ZenStore()

deriva_store = store(group="deriva_ml_config")
deriva_store(DerivaMLConfig, name="local",  hostname="localhost", catalog_id="1", working_dir="foobar")
deriva_store(DerivaMLConfig, name="eye-ai", hostname="www.eye-ai.org", catalog_id="eye-ai")

datasets_store = store(group="datasets")
datasets_test1 = DatasetConfigList(datasets=[DatasetConfig(rid="10", version="1.0.0")],
                                   description= "Test one datasets")
datasets_test2 = DatasetConfigList(datasets=[DatasetConfig(rid="15", version="1.0.0")])
datasets_test3 = DatasetConfigList(datasets =[])

datasets_store(datasets_test1, name="datasets_test1")
datasets_store(datasets_test2, name="datasets_test2")
datasets_store(datasets_test3, name="datasets_test3")

# Assets, typically the model file, but could be others as well.
asset_store = store(group="assets")
assets_test1 = ["asset3", "asset4"]
assets_test2 = ["asset5", "asset6"]
asset_store(assets_test1, name="assert1")
asset_store(assets_test2, name="assert2")

@store(name="app_config",
           populate_full_signature=True,
           hydra_defaults=["_self_", {"deriva_ml_config": "local"}],
       )
def main(
    deriva_ml_config: DerivaMLConfig,
    datasets: list[RID] = None,
    assets: list[RID] = None,
    test: bool = False,
):
    assets = assets or []
    """Parse arguments and set up execution environment."""

    hostname = deriva_ml_config.hostname
    catalog_id = deriva_ml_config.catalog_id
    print(f"hostname: {hostname}")
    print(f"catalog_id: {catalog_id}")
    print("working directory:", deriva_ml_config.working_dir)
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    #deriva_ml = DerivaML(**deriva_ml_config)  # This should be changed to the domain specific class.

    # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
    #deriva_ml.add_term(MLVocab.workflow_type, "Demo Script", description="Initial setup of Model Notebook")
    #workflow = deriva_ml.create_workflow('demo-workflow', 'Template Model Script')

    # Create an execution instance that will work with the latest version of the input datasets.
    #config = ExecutionConfiguration(
    #    datasets=datasets.datasets,
    #    assets=assets.assets,
    #    workflow=workflow,
    #)

   # execution = deriva_ml.create_execution(config, dry_run=args.dry_run)
   # with execution as e:
   #     self.do_stuff(e)
   # self.execution.upload_execution_outputs()


def do_stuff(execution: Execution):
    print(f" Execution with parameters: {execution.parameters}")
    print(f" Execution with input assets: {[a.as_posix() for a in execution.asset_paths]}")
    print(f"Execution datasets: {execution.datasets}")


from dataclasses import dataclass, field
@dataclass
class MyLogConf:
    version: int = 1
    formatters: dict = field(
        default_factory=lambda: {
            "simple": {"format": "[%(levelname)s] - %(message)s"}
        }
    )
    handlers: dict = field(
        default_factory=lambda: {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": "${hydra.run.dir}/my_custom_log.log",  # Use interpolation
            },
        }
    )
    root: dict = field(
        default_factory=lambda: {"handlers": ["file"], "level": "INFO"}
    )
    disable_existing_loggers: bool = False

store(HydraConf(run=RunDir("${deriva_ml_config.working_dir}/hydra/${now:%Y-%m-%d_%H-%M-%S}")))

# Overwrite the job_logging configuration with our custom setup
# Note: store may throw ZenStoreError if already exists, use overwrite=True
store(MyLogConf, group="hydra/job_logging", name="custom")


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="app_config",
        version_base="1.3",
        config_path=".",
    )