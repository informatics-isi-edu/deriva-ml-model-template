from hydra_zen import  ZenStore

from deriva_ml import (
    DatasetConfig,
    DatasetConfigList,
    DerivaMLConfig,
)


# Configure deriva_ml server connection.
def init_config() -> ZenStore:

    # Get a new version of a local repository store.
    store = ZenStore()

    # Set up deriva_ml configurations.
    deriva_store = store(group="deriva_ml")
    deriva_store(DerivaMLConfig, name="local",
                 hostname="localhost",
                 catalog_id=2,
                 use_minid=False)
    deriva_store(DerivaMLConfig, name="eye-ai",
                 hostname="www.eye-ai.org",
                 catalog_id="eye-ai")

    # Configure datasets to use.
    datasets_test1 = DatasetConfigList(datasets=[DatasetConfig(rid="4RP", version="1.3.0")],
                                       description= "Test one datasets")
    datasets_test2 = DatasetConfigList(datasets=[DatasetConfig(rid="4S2", version="1.3.0")])
    datasets_test3 = DatasetConfigList(datasets =[])

    datasets_store = store(group="datasets")
    datasets_store(datasets_test1, name="test1")
    datasets_store(datasets_test2, name="test2")
    datasets_store(datasets_test3, name="test3")

    # Assets, typically the model file, but could be others as well.
    assets_test1 = ["3QG", "3QJ"]
    assets_test2 = ["3QM", "3QP"]
    asset_store = store(group="assets")
    asset_store(assets_test1, name="asset1")
    asset_store(assets_test2, name="asset2")

    return store