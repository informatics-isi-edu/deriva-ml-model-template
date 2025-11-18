# Set up deriva_ml configurations.'
from hydra_zen import store
from deriva_ml.dataset import DatasetConfigList, DatasetConfig

# Configure datasets to use.
datasets_test1 = DatasetConfigList(
    datasets=[DatasetConfig(rid="4RP", version="1.3.0")],
    description="Test one datasets",
)
datasets_test2 = DatasetConfigList(datasets=[DatasetConfig(rid="4S2", version="1.3.0")])
datasets_test3 = DatasetConfigList(datasets=[])

datasets_store = store(group="datasets")
datasets_store(datasets_test1, name="test1")
datasets_store(datasets_test2, name="test2")
datasets_store(datasets_test3, name="test3")
