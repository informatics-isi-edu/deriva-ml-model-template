"""Dataset configurations for alternate catalog.

Configuration Group: datasets

Add dataset configs here that reference RIDs in a different catalog
than the default (localhost:6). Use a ``_dev`` or ``_prod`` suffix
convention so experiment configs can override cleanly.

Example:
    uv run deriva-ml-run datasets=my_dataset_dev
"""

# from hydra_zen import store
# from deriva_ml.dataset import DatasetSpecConfig
# from deriva_ml.execution import with_description
#
# datasets_store = store(group="datasets")
#
# datasets_store(
#     with_description(
#         [DatasetSpecConfig(rid="XXXX", version="1.0.0")],
#         "Description of what this dataset contains.",
#     ),
#     name="my_dataset_dev",
# )
