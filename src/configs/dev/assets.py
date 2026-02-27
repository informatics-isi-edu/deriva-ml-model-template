"""Asset configurations for alternate catalog.

Configuration Group: assets

Add asset configs here that reference RIDs in a different catalog
than the default (localhost:6). Use a ``_dev`` or ``_prod`` suffix
convention so experiment configs can override cleanly.

Example:
    uv run deriva-ml-run assets=my_weights_dev
"""

# from hydra_zen import store
# from deriva_ml.asset.aux_classes import AssetSpecConfig
# from deriva_ml.execution import with_description
#
# asset_store = store(group="assets")
#
# # Example: large model weights with caching enabled
# asset_store(
#     with_description(
#         [AssetSpecConfig(rid="XXXX", cache=True)],
#         "Pre-trained model weights. ~3.7GB.",
#     ),
#     name="my_weights_dev",
# )
