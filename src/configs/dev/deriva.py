"""DerivaML Connection Configuration for alternate catalog.

Configuration Group: deriva_ml

When your project uses multiple catalogs (e.g., dev + production), add
connection configs here. Name them descriptively so experiments can
reference them with ``{"override /deriva_ml": "my-server"}``.

Example:
    uv run deriva-ml-run deriva_ml=my-server
"""

from hydra_zen import store
from deriva_ml import DerivaMLConfig

deriva_store = store(group="deriva_ml")

# Example: alternate catalog on a remote server
# deriva_store(
#     DerivaMLConfig,
#     name="my-server",
#     hostname="my-server.example.org",
#     catalog_id="my-catalog",
# )
