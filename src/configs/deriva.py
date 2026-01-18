"""DerivaML Connection Configuration.

This module defines connection configurations for the Deriva catalog.

Configuration Group: deriva_ml
------------------------------
This group specifies which Deriva catalog to connect to. Each configuration
provides connection parameters (hostname, catalog_id, credentials).

REQUIRED: A configuration named "default_deriva" must be defined.
This is used as the default connection when no override is specified.

Example usage:
    # Use default connection
    uv run src/deriva_run.py

    # Use a specific connection
    uv run src/deriva_run.py deriva_ml=eye-ai
"""

from hydra_zen import store
from deriva_ml import DerivaMLConfig

# ---------------------------------------------------------------------------
# DerivaML Connection Configurations
# ---------------------------------------------------------------------------
# The group name "deriva_ml" must match the parameter name in BaseConfig.

deriva_store = store(group="deriva_ml")

# REQUIRED: default_deriva - used when no connection is specified
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="localhost",
    catalog_id=2,
    use_minid=False,
    zen_meta={
        "description": (
            "Local development catalog (localhost:2) with CIFAR-10 data. "
            "Schema: cifar10_e2e_test. Use for E2E testing and development."
        )
    },
)