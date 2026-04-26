"""Localhost catalog connection configs.

Concrete ``deriva_ml`` group entries that point at specific localhost
catalogs. Select one at the CLI:

    uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
        deriva_ml=localhost_1407 assets=roc_lr_sweep_localhost

The default ``deriva.py`` ships ``default_deriva`` as a placeholder
(catalog_id=0) — concrete connections live here so that ``deriva_ml=...``
overrides keep the defaults file portable across environments.
"""

from hydra_zen import store

from deriva_ml import DerivaMLConfig

deriva_store = store(group="deriva_ml")

deriva_store(
    DerivaMLConfig,
    name="localhost_1407",
    hostname="localhost",
    catalog_id=1407,
    use_minid=False,
    zen_meta={
        "description": (
            "Localhost catalog 1407 (cifar10_e2e schema, 200 CIFAR-10 images, "
            "fresh end-to-end test catalog)."
        )
    },
)
