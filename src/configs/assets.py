"""Asset configurations.

Configuration Group: ``assets``
-------------------------------
An *asset config* names a list of asset RIDs (model weights, prediction files,
plots, etc.) that a model or notebook should consume as inputs. The runner
selects one with ``assets=<name>`` and the model reads it from
``config.assets``.

Asset RIDs are produced *by* prior executions in your catalog — a training run
that uploaded a weights file, for example — so the template ships with **no
live asset configs**, only the empty entries the runner requires. After an
execution prints its output RIDs, register them here. Two patterns:

1. **Edit this file** — add an ``asset_store([...], name="...")`` entry.
2. **Add a per-environment override** — register ``<name>_<env>`` configs in
   ``src/configs/dev/assets_<env>.py`` under this same ``assets`` group, then
   select with ``deriva-ml-run assets=<name>_<env>``.

Both a plain list and a ``with_description(...)`` wrapper work — the latter
surfaces a description in ``deriva-ml-run --info``. ``BaseConfig.assets`` is
typed ``Any = None`` so OmegaConf does not type-lock the slot, and
``with_description`` instantiates to a ``DescribedList`` that behaves like a
plain list at runtime.
"""

from hydra_zen import store
from deriva_ml.execution import with_description  # noqa: F401  (re-exported for users editing this file)

asset_store = store(group="assets")

# REQUIRED: ``default_asset`` is used when no ``assets`` override is given.
asset_store([], name="default_asset")

# Alias for clarity in notebook configs that take assets but no datasets.
asset_store([], name="no_assets")

# -----------------------------------------------------------------------------
# Add per-experiment asset configs below as you generate them. Example (replace
# the placeholder RIDs with the ones your execution printed):
# -----------------------------------------------------------------------------
#
#   asset_store(
#       with_description(
#           ["<your-rid>"],
#           "<what this asset is, e.g. trained weights from run X>",
#       ),
#       name="<your_assets>",
#   )
