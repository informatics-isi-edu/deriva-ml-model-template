"""Dataset configurations.

Configuration Group: ``datasets``
---------------------------------
A dataset config names a set of catalog datasets (by RID + version) that a
model or notebook should consume. The runner selects one with
``datasets=<name>`` and the model reads it from ``execution.datasets``.

RIDs are catalog-specific, so the template ships with **no live dataset
configs** — only the special empty entries the runner requires. Add your own
once you have datasets in your catalog (discover them with
``ml.find_datasets()``). Two patterns:

1. **Edit this file in place.** Register a ``DatasetSpecConfig(rid=...,
   version=...)`` under a name of your choosing. Wrap with
   ``with_description(...)`` to surface a description in
   ``deriva-ml-run --info``.

2. **Add a per-environment override.** Create
   ``src/configs/dev/datasets_<env>.py`` registering ``<name>_<env>`` configs
   in this same ``datasets`` group, then select on the CLI:
   ``deriva-ml-run datasets=<name>_<env>``.

The empty ``default_dataset`` passes config validation but fails at execution
time ("Dataset '' not found") — which is intended: a fresh clone must not
silently run against someone else's RIDs.
"""

from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig  # noqa: F401  (re-exported for users editing this file)
from deriva_ml.execution import with_description  # noqa: F401

datasets_store = store(group="datasets")

# -----------------------------------------------------------------------------
# Add your dataset configs below. Example (replace the placeholders with the
# RID and version printed by your loader, or read from ``ml.find_datasets()``):
# -----------------------------------------------------------------------------
#
#   datasets_store(
#       with_description(
#           [DatasetSpecConfig(rid="<your-rid>", version="<ver>")],
#           "<one-line description of this dataset>",
#       ),
#       name="<your_dataset>",
#   )

# -----------------------------------------------------------------------------
# Special-case configs (always empty by design — keep these).
# -----------------------------------------------------------------------------

# For notebooks/analyses that consume asset RIDs, not datasets.
datasets_store([], name="no_datasets")

# For script-only experiments that manage their own data.
datasets_store([], name="none")

# REQUIRED: ``default_dataset`` is used when no ``datasets`` override is given.
# Point this at your most-frequently-used dataset once you add one above.
datasets_store([], name="default_dataset")
