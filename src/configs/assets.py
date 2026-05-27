"""Asset Configurations.

Configuration Group: ``assets``

Asset RIDs are produced *by* prior executions in your catalog (e.g., a
training run that uploaded a weights file). They cannot be supplied by the
template — they only exist after you run experiments.

The defaults here are empty. After running an experiment, take the asset RIDs
it printed and either:

1. **Edit this file** — add an ``asset_store(["<rid>", ...], name="...")``
   entry referencing the RIDs.
2. **Add a per-environment override** — drop ``src/configs/dev/assets_<env>.py``
   that registers ``<name>_<env>`` configs in the same ``assets`` group, then
   select on the CLI: ``deriva-ml-run assets=quick_weights_<env>``.

Pattern for an entry with a description (recommended — descriptions show up
in ``deriva-ml-run --info`` output):

    asset_store(
        with_description(
            ["3WS2"],
            "Pre-trained weights from cifar10_quick (3 epochs).",
        ),
        name="quick_weights",
    )

A plain list also works if you don't want a description:

    asset_store(["3WS6", "3X20"], name="roc_quick_vs_extended")

Both forms compose with notebook configs — ``BaseConfig.assets`` is typed
``Any = None`` so OmegaConf doesn't type-lock the slot, and
``with_description`` instantiates to a ``DescribedList`` that behaves like
a plain list at runtime.
"""

from hydra_zen import store
from deriva_ml.execution import with_description  # noqa: F401  (re-exported for users editing this file)

asset_store = store(group="assets")

# REQUIRED: ``default_asset`` is used when no ``assets`` override is given.
asset_store([], name="default_asset")

# Alias for clarity in notebook configs.
asset_store([], name="no_assets")

# -----------------------------------------------------------------------------
# Add per-experiment asset configs below as you generate them.
# Examples (commented out — uncomment and replace RIDs after running):
# -----------------------------------------------------------------------------

# asset_store(["<rid_quick>", "<rid_extended>"], name="roc_quick_vs_extended")
#
# asset_store(
#     with_description(
#         ["<rid_quick>"],
#         "Pre-trained weights from cifar10_quick.",
#     ),
#     name="quick_weights",
# )
#
# asset_store(
#     with_description(
#         ["<rid_extended>"],
#         "Pre-trained weights from cifar10_extended.",
#     ),
#     name="extended_weights",
# )

# -----------------------------------------------------------------------------
# 2026-05-27 e2e Analyst arc: prediction CSVs from Developer's three runs.
#
#   Y0E — XYG (cifar10_default, TX0, 10 ep, seed=123)
#   YCP — YAP (cifar10_regularized, TX0+XEM, 10 ep, seed=2026)
#   XQC — XNE (cifar10_quick,    WD2, 3 ep, seed=42)
#
# Override the default roc_analysis config with this group:
#   uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \
#       assets=roc_e2e_2026_05_27
# -----------------------------------------------------------------------------
asset_store(
    with_description(
        ["Y0E", "YCP", "XQC"],
        "2026-05-27 e2e: prediction probabilities from XYG, YAP, XNE.",
    ),
    name="roc_e2e_2026_05_27",
)
