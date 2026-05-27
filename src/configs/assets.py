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
# [E2E-DROP] 2026-05-27-d Modeler arc outputs (catalog 96, workflow XDG, WD2 split).
#
# Three differentiated training runs on cifar10_small_labeled_split (WD2):
#   - XDP (cifar10_quick): 3 epochs, 32->64ch.  Test acc 24%.  Smoke.
#   - XPR (default_model):  10 epochs, 32->64ch. Test acc 38%.  Moderate baseline.
#   - XZT (cifar10_extended): 50 epochs, 64->128ch + dropout 0.25 + wd 1e-4.
#       Test acc 41% (overfit: train acc 100% / test 41%).
#
# Output asset triples (weights / training_log / prediction_probabilities) per
# execution.  RIDs verified via direct deriva-ml + MCP cross-channel.
# -----------------------------------------------------------------------------

# Prediction-probability CSVs — feed the roc_analysis notebook's
# `roc_quick_vs_extended` config.  Order: quick, default, extended.
asset_store(
    with_description(
        ["XFM", "XRP", "Y1R"],
        "Prediction probabilities from the three WD2 runs "
        "(XDP/quick, XPR/default, XZT/extended). Order matches the "
        "3-run comparison the Analyst should plot.",
    ),
    name="roc_quick_vs_extended",
)

# Model weights from each run — for test-only / inference re-runs.
asset_store(
    with_description(
        ["XFG"],
        "Weights from XDP (cifar10_quick, 3 epochs, 32->64ch).",
    ),
    name="quick_weights",
)
asset_store(
    with_description(
        ["XRJ"],
        "Weights from XPR (default_model, 10 epochs, 32->64ch).",
    ),
    name="default_weights",
)
asset_store(
    with_description(
        ["Y1M"],
        "Weights from XZT (cifar10_extended, 50 epochs, 64->128ch, regularized).",
    ),
    name="extended_weights",
)
