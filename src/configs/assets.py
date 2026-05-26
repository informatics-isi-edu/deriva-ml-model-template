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
# E2E catalog 18 (e2e-test-20260526), 2026-05-26 — Analyst arc.
# Prediction probability CSV RIDs from Developer arc (tk-004 / handoff).
# All 6 viable training executions tested on the same 50-image CSA partition;
# apples-to-apples comparison across test_acc, AUC, confusion matrices.
# -----------------------------------------------------------------------------

# Quick (DYC) vs Extended (E4A) — controlled architecture/epochs comparison
# on the same CRR training data.
asset_store(
    with_description(
        ["E0A", "E68"],
        "Predictions from quick (DYC, 3 ep) and extended (E4A, 50 ep) runs on CRR.",
    ),
    name="roc_quick_vs_extended",
)

# Learning-rate sweep (4 children of EA8): lr ∈ {1e-4, 1e-3, 1e-2, 1e-1}.
asset_store(
    with_description(
        ["EE0", "EM0", "ET0", "F00"],
        "Predictions from lr_sweep children EC0/EJ0/ER0/EY0 (lr=1e-4, 1e-3, 1e-2, 1e-1).",
    ),
    name="roc_lr_sweep",
)

# All 6 viable Developer executions (DYC + E4A + lr_sweep × 4).
# Use this for an across-the-board ranking; F40 deliberately excluded
# (degenerate, no predictions — finding developer/01).
asset_store(
    with_description(
        ["E0A", "E68", "EE0", "EM0", "ET0", "F00"],
        "All 6 viable training-run predictions on CSA test set "
        "(DYC quick, E4A extended, lr_sweep × 4).",
    ),
    name="roc_all_six",
)
