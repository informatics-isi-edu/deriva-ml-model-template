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
# [E2E-DROP] Modeler e2e outputs — catalog 2 (e2e-test-20260601), 2026-06-01.
#
# Three training executions, ALL within the leakage-safe small labeled family
# PJM = cifar10_small_labeled_split (train PJW / eval PK6, both stratified from
# F3T seed=42). Per the Curator handoff (findings/curator/), train and eval stay
# inside ONE family — never crossed with F3T/F38/F44 — so the held-out eval on
# PK6 is honest. Each run produced weights (.pt), a training log, and a
# per-image prediction-probability CSV; each recorded 100 Image_Classification
# feature rows on PK6 with a populated Confidence column.
#
#   Run            Execution  weights  log   prediction CSV
#   smoke (quick)  QK8        QN2      QN4   QN6   (3 epochs, lr 1e-3, batch 128)
#   A (regularized) QWA       QY4      QY6   QY8   (20 ep, dropout 0.25, wd 1e-4)
#   B (fast_lr)    R5C        R76      R78   R7A   (15 ep, lr 1e-2, no dropout)
# -----------------------------------------------------------------------------

# Prediction-probability CSVs (the surface ROC analysis consumes).
asset_store(
    with_description(
        ["QN6"],
        "Prediction probabilities from smoke run QK8 (cifar10_quick, 3 epochs) on PK6.",
    ),
    name="preds_smoke_quick",
)
asset_store(
    with_description(
        ["QY8"],
        "Prediction probabilities from Run A QWA (regularized, 20 epochs) on PK6.",
    ),
    name="preds_run_a_regularized",
)
asset_store(
    with_description(
        ["R7A"],
        "Prediction probabilities from Run B R5C (fast_lr 1e-2, 15 epochs) on PK6.",
    ),
    name="preds_run_b_fast_lr",
)

# Three-way ROC comparison — all predictions are on the same eval set (PK6),
# so the curves are directly comparable. Order: smoke, regularized, fast_lr.
asset_store(
    with_description(
        ["QN6", "QY8", "R7A"],
        "ROC comparison of the three Modeler e2e runs on PK6: smoke (QK8), "
        "regularized (QWA), fast_lr (R5C). All predictions on the same eval set.",
    ),
    name="roc_modeler_e2e_three_way",
)

# Trained weights (for test-only re-evaluation or fine-tuning).
asset_store(
    with_description(
        ["QY4"],
        "Trained weights from Run A QWA (regularized, 20 epochs).",
    ),
    name="weights_run_a_regularized",
)
asset_store(
    with_description(
        ["R76"],
        "Trained weights from Run B R5C (fast_lr 1e-2, 15 epochs).",
    ),
    name="weights_run_b_fast_lr",
)
