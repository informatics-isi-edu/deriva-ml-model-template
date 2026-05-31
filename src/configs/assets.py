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
# Outputs from the Modeler arc: clean Toronto-split holdout runs (catalog 168).
#
# Two training executions trained on the F2T Training partition and recorded
# final-epoch predictions on the disjoint F34 held-out Testing partition
# (F2T n F34 = 0 — a genuine held-out metric; see tacit-knowledge.md tk-004).
#
#   RM8  cifar10_quick_toronto  (3 epochs, 32->64 ch)  -> F34 held-out acc 27.64%
#     RP2 weights, RP4 training_log, RP6 prediction_probabilities.csv
#   SSE  cifar10_large_toronto  (20 epochs, 64->128 ch) -> F34 held-out acc 37.64%
#     SV8 weights, SVA training_log, SVC prediction_probabilities.csv
#
# The two prediction CSVs (RP6, SVC) are the join targets for ROC / accuracy
# analysis against the Image_Classification ground-truth feature on F34. The
# weights (RP2, SV8) are reusable as test_only / fine-tune checkpoints.
# -----------------------------------------------------------------------------

# Prediction CSVs from both Toronto runs — the ROC-analysis input set.
asset_store(  # [E2E-DROP] catalog 168
    with_description(
        ["RP6", "SVC"],
        "F34 held-out prediction CSVs: cifar10_quick_toronto (RP6) and "
        "cifar10_large_toronto (SVC). Join against Image_Classification "
        "ground truth on F34 for ROC / accuracy comparison.",
    ),
    name="roc_quick_vs_large_toronto",
)

# Per-run weights (reusable as test_only checkpoints).
asset_store(  # [E2E-DROP] catalog 168
    with_description(
        ["RP2"],
        "Weights from cifar10_quick_toronto (exec RM8): 3 epochs, 32->64 ch, "
        "F34 held-out accuracy 27.64%.",
    ),
    name="quick_toronto_weights",
)

asset_store(  # [E2E-DROP] catalog 168
    with_description(
        ["SV8"],
        "Weights from cifar10_large_toronto (exec SSE): 20 epochs, 64->128 ch, "
        "F34 held-out accuracy 37.64% (overfit; train_acc reached 100%).",
    ),
    name="large_toronto_weights",
)
