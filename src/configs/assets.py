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
# [E2E-DROP] Modeler arc 2026-05-28 — Toronto-pair training executions
# -----------------------------------------------------------------------------
# Three training executions on the cifar10_toronto_pair (M16 train, M1G test,
# leakage-free per Curator tk-002). Each execution produced three assets:
# weights (.pt), training_log.txt, prediction_probabilities.csv on M1G.
#
#   Execution W76 — cifar10_toronto_quick   ( 3 ep / 32-64-128 / lr=1e-3 / bs=128)
#                   final test_acc=24.00%   weights=W92  log=W94  preds=W96
#   Execution XCE — cifar10_toronto_default (10 ep / 32-64-128 / lr=1e-3 / bs=64)
#                   final test_acc=37.82%   weights=XEA  log=XEC  preds=XEE
#   Execution YHP — cifar10_toronto_large   (20 ep / 64-128-256 / lr=1e-3 / bs=64)
#                   final test_acc=41.09%   weights=YKJ  log=YKM  preds=YKP
#
# Numbers above are the *emission-time* test accuracy printed by
# record_predictions; they are what a downstream consumer reproduces by
# joining the committed CSV (W96 / XEE / YKP) against the ground-truth
# Image_Classification feature, filtered to execution HSR (tk-001).

asset_store(
    with_description(
        ["W96", "XEE", "YKP"],
        "Toronto-pair prediction CSVs from Modeler arc: W76(quick,3ep) XCE(default,10ep) YHP(large,20ep). All evaluated on M1G test partition.",
    ),
    name="toronto_predictions",
)

asset_store(
    with_description(
        ["W92", "W94", "W96"],
        "Outputs of execution W76 (cifar10_toronto_quick): weights + log + prediction CSV on M1G.",
    ),
    name="toronto_quick_outputs",
)

asset_store(
    with_description(
        ["XEA", "XEC", "XEE"],
        "Outputs of execution XCE (cifar10_toronto_default): weights + log + prediction CSV on M1G.",
    ),
    name="toronto_default_outputs",
)

asset_store(
    with_description(
        ["YKJ", "YKM", "YKP"],
        "Outputs of execution YHP (cifar10_toronto_large): weights + log + prediction CSV on M1G.",
    ),
    name="toronto_large_outputs",
)
