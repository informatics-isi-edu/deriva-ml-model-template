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
# [E2E-DROP] 2026-05-27-e Modeler arc — Family A (K0M, K0W train / K16 test).
# Three prediction CSVs from three model configs trained on the same data,
# so the Analyst can compare predictions directly against a genuinely
# held-out test partition (K16 ⊆ Toronto test_batch). See tacit-knowledge.md
# tk-004 for the rationale (and tk-002 for why Family A is the right choice
# for ROC analysis here).
# -----------------------------------------------------------------------------

asset_store(
    with_description(
        ["Y1M", "Z3P", "105R"],
        "Modeler arc 2026-05-27-e: three prediction CSVs on Family A K16 "
        "test partition. Y1M=cifar10_quick/3ep (XZP, 25.20%), "
        "Z3P=default_model/10ep (Z1R, 36.00%), "
        "105R=cifar10_large/20ep (103T, 36.80%, overfits).",
    ),
    name="modeler_familyA_triplet",
)

# Individual per-run weights — handy for the Analyst's test-only inference runs
# (model_config=cifar10_test_only + assets=<one_of_these>).
asset_store(
    with_description(
        ["Y1G"],
        "Weights from XZP — cifar10_quick (3 epochs, 32→64) on K0M.",
    ),
    name="modeler_quick_weights",
)
asset_store(
    with_description(
        ["Z3J"],
        "Weights from Z1R — default_model (10 epochs, 32→64) on K0M.",
    ),
    name="modeler_default_weights",
)
asset_store(
    with_description(
        ["105M"],
        "Weights from 103T — cifar10_large (20 epochs, 64→128) on K0M.",
    ),
    name="modeler_large_weights",
)
