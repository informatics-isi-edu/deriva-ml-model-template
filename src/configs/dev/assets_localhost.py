"""Localhost catalog 46 asset RIDs (prediction CSVs from multirun children).

Asset RIDs are produced by *executions*, not by the loader, and so are
populated only after the corresponding experiments have run. Phase 1 of
the e2e test creates the catalog and stops; Phase 2 runs the quick
training (populates `roc_e2e_localhost`); Phase 4 runs the lr_sweep
multirun (populates `roc_lr_sweep_localhost`). Until those phases run,
the entries below are empty placeholders.

Select on the CLI:

    uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
        --host localhost --catalog 46 \\
        assets=roc_lr_sweep_localhost
"""

from hydra_zen import store

asset_store = store(group="assets")

# -----------------------------------------------------------------------------
# Learning-rate sweep multirun on catalog 46.
# Populated by Phase 4. Children + their prediction CSVs go here as
# [child_csv_rid_1, child_csv_rid_2, ...] once known.
# -----------------------------------------------------------------------------

asset_store(
    [],
    name="roc_lr_sweep_localhost",
)

# -----------------------------------------------------------------------------
# E2E quick_vs_extended multirun on catalog 46.
# Populated by Phase 4 (multirun parent CTA -> children CVP cifar10_quick +
# D14 cifar10_extended). The two prediction CSVs let the ROC notebook
# compare the 3-epoch and 50-epoch architectures on the same test set.
# -----------------------------------------------------------------------------

asset_store(
    ["CXA", "D2R"],
    name="roc_e2e_localhost",
)
