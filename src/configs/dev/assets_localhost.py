"""Localhost catalog 1337 asset RIDs (prediction CSVs from multirun children).

These are the per-experiment ``prediction_probabilities.csv`` outputs from
the lr_sweep multirun executed against catalog 1337. The ROC analysis
notebook reads these CSVs to compute per-class AUC and plot the curves.

Select on the CLI:

    uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
        --host localhost --catalog 1337 \\
        assets=roc_lr_sweep_localhost

The default ``assets.py`` configs in this repo ship empty (since asset
RIDs are produced by *executions*, not by the loader, and so are
catalog-specific). After running additional experiments, append more
``*_localhost`` entries here mapping name → list of CSV RIDs.
"""

from hydra_zen import store

asset_store = store(group="assets")

# -----------------------------------------------------------------------------
# Learning-rate sweep multirun (parent 4GCC; 4 children, 1 CSV per child).
# -----------------------------------------------------------------------------
# Children:
#   4GDW (LR=0.0001) -> 4GFE
#   4GQ2 (LR=0.001)  -> 4GRT
#   4H0E (LR=0.01)   -> 4H26
#   4H9T (LR=0.1)    -> 4HBJ

asset_store(
    ["4GFE", "4GRT", "4H26", "4HBJ"],
    name="roc_lr_sweep_localhost",
)
