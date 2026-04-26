"""Localhost catalog 1256 asset RIDs (prediction CSVs from multirun children).

These are the per-experiment ``prediction_probabilities.csv`` outputs from
the lr_sweep multirun executed against catalog 1256. The ROC analysis
notebook reads these CSVs to compute per-class AUC and plot the curves.

Select on the CLI:

    uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
        --host localhost --catalog 1256 \\
        assets=roc_lr_sweep_localhost

The default ``assets.py`` configs in this repo ship empty (since asset
RIDs are produced by *executions*, not by the loader, and so are
catalog-specific). After running additional experiments, append more
``*_localhost`` entries here mapping name → list of CSV RIDs.
"""

from hydra_zen import store

asset_store = store(group="assets")

# -----------------------------------------------------------------------------
# Learning-rate sweep multirun (parent 4HKJ; 4 children, 1 CSV per child).
# -----------------------------------------------------------------------------
# Children:
#   4HN0 (LR=0.0001) -> 4HPJ
#   4HY6 (LR=0.001)  -> 4HZT
#   4J7E (LR=0.01)   -> 4J92
#   4JGP (LR=0.1)    -> 4JJA

asset_store(
    ["4HPJ", "4HZT", "4J92", "4JJA"],
    name="roc_lr_sweep_localhost",
)
