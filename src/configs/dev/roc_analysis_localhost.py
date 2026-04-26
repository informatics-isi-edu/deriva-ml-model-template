"""Localhost notebook configs that pair with `dev/assets_localhost.py`.

These are the ROC-analysis notebook configurations that select the
catalog-1337 asset entries. Use one at the CLI:

    uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
        --host localhost --catalog 1337 \\
        --config-name roc_lr_sweep_localhost

The notebook itself reads its config via
``ml, execution, config = run_notebook("roc_lr_sweep_localhost")``.
"""

from configs.roc_analysis import ROCAnalysisConfig
from deriva_ml.execution import notebook_config


notebook_config(
    "roc_lr_sweep_localhost",
    config_class=ROCAnalysisConfig,
    defaults={
        "deriva_ml": "localhost_1407",
        "assets": "roc_lr_sweep_localhost",
        "datasets": "no_datasets",
    },
    description=(
        "ROC analysis on localhost 1407: learning-rate sweep "
        "(0.0001, 0.001, 0.01, 0.1) — 4 prediction CSVs from multirun parent 4HKJ."
    ),
)


notebook_config(
    "roc_e2e_localhost",
    config_class=ROCAnalysisConfig,
    defaults={
        "deriva_ml": "localhost_1407",
        "assets": "roc_e2e_localhost",
        "datasets": "no_datasets",
    },
    description=(
        "ROC analysis on localhost 1407: quick (3 epochs) vs extended (50 epochs) "
        "from multirun parent 804 — 2 prediction CSVs (836, 87J)."
    ),
)
