"""Localhost notebook configs that pair with `dev/assets_localhost.py`.

These are the ROC-analysis notebook configurations that select the
catalog-46 asset entries. Use one at the CLI:

    uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
        --host localhost --catalog 46 \\
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
        "deriva_ml": "localhost_46",
        "assets": "roc_lr_sweep_localhost",
        "datasets": "no_datasets",
    },
    description=(
        "ROC analysis on localhost 46: learning-rate sweep "
        "(populated by Phase 4 of the e2e test)."
    ),
)


notebook_config(
    "roc_e2e_localhost",
    config_class=ROCAnalysisConfig,
    defaults={
        "deriva_ml": "localhost_46",
        "assets": "roc_e2e_localhost",
        "datasets": "no_datasets",
    },
    description=(
        "ROC analysis on localhost 46: quick (3 epochs) vs extended (50 epochs) "
        "(populated by Phase 2 + a follow-up run)."
    ),
)
