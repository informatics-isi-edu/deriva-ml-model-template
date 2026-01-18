"""Configuration for ROC analysis notebook.

This module defines the hydra-zen configuration for the ROC curve analysis notebook.

Usage:
    In notebook:

        from deriva_ml.execution import run_notebook

        ml, execution, config = run_notebook("roc_analysis")
        # Ready to use!
        # - config.assets: probability file RIDs
        # - config.show_per_class: whether to show individual class curves

    From command line:

        # Run with specific configuration
        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            --host localhost --catalog 65 \\
            --config roc_quick_vs_extended

        # Or override assets directly
        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            assets=roc_lr_sweep

Available Configurations:
    - roc_analysis: Default (quick vs extended on small dataset)
    - roc_quick_vs_extended: Compare quick vs extended training (small dataset)
    - roc_full_quick_vs_extended: Compare quick vs extended training (full dataset)
    - roc_lr_sweep: Compare learning rate sweep (0.0001, 0.001, 0.01, 0.1)
    - roc_epoch_sweep: Compare epoch sweep (5, 10, 25, 50 epochs)
    - roc_lr_batch_grid: Compare LR x batch size grid (2x2)

Configuration Groups:
    - deriva_ml: DerivaML connection settings (default_deriva, eye_ai, etc.)
    - assets: Asset RID lists for probability files
"""

from dataclasses import dataclass

from deriva_ml.execution import BaseConfig, notebook_config


@dataclass
class ROCAnalysisConfig(BaseConfig):
    """Configuration for ROC analysis notebook.

    Attributes:
        show_per_class: If True, plot individual ROC curves for each class.
            If False, only show micro/macro averaged curves.
        confidence_threshold: Minimum confidence threshold for predictions
            to be included in the analysis (0.0 to 1.0).
    """

    show_per_class: bool = True
    confidence_threshold: float = 0.0


# =============================================================================
# ROC Analysis Notebook Configurations
# =============================================================================

# Default: Use test-only prediction probabilities
notebook_config(
    "roc_analysis",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_test_only"},
    description="ROC curve analysis (default: test-only predictions)",
)

# -----------------------------------------------------------------------------
# Model Comparison Configurations
# -----------------------------------------------------------------------------
# NOTE: To run these configs with full ROC comparison, first run test-only mode
# on each trained model to generate prediction_probabilities.csv files, then
# update assets.py with the corresponding asset RIDs.

notebook_config(
    "roc_quick_vs_extended",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_test_only"},  # TODO: Update when comparison probabilities exist
    description="ROC analysis: quick vs extended training (needs prediction files)",
)

notebook_config(
    "roc_full_quick_vs_extended",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_test_only"},  # TODO: Update when comparison probabilities exist
    description="ROC analysis: quick vs extended training (full dataset)",
)

notebook_config(
    "roc_lr_sweep",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_test_only"},  # TODO: Update when comparison probabilities exist
    description="ROC analysis: learning rate sweep (needs prediction files)",
)

notebook_config(
    "roc_epoch_sweep",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_test_only"},  # TODO: Update when comparison probabilities exist
    description="ROC analysis: epoch sweep (needs prediction files)",
)

notebook_config(
    "roc_lr_batch_grid",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_test_only"},  # TODO: Update when comparison probabilities exist
    description="ROC analysis: LR x batch size grid (needs prediction files)",
)
