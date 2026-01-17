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

        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            --host localhost --catalog 45 \\
            assets=roc_quick_probabilities

Configuration Groups:
    - deriva_ml: DerivaML connection settings (default_deriva, eye_ai, etc.)
    - assets: Asset RID lists for probability files (roc_quick_probabilities, etc.)
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


# Register the ROC analysis notebook configuration with custom parameters.
notebook_config(
    "roc_analysis",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_comparison_probabilities"},
    description="ROC curve analysis",
)
