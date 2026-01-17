"""Configuration for ROC analysis notebook.

This module defines the hydra-zen configuration for the ROC curve analysis notebook.
The configuration inherits from BaseConfig and adds ROC-specific parameters.

Usage:
    In notebook (using hydra-zen configuration):

        from configs import load_all_configs
        from configs.roc_analysis import ROCAnalysisConfigBuilds
        from deriva_ml.execution import get_notebook_configuration

        load_all_configs()
        config = get_notebook_configuration(
            ROCAnalysisConfigBuilds,
            config_name="roc_analysis",
            overrides=["assets=roc_comparison_probabilities"],
        )

    From command line (using deriva-ml-run-notebook):

        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            --host localhost --catalog 45 \\
            assets=roc_comparison_probabilities

Configuration Groups:
    - deriva_ml: DerivaML connection settings (default_deriva, eye_ai, etc.)
    - assets: Asset RID lists for probability files (roc_quick_probabilities, etc.)
"""

from dataclasses import dataclass, field

from hydra_zen import builds, store

from deriva_ml.execution import BaseConfig, base_defaults


# ---------------------------------------------------------------------------
# ROC Analysis Configuration
# ---------------------------------------------------------------------------


@dataclass
class ROCAnalysisConfig(BaseConfig):
    """Configuration for ROC analysis notebook.

    Inherits standard DerivaML configuration fields from BaseConfig and adds
    ROC-specific parameters. The assets field is used to specify which
    prediction probability files to analyze.

    Attributes:
        deriva_ml: DerivaML connection configuration.
        assets: List of asset RIDs for prediction_probabilities.csv files.
        description: Human-readable description of this analysis run.

    Example:
        >>> config = get_notebook_configuration(
        ...     ROCAnalysisConfigBuilds,
        ...     config_name="roc_analysis",
        ...     overrides=["assets=roc_comparison_probabilities"],
        ... )
        >>> print(config.assets)  # ['42JE', '44KE']
    """

    # Note: All fields inherited from BaseConfig:
    # - deriva_ml: DerivaML connection config
    # - datasets: Dataset specs (not used for ROC analysis)
    # - assets: Asset RIDs to load (this is the main input for ROC analysis)
    # - dry_run: Skip catalog writes
    # - description: Run description
    pass


# ---------------------------------------------------------------------------
# Hydra-zen Configuration Builds
# ---------------------------------------------------------------------------

# Define defaults for ROC analysis notebook
# We only need deriva_ml and assets for this notebook
roc_analysis_defaults = [
    "_self_",
    {"deriva_ml": "default_deriva"},
    {"assets": "roc_comparison_probabilities"},  # Default to comparison of both experiments
]

# Build the configuration class with hydra-zen
ROCAnalysisConfigBuilds = builds(
    ROCAnalysisConfig,
    populate_full_signature=True,
    hydra_defaults=roc_analysis_defaults,
)

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------

roc_store = store(group="roc_analysis")
roc_store(ROCAnalysisConfigBuilds, name="roc_analysis")

# Also register as a top-level config for direct use
store(ROCAnalysisConfigBuilds, name="roc_analysis")
