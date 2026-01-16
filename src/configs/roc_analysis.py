"""Configuration for ROC analysis notebook.

This module defines the configuration for the ROC curve analysis notebook.
It inherits from BaseConfig and adds notebook-specific parameters.
"""

from dataclasses import dataclass, field

from hydra_zen import builds, store

from deriva_ml.execution import BaseConfig


@dataclass
class ROCAnalysisConfig(BaseConfig):
    """Configuration for ROC analysis notebook.

    Attributes:
        execution_rids: List of execution RIDs to analyze. The notebook will
            download probability files from each execution and compute ROC curves.
    """
    execution_rids: list[str] = field(default_factory=list)


# Notebook-specific defaults - minimal config, no datasets or assets needed
roc_defaults = [
    "_self_",
    {"deriva_ml": "default_deriva"},
]

# Create and register the config
ROCAnalysisConfigBuilds = builds(
    ROCAnalysisConfig,
    populate_full_signature=True,
    hydra_defaults=roc_defaults,
)

store(ROCAnalysisConfigBuilds, name="roc_analysis")

# Pre-configured analysis for multirun comparison experiments (catalog 45)
# cifar10_quick (3JRC): 3 epochs, 32→64 channels
# cifar10_extended (3KT0): 50 epochs, 64→128 channels
store(
    ROCAnalysisConfigBuilds,
    name="multirun_comparison",
    execution_rids=["3JRC", "3KT0"],
)
