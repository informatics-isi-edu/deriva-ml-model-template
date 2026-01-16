"""Configuration for ROC analysis notebook.

This module defines the configuration for the ROC curve analysis notebook.
It uses a simple dataclass with just the fields needed for ROC analysis.

Note: We don't inherit from BaseConfig here because BaseConfig contains
DerivaMLConfig (a Pydantic model) which is not compatible with OmegaConf's
structured configs. Instead, we define the fields we need directly and
use hydra_defaults to resolve the deriva_ml configuration group.
"""

from dataclasses import dataclass, field
from typing import Any

from hydra_zen import builds, store


@dataclass
class ROCAnalysisConfig:
    """Configuration for ROC analysis notebook.

    Attributes:
        deriva_ml: DerivaML connection configuration (resolved from deriva_ml group).
        execution_rids: List of execution RIDs to analyze. The notebook will
            download probability files from each execution and compute ROC curves.
    """
    deriva_ml: Any = None  # Will be resolved from hydra defaults
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
