"""Configuration for ROC analysis notebook.

This module defines the configuration for the ROC curve analysis notebook.
The notebook uses the execution context with assets to download probability files.

Usage:
    # Run with default probability files
    deriva-ml-run-notebook notebooks/roc_analysis.ipynb --host localhost --catalog 45

    # Run with specific assets (probability files)
    deriva-ml-run-notebook notebooks/roc_analysis.ipynb \
        --host localhost --catalog 45 \
        -p assets '["3JSJ", "3KVC"]'

The assets parameter should contain RIDs for prediction_probabilities.csv files
from completed CIFAR-10 CNN executions.
"""

# This config module is kept for reference but the notebook now uses
# papermill parameters directly rather than hydra-zen configuration.
# See assets.py for pre-configured asset lists:
#   - multirun_quick_probabilities: ["3JSJ"]
#   - multirun_extended_probabilities: ["3KVC"]
#   - multirun_comparison_probabilities: ["3JSJ", "3KVC"]
