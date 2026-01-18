"""Asset Configuration.

This module defines asset configurations for execution (model weights, checkpoints, etc.).

Configuration Group: assets
---------------------------
Assets are additional files needed beyond the dataset. They are specified as
lists of Resource IDs (RIDs) and automatically downloaded when the execution
is initialized.

Typical assets include:
- Pre-trained model weights
- Model checkpoints
- Configuration files
- Reference data files

REQUIRED: A configuration named "default_asset" must be defined.
This is used as the default (typically an empty list) when no override is specified.

Example usage:
    # Use default (no assets)
    uv run deriva-ml-run

    # Use specific assets
    uv run deriva-ml-run assets=multirun_quick_weights

Configuration Format:
    asset_store(
        with_description(
            ["RID1", "RID2"],
            "Description of what these assets are for",
        ),
        name="my_asset_config",
    )
"""

from hydra_zen import store
from deriva_ml.execution import with_description

# ---------------------------------------------------------------------------
# Asset Store
# ---------------------------------------------------------------------------
asset_store = store(group="assets")

# REQUIRED: default_asset - used when no assets are specified (typically empty)
asset_store(
    with_description([], "No assets - empty default configuration"),
    name="default_asset",
)

# =============================================================================
# Catalog 65: CIFAR-10 Multi-Experiment Assets (localhost, schema: cifar10)
# =============================================================================

# -----------------------------------------------------------------------------
# quick_vs_extended multirun (parent: 3WJA) - Small dataset comparison
# -----------------------------------------------------------------------------
# Compares quick training (3 epochs) vs extended training (50 epochs)
# on small labeled dataset (500 images)

asset_store(
    with_description(
        ["3WMG"],
        "Model weights (cifar10_cnn_weights.pt) from cifar10_quick: "
        "3 epochs, 32→64 channels, batch 128. Source: execution 3WKE.",
    ),
    name="multirun_quick_weights",
)

asset_store(
    with_description(
        ["3XPA"],
        "Model weights (cifar10_cnn_weights.pt) from cifar10_extended: "
        "50 epochs, 64→128 channels, dropout 0.25, weight decay 1e-4. Source: execution 3XN2.",
    ),
    name="multirun_extended_weights",
)

asset_store(
    with_description(
        ["3WMG", "3XPA"],
        "Both model weights from quick_vs_extended multirun for comparison analysis.",
    ),
    name="multirun_comparison_weights",
)

asset_store(
    with_description(
        ["3WMJ"],
        "Training log (training_log.txt) from cifar10_quick execution.",
    ),
    name="multirun_quick_log",
)

asset_store(
    with_description(
        ["3XPC"],
        "Training log (training_log.txt) from cifar10_extended execution.",
    ),
    name="multirun_extended_log",
)

asset_store(
    with_description(
        ["3WMM"],
        "Prediction probabilities (prediction_probabilities.csv) from cifar10_quick. "
        "Use for ROC analysis with labeled test data.",
    ),
    name="multirun_quick_probabilities",
)

asset_store(
    with_description(
        ["3XPE"],
        "Prediction probabilities (prediction_probabilities.csv) from cifar10_extended. "
        "Use for ROC analysis with labeled test data.",
    ),
    name="multirun_extended_probabilities",
)

asset_store(
    with_description(
        ["3WMM", "3XPE"],
        "Both prediction probability files from quick_vs_extended multirun. "
        "Use for comparative ROC analysis.",
    ),
    name="multirun_comparison_probabilities",
)

asset_store(
    with_description(
        ["3WMG", "3WJM"],
        "Complete asset set from cifar10_quick: model weights + hydra config.",
    ),
    name="multirun_quick_assets",
)

asset_store(
    with_description(
        ["3XPA", "3XN8"],
        "Complete asset set from cifar10_extended: model weights + hydra config.",
    ),
    name="multirun_extended_assets",
)

# -----------------------------------------------------------------------------
# quick_vs_extended_full multirun (parent: 3YQM) - Full dataset (10,000 images)
# -----------------------------------------------------------------------------
# Same comparison but on the full CIFAR-10 dataset

asset_store(
    with_description(
        ["3YST"],
        "Model weights from cifar10_quick_full: 3 epochs on full dataset (10,000 images). "
        "Source: execution 3YRR.",
    ),
    name="full_quick_weights",
)

asset_store(
    with_description(
        ["48MW"],
        "Model weights from cifar10_extended_full: 50 epochs on full dataset. "
        "Source: execution 48KM.",
    ),
    name="full_extended_weights",
)

asset_store(
    with_description(
        ["3YST", "48MW"],
        "Both model weights from full dataset quick_vs_extended comparison.",
    ),
    name="full_comparison_weights",
)

asset_store(
    with_description(
        ["3YSW"],
        "Training log from cifar10_quick_full execution.",
    ),
    name="full_quick_log",
)

asset_store(
    with_description(
        ["48MY"],
        "Training log from cifar10_extended_full execution.",
    ),
    name="full_extended_log",
)

asset_store(
    with_description(
        ["3YSY"],
        "Prediction probabilities from cifar10_quick_full for ROC analysis.",
    ),
    name="full_quick_probabilities",
)

asset_store(
    with_description(
        ["48N0"],
        "Prediction probabilities from cifar10_extended_full for ROC analysis.",
    ),
    name="full_extended_probabilities",
)

asset_store(
    with_description(
        ["3YSY", "48N0"],
        "Both prediction probability files from full dataset comparison for ROC analysis.",
    ),
    name="full_comparison_probabilities",
)

# -----------------------------------------------------------------------------
# lr_sweep multirun (parent: 4JFE) - Learning rate hyperparameter sweep
# -----------------------------------------------------------------------------
# Tests learning rates: 0.0001, 0.001, 0.01, 0.1
# All use: 10 epochs, 32→64 channels, batch 128

asset_store(
    with_description(
        ["4JHM"],
        "Model weights from lr=0.0001 experiment. Source: execution 4JGJ.",
    ),
    name="lr_sweep_0001_weights",
)

asset_store(
    with_description(
        ["4KKE"],
        "Model weights from lr=0.001 experiment (default lr). Source: execution 4KJ6.",
    ),
    name="lr_sweep_001_weights",
)

asset_store(
    with_description(
        ["4MN8"],
        "Model weights from lr=0.01 experiment. Source: execution 4MM0.",
    ),
    name="lr_sweep_01_weights",
)

asset_store(
    with_description(
        ["4NQ2"],
        "Model weights from lr=0.1 experiment (high lr). Source: execution 4NNT.",
    ),
    name="lr_sweep_1_weights",
)

asset_store(
    with_description(
        ["4JHM", "4KKE", "4MN8", "4NQ2"],
        "All model weights from learning rate sweep (lr=0.0001, 0.001, 0.01, 0.1). "
        "Use for comparing effect of learning rate on model performance.",
    ),
    name="lr_sweep_all_weights",
)

asset_store(
    with_description(
        ["4JHR"],
        "Prediction probabilities from lr=0.0001 experiment for ROC analysis.",
    ),
    name="lr_sweep_0001_probabilities",
)

asset_store(
    with_description(
        ["4KKJ"],
        "Prediction probabilities from lr=0.001 experiment for ROC analysis.",
    ),
    name="lr_sweep_001_probabilities",
)

asset_store(
    with_description(
        ["4MNC"],
        "Prediction probabilities from lr=0.01 experiment for ROC analysis.",
    ),
    name="lr_sweep_01_probabilities",
)

asset_store(
    with_description(
        ["4NQ6"],
        "Prediction probabilities from lr=0.1 experiment for ROC analysis.",
    ),
    name="lr_sweep_1_probabilities",
)

asset_store(
    with_description(
        ["4JHR", "4KKJ", "4MNC", "4NQ6"],
        "All prediction probabilities from learning rate sweep. "
        "Use for ROC analysis comparing learning rate effects.",
    ),
    name="lr_sweep_all_probabilities",
)

# -----------------------------------------------------------------------------
# epoch_sweep multirun (parent: 4PRC) - Training duration sweep
# -----------------------------------------------------------------------------
# Tests epochs: 5, 10, 25, 50
# All use: 64→128 channels, 256 hidden, dropout 0.25, weight decay 1e-4

asset_store(
    with_description(
        ["4PTJ"],
        "Model weights from epochs=5 experiment. Source: execution 4PSG.",
    ),
    name="epoch_sweep_5_weights",
)

asset_store(
    with_description(
        ["4QWC"],
        "Model weights from epochs=10 experiment. Source: execution 4QV4.",
    ),
    name="epoch_sweep_10_weights",
)

asset_store(
    with_description(
        ["4RY6"],
        "Model weights from epochs=25 experiment. Source: execution 4RWY.",
    ),
    name="epoch_sweep_25_weights",
)

asset_store(
    with_description(
        ["4T00"],
        "Model weights from epochs=50 experiment. Source: execution 4SYR.",
    ),
    name="epoch_sweep_50_weights",
)

asset_store(
    with_description(
        ["4PTJ", "4QWC", "4RY6", "4T00"],
        "All model weights from epoch sweep (5, 10, 25, 50 epochs). "
        "Use for comparing effect of training duration on model quality.",
    ),
    name="epoch_sweep_all_weights",
)

asset_store(
    with_description(
        ["4PTP"],
        "Prediction probabilities from epochs=5 experiment for ROC analysis.",
    ),
    name="epoch_sweep_5_probabilities",
)

asset_store(
    with_description(
        ["4QWG"],
        "Prediction probabilities from epochs=10 experiment for ROC analysis.",
    ),
    name="epoch_sweep_10_probabilities",
)

asset_store(
    with_description(
        ["4RYA"],
        "Prediction probabilities from epochs=25 experiment for ROC analysis.",
    ),
    name="epoch_sweep_25_probabilities",
)

asset_store(
    with_description(
        ["4T04"],
        "Prediction probabilities from epochs=50 experiment for ROC analysis.",
    ),
    name="epoch_sweep_50_probabilities",
)

asset_store(
    with_description(
        ["4PTP", "4QWG", "4RYA", "4T04"],
        "All prediction probabilities from epoch sweep. "
        "Use for ROC analysis comparing training duration effects.",
    ),
    name="epoch_sweep_all_probabilities",
)

# -----------------------------------------------------------------------------
# lr_batch_grid multirun (parent: 4V1A) - Learning rate and batch size grid search
# -----------------------------------------------------------------------------
# 2x2 grid: lr in [0.001, 0.01], batch_size in [64, 128]
# All use: 10 epochs, 32→64 channels

asset_store(
    with_description(
        ["4V3G"],
        "Model weights from lr=0.001, batch=64 experiment. Source: execution 4V2E.",
    ),
    name="grid_lr001_batch64_weights",
)

asset_store(
    with_description(
        ["4W5A"],
        "Model weights from lr=0.001, batch=128 experiment. Source: execution 4W42.",
    ),
    name="grid_lr001_batch128_weights",
)

asset_store(
    with_description(
        ["4X74"],
        "Model weights from lr=0.01, batch=64 experiment. Source: execution 4X5W.",
    ),
    name="grid_lr01_batch64_weights",
)

asset_store(
    with_description(
        ["4Y8Y"],
        "Model weights from lr=0.01, batch=128 experiment. Source: execution 4Y7P.",
    ),
    name="grid_lr01_batch128_weights",
)

asset_store(
    with_description(
        ["4V3G", "4W5A", "4X74", "4Y8Y"],
        "All model weights from LR x batch size grid search (2x2 grid). "
        "Use for analyzing interaction between learning rate and batch size.",
    ),
    name="grid_all_weights",
)

asset_store(
    with_description(
        ["4V3M"],
        "Prediction probabilities from lr=0.001, batch=64 for ROC analysis.",
    ),
    name="grid_lr001_batch64_probabilities",
)

asset_store(
    with_description(
        ["4W5E"],
        "Prediction probabilities from lr=0.001, batch=128 for ROC analysis.",
    ),
    name="grid_lr001_batch128_probabilities",
)

asset_store(
    with_description(
        ["4X78"],
        "Prediction probabilities from lr=0.01, batch=64 for ROC analysis.",
    ),
    name="grid_lr01_batch64_probabilities",
)

asset_store(
    with_description(
        ["4Y92"],
        "Prediction probabilities from lr=0.01, batch=128 for ROC analysis.",
    ),
    name="grid_lr01_batch128_probabilities",
)

asset_store(
    with_description(
        ["4V3M", "4W5E", "4X78", "4Y92"],
        "All prediction probabilities from LR x batch size grid search. "
        "Use for ROC analysis of hyperparameter interactions.",
    ),
    name="grid_all_probabilities",
)

# -----------------------------------------------------------------------------
# cifar10_test_only experiment (execution: 4ZA8) - Evaluation only
# -----------------------------------------------------------------------------
# Uses multirun_quick_weights as input, runs evaluation on test set without training

asset_store(
    with_description(
        ["4ZBG"],
        "Evaluation results (evaluation_results.txt) from test-only experiment. "
        "Shows performance of quick model on test set without retraining.",
    ),
    name="test_only_evaluation_results",
)

# =============================================================================
# Labeled Dataset Experiments (for ROC analysis)
# =============================================================================
# These experiments use labeled split datasets where both train and test
# partitions have ground truth labels, enabling proper ROC analysis.
# The test set has ground truth labels (unlike the Kaggle test set).

# -----------------------------------------------------------------------------
# quick_vs_extended_labeled multirun (parent: 50DE) - Small labeled dataset
# -----------------------------------------------------------------------------

asset_store(
    with_description(
        ["50FR"],
        "Prediction probabilities from cifar10_quick on labeled test data. "
        "Has ground truth for ROC curve generation. Source: execution 50EJ.",
    ),
    name="labeled_quick_probabilities",
)

asset_store(
    with_description(
        ["50RJ"],
        "Prediction probabilities from cifar10_extended on labeled test data. "
        "Has ground truth for ROC curve generation. Source: execution 50Q6.",
    ),
    name="labeled_extended_probabilities",
)

asset_store(
    with_description(
        ["50FR", "50RJ"],
        "Both probability files from labeled quick_vs_extended comparison. "
        "Use for ROC analysis comparing quick vs extended training.",
    ),
    name="labeled_comparison_probabilities",
)

# -----------------------------------------------------------------------------
# lr_sweep_labeled multirun (parent: 510R) - Learning rate sweep with labeled data
# -----------------------------------------------------------------------------

asset_store(
    with_description(
        ["5132"],
        "Prediction probabilities from lr=0.0001 on labeled test data. Source: 511W.",
    ),
    name="labeled_lr_sweep_0001_probabilities",
)

asset_store(
    with_description(
        ["51BW"],
        "Prediction probabilities from lr=0.001 on labeled test data. Source: 51AG.",
    ),
    name="labeled_lr_sweep_001_probabilities",
)

asset_store(
    with_description(
        ["51MP"],
        "Prediction probabilities from lr=0.01 on labeled test data. Source: 51KA.",
    ),
    name="labeled_lr_sweep_01_probabilities",
)

asset_store(
    with_description(
        ["51XG"],
        "Prediction probabilities from lr=0.1 on labeled test data. Source: 51W4.",
    ),
    name="labeled_lr_sweep_1_probabilities",
)

asset_store(
    with_description(
        ["5132", "51BW", "51MP", "51XG"],
        "All prediction probabilities from labeled learning rate sweep. "
        "Use for ROC analysis of learning rate effects with ground truth labels.",
    ),
    name="labeled_lr_sweep_all_probabilities",
)

# -----------------------------------------------------------------------------
# epoch_sweep_labeled multirun (parent: 525P) - Epoch sweep with labeled data
# -----------------------------------------------------------------------------

asset_store(
    with_description(
        ["5280"],
        "Prediction probabilities from epochs=5 on labeled test data. Source: 526T.",
    ),
    name="labeled_epoch_sweep_5_probabilities",
)

asset_store(
    with_description(
        ["52GT"],
        "Prediction probabilities from epochs=10 on labeled test data. Source: 52FE.",
    ),
    name="labeled_epoch_sweep_10_probabilities",
)

asset_store(
    with_description(
        ["52SM"],
        "Prediction probabilities from epochs=25 on labeled test data. Source: 52R8.",
    ),
    name="labeled_epoch_sweep_25_probabilities",
)

asset_store(
    with_description(
        ["532E"],
        "Prediction probabilities from epochs=50 on labeled test data. Source: 5312.",
    ),
    name="labeled_epoch_sweep_50_probabilities",
)

asset_store(
    with_description(
        ["5280", "52GT", "52SM", "532E"],
        "All prediction probabilities from labeled epoch sweep. "
        "Use for ROC analysis of training duration effects with ground truth labels.",
    ),
    name="labeled_epoch_sweep_all_probabilities",
)

# -----------------------------------------------------------------------------
# lr_batch_grid_labeled multirun (parent: 53AM) - Grid search with labeled data
# -----------------------------------------------------------------------------

asset_store(
    with_description(
        ["53CY"],
        "Prediction probabilities from lr=0.001, batch=64 on labeled test data. Source: 53BR.",
    ),
    name="labeled_grid_lr001_batch64_probabilities",
)

asset_store(
    with_description(
        ["53NR"],
        "Prediction probabilities from lr=0.001, batch=128 on labeled test data. Source: 53MC.",
    ),
    name="labeled_grid_lr001_batch128_probabilities",
)

asset_store(
    with_description(
        ["53YJ"],
        "Prediction probabilities from lr=0.01, batch=64 on labeled test data. Source: 53X6.",
    ),
    name="labeled_grid_lr01_batch64_probabilities",
)

asset_store(
    with_description(
        ["547C"],
        "Prediction probabilities from lr=0.01, batch=128 on labeled test data. Source: 5460.",
    ),
    name="labeled_grid_lr01_batch128_probabilities",
)

asset_store(
    with_description(
        ["53CY", "53NR", "53YJ", "547C"],
        "All prediction probabilities from labeled LR x batch grid search. "
        "Use for ROC analysis of hyperparameter interactions with ground truth labels.",
    ),
    name="labeled_grid_all_probabilities",
)

# =============================================================================
# ROC Analysis Asset Configurations
# =============================================================================
# Named configurations specifically for the ROC analysis notebook.
# These point to the labeled dataset experiments for proper ground truth matching.

asset_store(
    with_description(
        ["50FR", "50RJ"],
        "ROC analysis: Quick vs Extended training comparison on labeled data. "
        "Compare 3-epoch vs 50-epoch training with ground truth test labels.",
    ),
    name="roc_quick_vs_extended",
)

asset_store(
    with_description(
        ["5132", "51BW", "51MP", "51XG"],
        "ROC analysis: Learning rate sweep (0.0001, 0.001, 0.01, 0.1). "
        "Compare effect of learning rate on classification performance.",
    ),
    name="roc_lr_sweep",
)

asset_store(
    with_description(
        ["5280", "52GT", "52SM", "532E"],
        "ROC analysis: Epoch sweep (5, 10, 25, 50 epochs). "
        "Compare effect of training duration on classification performance.",
    ),
    name="roc_epoch_sweep",
)

asset_store(
    with_description(
        ["53CY", "53NR", "53YJ", "547C"],
        "ROC analysis: LR x Batch size grid search (2x2 grid). "
        "Compare interaction between learning rate and batch size on performance.",
    ),
    name="roc_lr_batch_grid",
)
