"""Model Configuration (CIFAR-10 CNN).

This module defines model configurations for the CIFAR-10 2-layer CNN.

Configuration Group: model_config
---------------------------------
Model configurations define the hyperparameters and settings for your ML model.
Each configuration is a set of parameters that can be selected at runtime.

REQUIRED: A configuration named "default_model" must be defined.
This is used as the default model configuration when no override is specified.

All model parameters are configurable via Hydra:
- Architecture: conv1_channels, conv2_channels, hidden_size, dropout_rate
- Training: learning_rate, epochs, batch_size, weight_decay

Example usage:
    # Run with default config
    uv run src/deriva_run.py

    # Run with a specific model config
    uv run src/deriva_run.py model_config=cifar10_extended

    # Override specific parameters
    uv run src/deriva_run.py model_config.epochs=50 model_config.learning_rate=0.01
"""
from __future__ import annotations

from hydra_zen import builds, store

from models.cifar10_cnn import cifar10_cnn

# Build the base CIFAR-10 CNN configuration.
# All parameters have sensible defaults for a simple training run.
Cifar10CNNConfig = builds(
    cifar10_cnn,
    # Architecture parameters
    conv1_channels=32,
    conv2_channels=64,
    hidden_size=128,
    dropout_rate=0.0,
    # Training parameters
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    weight_decay=0.0,
    # Hydra-zen settings
    populate_full_signature=True,
    zen_partial=True,  # Execution context added later
)

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "model_config" must match the parameter name in BaseConfig.

model_store = store(group="model_config")

# REQUIRED: default_model - used when no model config is specified
model_store(Cifar10CNNConfig, name="default_model")

# Quick training - fewer epochs for testing
model_store(
    Cifar10CNNConfig,
    name="cifar10_quick",
    epochs=3,
    batch_size=128,
)

# Larger model - more capacity
model_store(
    Cifar10CNNConfig,
    name="cifar10_large",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    epochs=20,
)

# With dropout for regularization
model_store(
    Cifar10CNNConfig,
    name="cifar10_regularized",
    dropout_rate=0.25,
    weight_decay=1e-4,
    epochs=20,
)

# Fast learning rate - may converge faster but less stable
model_store(
    Cifar10CNNConfig,
    name="cifar10_fast_lr",
    learning_rate=1e-2,
    epochs=15,
)

# Slow learning rate - more stable, may need more epochs
model_store(
    Cifar10CNNConfig,
    name="cifar10_slow_lr",
    learning_rate=1e-4,
    epochs=30,
)

# Extended training - for best accuracy
model_store(
    Cifar10CNNConfig,
    name="cifar10_extended",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    dropout_rate=0.25,
    weight_decay=1e-4,
    learning_rate=1e-3,
    epochs=50,
)

# Test-only mode - load weights and run evaluation without training
# Use with assets=cifar10_small_experiment_weights and a testing dataset
model_store(
    Cifar10CNNConfig,
    name="cifar10_test_only",
    test_only=True,
    weights_filename="cifar10_cnn_weights.pt",
)

# ---------------------------------------------------------------------------
# Hyperparameter Sweep Base Configs
# ---------------------------------------------------------------------------
# These configs are designed for multirun sweeps where specific parameters
# will be overridden via command line (e.g., model_config.learning_rate=0.001,0.01)

# Learning rate sweep base - 10 epochs, standard architecture
# Override learning_rate on command line
model_store(
    Cifar10CNNConfig,
    name="cifar10_lr_sweep",
    epochs=10,
    batch_size=128,
    # learning_rate will be overridden in multirun
)

# Epoch sweep base - extended architecture with regularization
# Override epochs on command line
model_store(
    Cifar10CNNConfig,
    name="cifar10_epoch_sweep",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    dropout_rate=0.25,
    weight_decay=1e-4,
    batch_size=64,
    # epochs will be overridden in multirun
)
