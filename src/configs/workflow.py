"""Workflow Configuration.

This module defines workflow configurations for DerivaML executions.

Configuration Group: workflow
-----------------------------
A Workflow represents a computational pipeline and its metadata. It describes
what the code does (name, description, type) and automatically captures Git
repository information (URL, checksum, version) for provenance tracking.

REQUIRED: A configuration named "default_workflow" must be defined.
This is used as the default workflow when no override is specified.

Example usage:
    # Use default workflow
    uv run src/deriva_run.py

    # Use a specific workflow
    uv run src/deriva_run.py workflow=cifar10_cnn
"""

from hydra_zen import store, builds

from deriva_ml.execution import Workflow

# ---------------------------------------------------------------------------
# Workflow Configurations
# ---------------------------------------------------------------------------

# CIFAR-10 CNN workflow - default for this template
Cifar10CNNWorkflow = builds(
    Workflow,
    name="CIFAR-10 2-Layer CNN",
    workflow_type="Image Classification",
    description="""
Train a 2-layer convolutional neural network on CIFAR-10 image data.

## Architecture
- **Conv Layer 1**: 3 → 32 channels, 3×3 kernel, ReLU, MaxPool 2×2
- **Conv Layer 2**: 32 → 64 channels, 3×3 kernel, ReLU, MaxPool 2×2
- **FC Layer**: 64×8×8 → 128 hidden units → 10 classes

## Features
- Configurable architecture (channel sizes, hidden units, dropout)
- Training with Adam optimizer and cross-entropy loss
- Automatic data loading from DerivaML datasets via `restructure_assets()`
- Outputs: model weights (`.pt`) and training log

## Expected Performance
~60-70% test accuracy with default parameters on CIFAR-10.
""".strip(),
    populate_full_signature=True,
)


# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------

workflow_store = store(group="workflow")

# REQUIRED: default_workflow - used when no workflow is specified
workflow_store(Cifar10CNNWorkflow, name="default_workflow")

# Additional workflow configurations
workflow_store(Cifar10CNNWorkflow, name="cifar10_cnn")
