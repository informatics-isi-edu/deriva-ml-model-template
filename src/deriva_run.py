"""
Deriva ML Model Runner
======================

A standalone driver to execute machine learning models using hydra-zen for configuration.

This module uses the `run_model` function from deriva-ml to execute ML models
within a DerivaML execution context. The function handles:
- Connecting to the Deriva catalog
- Creating execution records with provenance tracking
- Multirun/sweep support with parent-child execution nesting
- Uploading results to the catalog

Usage
-----
Run from the command line with hydra configuration overrides:

    python deriva_run.py +experiment=my_experiment
    python deriva_run.py model_config.epochs=100 dry_run=True

The hydra configuration system allows you to:
- Override individual parameters on the command line
- Compose configurations from multiple groups
- Run parameter sweeps with -m/--multirun

Multirun Example
----------------
Running a parameter sweep automatically creates a parent execution that
groups all child executions:

    python deriva_run.py --multirun +experiment=cifar10_quick,cifar10_extended

This creates:
- Parent execution: "Multirun sweep: ..." with the sweep configuration
- Child execution 0: cifar10_quick (linked to parent, sequence=0)
- Child execution 1: cifar10_extended (linked to parent, sequence=1)

Customization
-------------
To adapt this template for a specific domain (e.g., EyeAI):
1. Import your domain class instead of DerivaML
2. Modify configs/base.py to use create_model_config(YourClass)
3. Modify the config imports to include your domain configs
"""

from hydra_zen import store, zen

from deriva_ml.execution import run_model


# =============================================================================
# Hydra-Zen Configuration Setup
# =============================================================================
# Load configuration modules from the configs package. This includes:
# - configs/base.py: Creates and registers DerivaModelConfig
# - configs/experiments.py: Defines experiment presets
# - Other config modules (datasets, assets, workflows, etc.)
#
# The main config (deriva_model) is created in configs/base.py so that
# experiments can properly inherit from it.

from configs import load_all_configs  # noqa: E402
load_all_configs()


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Finalize the hydra-zen store by adding all registered configs to hydra
    store.add_to_hydra_store()

    # Launch the hydra application. This will:
    # 1. Parse command-line arguments
    # 2. Compose the configuration from defaults and overrides
    # 3. Call run_model() with the resolved configuration
    #
    # Multirun execution nesting is handled inside run_model() by:
    # - Detecting multirun mode via HydraConfig
    # - Creating a parent execution to group all sweep jobs
    # - Linking each child execution to the parent with sequence ordering
    zen(run_model).hydra_main(
        config_name="deriva_model",
        version_base="1.3",
        config_path=None,
    )
