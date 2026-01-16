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
2. Use create_model_config(YourClass) to create the config
3. Modify the config imports to include your domain configs
"""

from hydra_zen import store, zen

from deriva_ml import DerivaML
from deriva_ml.execution import run_model, create_model_config


# =============================================================================
# Hydra-Zen Configuration Setup
# =============================================================================
# This section configures the hydra-zen command-line interface. The
# create_model_config() function creates a hydra-zen builds() for run_model
# with the specified DerivaML class.

# Create the main configuration schema for this application.
# For domain-specific classes (e.g., EyeAI), replace DerivaML with your class:
#   deriva_model = create_model_config(EyeAI, description="EyeAI analysis")
deriva_model = create_model_config(
    DerivaML,
    description="Simple model run",
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
    ],
)

# Register the main config in the hydra-zen store
store(deriva_model, name="deriva_model")

# ---------------------------------------------------------------------------
# Load configuration modules
# ---------------------------------------------------------------------------
# Dynamically import all config modules from the configs package. Each module
# registers its configurations with the hydra-zen store when imported.
#
# To add new configuration options:
# 1. Create a new module in configs/ (e.g., configs/my_model.py)
# 2. Use store() to register your configs with the appropriate group name
# 3. The module will be automatically discovered and loaded

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
