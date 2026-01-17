"""Base configuration for the model runner.

This module creates and exports the main model runner configuration that
can be used as a base for experiments.

Usage:
    from configs.base import BaseConfig, DerivaModelConfig
"""

from hydra_zen import store

from deriva_ml import DerivaML
from deriva_ml.execution import BaseConfig, DerivaBaseConfig, base_defaults, create_model_config

# Create the main configuration schema for the model runner.
# This is a builds() of run_model with the standard hydra defaults.
# Experiments should inherit from this config.
DerivaModelConfig = create_model_config(
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

# Register with the hydra-zen store
store(DerivaModelConfig, name="deriva_model")

__all__ = ["BaseConfig", "DerivaBaseConfig", "DerivaModelConfig", "base_defaults"]
