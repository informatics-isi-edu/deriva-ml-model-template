"""Base configuration re-exported from deriva-ml.

This module re-exports the base configuration classes from deriva-ml
for convenient access within the configs package.

Usage:
    from configs.base import BaseConfig, base_defaults
"""

from deriva_ml.execution import BaseConfig, DerivaBaseConfig, base_defaults

__all__ = ["BaseConfig", "DerivaBaseConfig", "base_defaults"]
