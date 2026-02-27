"""Dev catalog configurations.

Config modules in this subpackage define datasets, assets, experiments,
and connection settings specific to a development/alternate catalog.
They are auto-discovered by load_configs() alongside the top-level configs.

Pattern:
    - Create dev/deriva.py for alternate catalog connections
    - Create dev/datasets.py for dev-specific dataset RIDs
    - Create dev/assets.py for dev-specific asset RIDs
    - Create dev/experiments.py for dev-specific experiment presets

Usage:
    # Run an experiment targeting the dev catalog
    uv run deriva-ml-run +experiment=my_experiment_dev
"""
