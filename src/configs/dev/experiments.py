"""Experiments for alternate catalog.

These experiments use dev/prod-specific datasets, assets, and connection
configs. Each experiment overrides ``/deriva_ml`` to point at the
alternate catalog.

Example:
    uv run deriva-ml-run +experiment=cifar10_phase7_localhost
"""

from hydra_zen import make_config, store

from configs.base import DerivaModelConfig

experiment_store = store(group="experiment", package="_global_")

# Phase 7 e2e: train cifar10_quick against the new three-way split's training
# partition (DKT = 150 Images of 85J at version 0.2.0). The training partition
# is registered as `cifar10_phase7_training_localhost` in
# dev/datasets_localhost.py. Verifies the new-dataset -> new-workflow ->
# trained-model loop the e2e spec calls for in Phase 7.
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
            {"override /datasets": "cifar10_phase7_training_localhost"},
            {"override /deriva_ml": "localhost_46"},
        ],
        description=(
            "Phase 7 e2e: train cifar10_quick against the new three-way "
            "split's training partition (DKT@0.2.0). Exercises the spec's "
            "'train against a new split' flow."
        ),
        bases=(DerivaModelConfig,),
    ),
    name="cifar10_phase7_localhost",
)
