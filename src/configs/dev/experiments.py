"""Experiments for alternate catalog.

These experiments use dev/prod-specific datasets, assets, and connection
configs. Each experiment overrides ``/deriva_ml`` to point at the
alternate catalog.

Example:
    uv run deriva-ml-run +experiment=cifar10_quick_dev
"""

# from hydra_zen import make_config, store
# from configs.base import DerivaModelConfig
#
# experiment_store = store(group="experiment", package="_global_")
#
# experiment_store(
#     make_config(
#         hydra_defaults=[
#             "_self_",
#             {"override /model_config": "cifar10_quick"},
#             {"override /datasets": "my_dataset_dev"},
#             {"override /assets": "my_weights_dev"},
#             {"override /deriva_ml": "my-server"},
#         ],
#         description="Quick CIFAR-10 on the dev catalog.",
#         bases=(DerivaModelConfig,),
#     ),
#     name="cifar10_quick_dev",
# )
