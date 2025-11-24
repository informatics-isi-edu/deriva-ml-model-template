"""
Model configuration registrations for Hydra/Hydra-Zen.

This module defines model configurations and registers them into Hydra's store
under the "model_config" group. We build the configuration once, then create
variants by instantiating the built config with overridden fields (no extra
builds required).
"""

from hydra_zen import builds, store

from models.simple_model import simple_model

# Build the base configuration once.
SimpleModelConfig = builds(
    simple_model,
    learning_rate=1e-3,
    epochs=10,
    populate_full_signature=True,
    zen_partial=True,
)

# Register the base config as the default model.
store(SimpleModelConfig, group="model_config", name="default_model")

# Register additional variants by extending (instantiating) the base config
# with overridden fields. This avoids multiple calls to `builds`.
store(SimpleModelConfig, epochs=20, group="model_config", name="epochs_20")
store(SimpleModelConfig, epochs=100, group="model_config", name="epochs_100")