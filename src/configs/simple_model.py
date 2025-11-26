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
model_store = store(group="model_config")
model_store(SimpleModelConfig, name="default_model")

# Register additional variants by extending (instantiating) the base config
# with overridden fields. This avoids multiple calls to `builds`.
model_store(SimpleModelConfig, name="epochs_20", epochs=20)
model_store(SimpleModelConfig, name="epochs_100", epochs=100)