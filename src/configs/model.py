"""Model configurations.

Configuration Group: ``model_config``
-------------------------------------
A *model config* binds your model function to a concrete set of
hyperparameters. The runner selects one with ``model_config=<name>`` and
passes every field through to the function as keyword arguments at dispatch
time.

REQUIRED: a config named ``default_model`` must be registered in this group.
``src/configs/base.py`` lists ``{"model_config": "default_model"}`` in its
Hydra defaults, so the whole config tree fails to resolve without it.

Replace ``example_model`` below with your own model function. Any callable
that implements the ``src/models/model_protocol.py`` interface works: it
receives its Hydra-configured fields as keyword arguments plus an
``ml_instance`` (the :class:`~deriva_ml.DerivaML` connection) and an
``execution`` (the run context), and returns ``None``. Build the config with
``zen_partial=True`` so hydra-zen leaves ``ml_instance`` / ``execution``
unbound — the runner supplies them when the execution starts.
"""

from __future__ import annotations

from typing import Any

from hydra_zen import builds, store


def example_model(
    learning_rate: float = 1e-3,
    epochs: int = 10,
    *,
    ml_instance: Any = None,
    execution: Any = None,
) -> None:
    """Placeholder model that exercises the execution plumbing.

    This stand-in does no real training. It exists so a freshly cloned
    template resolves and dry-runs end to end. Replace it with your own model
    function (and point ``default_model`` at it) — see
    ``src/models/model_protocol.py`` for the interface a model must satisfy.

    Args:
        learning_rate: Example hyperparameter, forwarded from the Hydra config.
        epochs: Example hyperparameter, forwarded from the Hydra config.
        ml_instance: The :class:`~deriva_ml.DerivaML` connection, injected by
            the runner at dispatch time. ``None`` until then.
        execution: The execution context, injected by the runner at dispatch
            time. ``None`` until then. Use it to read input datasets, register
            output assets, and update run status.

    Returns:
        None. Models report results through ``execution`` (registered assets,
        created datasets, status updates), never via a return value.

    Example:
        >>> example_model(learning_rate=0.01, epochs=3)  # doctest: +SKIP
        # No-op placeholder; replace with real training logic.
    """
    # Replace this body with your training / inference logic. ``ml_instance``
    # and ``execution`` are supplied by the runner once the execution starts.
    return None


# ---------------------------------------------------------------------------
# Register with the hydra-zen store under the ``model_config`` group.
# ---------------------------------------------------------------------------
model_store = store(group="model_config")

# Base config: the model callable bound to its default hyperparameters.
# ``zen_partial=True`` leaves ``ml_instance`` / ``execution`` unbound so the
# runner can inject them when the execution starts.
ExampleModelConfig = builds(
    example_model,
    learning_rate=1e-3,
    epochs=10,
    populate_full_signature=True,
    zen_partial=True,
)

# REQUIRED: ``default_model`` is selected when no ``model_config`` override is
# given. Point this at your own model config once you replace ``example_model``.
model_store(ExampleModelConfig, name="default_model")

# -----------------------------------------------------------------------------
# Add alternate hyperparameter sets as named configs. Each one reuses the base
# build and overrides only the fields that differ — select with
# ``deriva-ml-run model_config=<name>``.
# -----------------------------------------------------------------------------
#
#   model_store(
#       ExampleModelConfig,
#       name="<your_variant>",
#       learning_rate=1e-2,
#       epochs=30,
#       zen_meta={"description": "Higher learning rate, longer training."},
#   )
