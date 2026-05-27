"""Base configuration for the model runner.

This module creates and exports the main model runner configuration that
can be used as a base for experiments.

Usage:
    from configs.base import BaseConfig, DerivaModelConfig
"""

from __future__ import annotations

from functools import partial, wraps
from typing import Any

from hydra_zen import builds, store

from deriva_ml import DerivaML
from deriva_ml.execution import (
    BaseConfig,
    DerivaBaseConfig,
    base_defaults,
    run_model,
)

# Sentinel default description. When Execution.description equals this
# AND no ``+experiment=`` was applied, the runner auto-composes a
# description from the resolved Hydra choices and scalar overrides.
# Users who set ``description=...`` on the command line or via an
# experiment config always win over this default.
_DEFAULT_DESCRIPTION = "Simple model run"

# Hydra defaults list applied to the top-level model runner config.
_HYDRA_DEFAULTS: list[Any] = [
    "_self_",
    {"deriva_ml": "default_deriva"},
    {"datasets": "default_dataset"},
    {"assets": "default_asset"},
    {"workflow": "default_workflow"},
    {"model_config": "default_model"},
    {"optional script_config": "none"},
]


def _compose_default_description() -> str | None:
    """Compose a description from the resolved Hydra choices and overrides.

    Used when the user invokes ``deriva-ml-run`` with bare config-group
    overrides (e.g. ``model_config=cifar10_quick datasets=cifar10_labeled_split``)
    and does *not* supply ``description=...`` or ``+experiment=...``. In
    that case the static default ``"Simple model run"`` makes
    ``ml.list_executions()`` output unreadable: every ad-hoc run shows the
    same string. Composing a description from the actual choices keeps
    provenance scannable without forcing the user to think about it.

    Returns:
        A composed description string (e.g.
        ``"cifar10_quick on cifar10_labeled_split (seed=7)"``), or
        ``None`` when no useful information is available from the Hydra
        runtime (e.g. running outside a Hydra context, or when only
        defaults were resolved).

    Example:
        >>> # Inside a Hydra run with overrides
        >>> # `model_config=cifar10_quick datasets=cifar10_labeled_split seed=7`
        >>> _compose_default_description()  # doctest: +SKIP
        'cifar10_quick on cifar10_labeled_split (seed=7)'
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
    except Exception:
        # Not inside a Hydra context (e.g. unit tests calling run_model
        # directly). Nothing to compose from.
        return None

    choices = {k: v for k, v in hydra_cfg.runtime.choices.items() if v is not None}
    model_choice = choices.get("model_config")
    dataset_choice = choices.get("datasets")

    # Parse scalar overrides from the task override list. We only surface
    # leaf-level scalar overrides (e.g. ``seed=7``); config-group choices
    # like ``model_config=...`` and ``datasets=...`` are already covered
    # by the ``<model> on <dataset>`` prefix below. Skip ``description=``
    # (it would mean the user *did* set one, and this composer wouldn't
    # have been called) and Hydra-internal ``+experiment=...``.
    scalar_overrides: list[str] = []
    try:
        task_overrides = list(hydra_cfg.overrides.task)
    except Exception:
        task_overrides = []

    # Bare config-group selections (``model_config=...``, ``datasets=...``,
    # ``+experiment=...``) are already captured by the
    # ``<model> on <dataset>`` prefix and shouldn't be repeated.
    # ``description=...`` would mean the user supplied one (and this
    # composer wouldn't have been called). Dotted overrides into these
    # groups (``model_config.epochs=5``) DO surface — the user twisted a
    # specific knob and we want it in the description.
    _SKIP_BARE_KEYS = {
        "model_config",
        "datasets",
        "assets",
        "workflow",
        "deriva_ml",
        "script_config",
        "experiment",
        "description",
    }
    for ov in task_overrides:
        # Strip Hydra's leading sigils (``+``, ``++``, ``~``).
        stripped = ov.lstrip("+~")
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key in _SKIP_BARE_KEYS:
            continue
        scalar_overrides.append(f"{key}={value}")

    # If we have neither a model nor a dataset choice, there is nothing
    # informative to compose. Let the static default stand.
    if not model_choice and not dataset_choice:
        return None

    if model_choice and dataset_choice:
        prefix = f"{model_choice} on {dataset_choice}"
    elif model_choice:
        prefix = model_choice
    else:
        prefix = f"on {dataset_choice}"

    if scalar_overrides:
        return f"{prefix} ({', '.join(scalar_overrides)})"
    return prefix


@wraps(run_model)
def _run_model_with_auto_description(*args: Any, **kwargs: Any) -> None:
    """Wrap ``run_model`` to auto-compose a description from Hydra overrides.

    When the user runs ``deriva-ml-run model_config=X datasets=Y`` without
    ``+experiment=`` and without an explicit ``description=...``, the
    ``description`` field stays at the static default
    (``"Simple model run"``). That makes ad-hoc runs indistinguishable
    in ``ml.list_executions()`` output. This wrapper detects that case
    and substitutes a description composed from the resolved choices and
    scalar overrides before delegating to ``run_model``.

    ``functools.wraps`` copies ``run_model``'s ``__wrapped__`` and
    ``__signature__`` so hydra-zen's ``populate_full_signature=True``
    resolves the same fields it would resolve against ``run_model``
    directly.

    Args:
        *args: Positional arguments forwarded to ``run_model``.
        **kwargs: Keyword arguments forwarded to ``run_model``. If
            ``description`` is the sentinel default and a useful
            composition is available, it is replaced.

    Returns:
        ``None``. Side effects are the same as ``run_model``: catalog
        execution record creation and result upload.

    Example:
        >>> # Invoked via hydra-zen, never directly. The user runs:
        >>> # uv run deriva-ml-run model_config=cifar10_quick \\
        >>> #     datasets=cifar10_labeled_split seed=7
        >>> # and the resulting Execution.description reads:
        >>> # 'cifar10_quick on cifar10_labeled_split (seed=7)'
    """
    if kwargs.get("description") == _DEFAULT_DESCRIPTION:
        composed = _compose_default_description()
        if composed is not None:
            kwargs["description"] = composed
    return run_model(*args, **kwargs)


# Create the main configuration schema for the model runner.
# This is a builds() of the auto-description wrapper with the standard
# hydra defaults. Experiments should inherit from this config.
DerivaModelConfig = builds(
    partial(_run_model_with_auto_description, ml_class=DerivaML),
    description=_DEFAULT_DESCRIPTION,
    populate_full_signature=True,
    hydra_defaults=_HYDRA_DEFAULTS,
)

# Register with the hydra-zen store
store(DerivaModelConfig, name="deriva_model")

__all__ = ["BaseConfig", "DerivaBaseConfig", "DerivaModelConfig", "base_defaults"]
