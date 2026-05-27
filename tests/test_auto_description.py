"""Tests for the auto-composed Execution description.

When a user runs ``deriva-ml-run model_config=X datasets=Y`` without
``+experiment=`` and without an explicit ``description=...``, the
template's ``base.py`` wrapper auto-composes a description from the
resolved Hydra choices and scalar overrides. This keeps
``ml.list_executions()`` output scannable without forcing the user to
remember to set ``description=``.

These tests pin the contract at the composer level: given a mocked
``HydraConfig`` reflecting the runtime choices and task overrides the
user actually typed, the composer returns the expected string. The
wrapper test pins that:

* an explicit ``description=`` wins over the auto-composer, and
* the sentinel default ``"Simple model run"`` triggers auto-composition.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from configs.base import (
    _DEFAULT_DESCRIPTION,
    _compose_default_description,
    _run_model_with_auto_description,
)


def _mock_hydra_config(
    choices: dict[str, str | None], task_overrides: list[str]
) -> MagicMock:
    """Build a MagicMock that mimics ``HydraConfig.get()``.

    Args:
        choices: Mapping for ``runtime.choices`` (e.g.
            ``{"model_config": "cifar10_quick", "experiment": None}``).
        task_overrides: List passed as ``overrides.task`` (e.g.
            ``["model_config=cifar10_quick", "seed=7"]``).

    Returns:
        A MagicMock with the two attributes the composer reads.
    """
    cfg = MagicMock()
    cfg.runtime.choices = choices
    cfg.overrides.task = task_overrides
    return cfg


# ---------------------------------------------------------------------------
# _compose_default_description
# ---------------------------------------------------------------------------


def test_compose_outside_hydra_context_returns_none():
    """Outside a Hydra run, the composer must return ``None`` so the
    static default ``"Simple model run"`` is preserved (e.g. when a unit
    test calls ``run_model`` directly without going through ``zen``).
    """
    assert _compose_default_description() is None


def test_compose_with_model_and_dataset_and_scalar_override():
    """The shipped golden path: ``model_config=X datasets=Y seed=N``.

    Produces ``"X on Y (seed=N)"`` — terse, scannable, and stable across
    runs.
    """
    mock_cfg = _mock_hydra_config(
        choices={
            "model_config": "cifar10_quick",
            "datasets": "cifar10_labeled_split",
            "experiment": None,
        },
        task_overrides=[
            "model_config=cifar10_quick",
            "datasets=cifar10_labeled_split",
            "seed=7",
        ],
    )
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg):
        result = _compose_default_description()
    assert result == "cifar10_quick on cifar10_labeled_split (seed=7)"


def test_compose_with_only_choices_no_scalar_overrides():
    """When only config-group overrides were typed, the parens are
    omitted — no clutter, just ``"<model> on <dataset>"``.
    """
    mock_cfg = _mock_hydra_config(
        choices={
            "model_config": "cifar10_quick",
            "datasets": "cifar10_labeled_split",
        },
        task_overrides=[
            "model_config=cifar10_quick",
            "datasets=cifar10_labeled_split",
        ],
    )
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg):
        result = _compose_default_description()
    assert result == "cifar10_quick on cifar10_labeled_split"


def test_compose_with_multiple_scalar_overrides():
    """Multiple scalar overrides join with ", " inside the parens.

    Preserves dotted paths (``model_config.epochs=5``) verbatim so the
    user sees the exact config knob they twisted.
    """
    mock_cfg = _mock_hydra_config(
        choices={
            "model_config": "cifar10_quick",
            "datasets": "cifar10_labeled_split",
        },
        task_overrides=[
            "model_config=cifar10_quick",
            "datasets=cifar10_labeled_split",
            "seed=7",
            "model_config.epochs=5",
        ],
    )
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg):
        result = _compose_default_description()
    assert result == (
        "cifar10_quick on cifar10_labeled_split (seed=7, model_config.epochs=5)"
    )


def test_compose_skips_config_group_overrides():
    """``model_config=...``, ``datasets=...``, ``+experiment=...``,
    ``description=...`` etc. are not emitted inside the parens — they
    are either already represented in the prefix or would never be in
    play when the composer fires.
    """
    mock_cfg = _mock_hydra_config(
        choices={
            "model_config": "cifar10_quick",
            "datasets": "cifar10_labeled_split",
        },
        task_overrides=[
            "model_config=cifar10_quick",
            "datasets=cifar10_labeled_split",
            "+experiment=cifar10_quick",  # would never co-occur, but guard against it
            "workflow=default_workflow",
            "seed=7",
        ],
    )
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg):
        result = _compose_default_description()
    assert result == "cifar10_quick on cifar10_labeled_split (seed=7)"


def test_compose_returns_none_when_no_useful_choices():
    """If neither a model nor a dataset choice is resolved, fall back
    to the static default rather than emit an empty/awkward string.
    """
    mock_cfg = _mock_hydra_config(
        choices={"experiment": None},
        task_overrides=[],
    )
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg):
        result = _compose_default_description()
    assert result is None


def test_compose_handles_hydra_sigils():
    """Hydra prefixes (``+``, ``++``, ``~``) on overrides are stripped
    before parsing so ``+seed=7`` reads as ``seed=7``.
    """
    mock_cfg = _mock_hydra_config(
        choices={
            "model_config": "cifar10_quick",
            "datasets": "cifar10_labeled_split",
        },
        task_overrides=["+seed=7"],
    )
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg):
        result = _compose_default_description()
    assert result == "cifar10_quick on cifar10_labeled_split (seed=7)"


# ---------------------------------------------------------------------------
# _run_model_with_auto_description (wrapper behavior)
# ---------------------------------------------------------------------------


def test_wrapper_substitutes_when_description_is_default():
    """If ``description`` is the sentinel default and the composer
    yields a useful string, the wrapper substitutes before calling
    ``run_model``.
    """
    mock_cfg = _mock_hydra_config(
        choices={
            "model_config": "cifar10_quick",
            "datasets": "cifar10_labeled_split",
        },
        task_overrides=["model_config=cifar10_quick", "datasets=cifar10_labeled_split"],
    )
    with (
        patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg),
        patch("configs.base.run_model") as mock_run,
    ):
        _run_model_with_auto_description(description=_DEFAULT_DESCRIPTION)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["description"] == "cifar10_quick on cifar10_labeled_split"


def test_wrapper_preserves_explicit_description():
    """An explicit ``description=...`` (set on the CLI or in an
    ``+experiment=`` config) wins — the wrapper must not overwrite it.
    """
    mock_cfg = _mock_hydra_config(
        choices={
            "model_config": "cifar10_quick",
            "datasets": "cifar10_labeled_split",
        },
        task_overrides=["model_config=cifar10_quick"],
    )
    explicit = "My custom description for this ablation run"
    with (
        patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg),
        patch("configs.base.run_model") as mock_run,
    ):
        _run_model_with_auto_description(description=explicit)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["description"] == explicit


def test_wrapper_falls_back_to_default_when_no_choices():
    """If the composer returns ``None`` (nothing meaningful to compose
    from), the wrapper leaves the sentinel default in place rather than
    blanking ``description``.
    """
    mock_cfg = _mock_hydra_config(
        choices={"experiment": None},
        task_overrides=[],
    )
    with (
        patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_cfg),
        patch("configs.base.run_model") as mock_run,
    ):
        _run_model_with_auto_description(description=_DEFAULT_DESCRIPTION)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["description"] == _DEFAULT_DESCRIPTION
