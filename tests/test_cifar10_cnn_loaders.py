"""Unit tests for build_loaders error paths in src/models/cifar10_cnn.py.

These tests use minimal mock objects rather than a real Execution because
the failure paths we care about are entirely upstream of bag download —
the function reads execution.datasets and decides what to do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from models.cifar10_cnn import build_loaders


@dataclass
class _MockExecution:
    """Minimal stand-in for deriva_ml.execution.Execution.

    build_loaders only reads execution.datasets, so that's all we model.
    """

    datasets: list[Any] = field(default_factory=list)


def test_empty_datasets_raises_clear_error() -> None:
    """When execution.datasets is empty, build_loaders raises with a
    clear "no input datasets" message — distinct from the deeper
    "no Training bag after flattening" failure, so the user can
    diagnose the misconfig (typically: a Hydra `datasets=foo` group
    that resolved to an empty list, e.g. a placeholder registry in
    src/configs/datasets.py that wasn't filled in for the catalog).
    """
    execution = _MockExecution(datasets=[])

    with pytest.raises(RuntimeError) as exc_info:
        build_loaders(execution, batch_size=32, require_training=True)

    message = str(exc_info.value)
    assert "no input datasets" in message.lower() or "empty" in message.lower(), (
        f"Expected the error to mention empty/no datasets; got: {message!r}"
    )
