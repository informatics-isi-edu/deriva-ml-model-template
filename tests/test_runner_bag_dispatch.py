"""Tests for the ``cifar10_cnn`` bag-dispatch layer.

Covers the refactor that replaced the closed-list ``_bag_role`` table with a
dispatch keyed off the catalog's ``Dataset_Type`` vocabulary. The dispatch
must:

- Route ``Training``-typed bags to the training lane.
- Ignore qualifier terms like ``Labeled`` / ``Complete`` (orthogonal to role).
- Route ``Validation``-typed bags to a dedicated validation lane (used per
  epoch in the training loop).
- Raise a clear error when ``require_training=True`` and no training bag was
  supplied — closes the catalog-18 F40-style silent-failure mode where a
  Validation-only execution looked exactly like a successful training run.
- Warn (not silently drop) bags with unrecognized ``Dataset_Type`` terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from models.cifar10_cnn import (
    _bag_dataset_types,
    _classify_bag,
    _flatten_bags,
    load_cifar10_from_execution,
)


class _TinyImageDataset(Dataset):
    """In-memory torch Dataset mimicking ``DatasetBag.as_torch_dataset`` output.

    Yields ``(tensor, label, rid)`` triples — the same shape the real
    adapter returns in :mod:`deriva_ml.dataset.dataset_bag` after the
    2026-05-19 change.
    """

    def __init__(self, n: int = 4, rid_prefix: str = "X") -> None:
        self._items: list[tuple[torch.Tensor, int, str]] = [
            (torch.zeros(3, 32, 32), i % 10, f"{rid_prefix}-{i}") for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        return self._items[idx]


@dataclass
class _FakeBag:
    """Stand-in for a real ``DatasetBag`` — exposes only what the runner reads.

    The dispatch path inside ``load_cifar10_from_execution`` touches:
    ``dataset_types``, ``dataset_rid``, ``list_dataset_children()`` (for
    Split parents), and ``as_torch_dataset(...)``. Everything else on the
    real bag is irrelevant for testing the dispatch.
    """

    dataset_rid: str
    dataset_types: list[str]
    children: list["_FakeBag"] = field(default_factory=list)
    n_samples: int = 4

    def list_dataset_children(self) -> list["_FakeBag"]:
        return self.children

    def as_torch_dataset(self, **_kwargs: Any) -> _TinyImageDataset:
        return _TinyImageDataset(n=self.n_samples, rid_prefix=self.dataset_rid)


@dataclass
class _FakeExecution:
    """Stand-in for a real ``Execution`` — exposes ``.datasets`` only."""

    datasets: list[_FakeBag]


# ---------------------------------------------------------------------------
# Classification primitives
# ---------------------------------------------------------------------------


def test_bag_dataset_types_returns_catalog_terms_verbatim():
    """``_bag_dataset_types`` returns the catalog terms unchanged (case kept)."""
    bag = _FakeBag(dataset_rid="A1", dataset_types=["Validation", "Labeled"])
    assert _bag_dataset_types(bag) == ["Validation", "Labeled"]


def test_classify_bag_separates_roles_from_qualifiers():
    """Role terms go into the lower-cased role set; qualifiers into ``other``."""
    bag = _FakeBag(dataset_rid="A1", dataset_types=["Training", "Labeled"])
    roles, other = _classify_bag(bag)
    assert roles == {"training"}
    assert other == ["Labeled"]


def test_classify_bag_handles_validation_with_labeled_qualifier():
    """``Validation`` is a role; ``Labeled`` is an orthogonal qualifier."""
    bag = _FakeBag(dataset_rid="DAP", dataset_types=["Validation", "Labeled"])
    roles, other = _classify_bag(bag)
    assert roles == {"validation"}
    assert other == ["Labeled"]


def test_classify_bag_unknown_term_lands_in_other():
    """User-added vocab terms the runner doesn't handle go to ``other``."""
    bag = _FakeBag(dataset_rid="CAL", dataset_types=["Calibration"])
    roles, other = _classify_bag(bag)
    assert roles == set()
    assert other == ["Calibration"]


# ---------------------------------------------------------------------------
# Split flattening
# ---------------------------------------------------------------------------


def test_flatten_bags_descends_into_split_parents():
    """A ``Split`` parent is replaced by its children in the flattened list."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training"])
    test = _FakeBag(dataset_rid="E", dataset_types=["Testing"])
    parent = _FakeBag(dataset_rid="P", dataset_types=["Split"], children=[train, test])
    leaves = _flatten_bags([parent])
    assert {b.dataset_rid for b in leaves} == {"T", "E"}


def test_flatten_bags_leaves_non_split_bags_intact():
    """Bags without a ``Split`` role pass through untouched."""
    val = _FakeBag(dataset_rid="DAP", dataset_types=["Validation", "Labeled"])
    assert _flatten_bags([val]) == [val]


# ---------------------------------------------------------------------------
# Dispatch — training + validation lanes engage together
# ---------------------------------------------------------------------------


def test_training_and_validation_bags_both_engage():
    """A Training bag + a Validation bag yields a train_loader + val_loader."""
    train = _FakeBag(
        dataset_rid="T", dataset_types=["Training", "Labeled"], n_samples=8
    )
    val = _FakeBag(
        dataset_rid="DAP", dataset_types=["Validation", "Labeled"], n_samples=4
    )
    execution = _FakeExecution(datasets=[train, val])

    train_loader, test_loader, val_loader, class_names = load_cifar10_from_execution(
        execution, batch_size=2, require_training=True
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is None  # no Testing bag in this execution
    assert len(class_names) == 10  # canonical CIFAR-10
    # Verify the loaders are wired to the right bags by RID prefix.
    train_batches = list(train_loader)
    val_batches = list(val_loader)
    train_rids = [r for batch in train_batches for r in batch[2]]
    val_rids = [r for batch in val_batches for r in batch[2]]
    assert all(rid.startswith("T-") for rid in train_rids)
    assert all(rid.startswith("DAP-") for rid in val_rids)


def test_training_plus_labeled_qualifier_dispatches_to_training():
    """``["Training", "Labeled"]`` reaches the training handler (qualifier ignored)."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training", "Labeled"])
    execution = _FakeExecution(datasets=[train])

    train_loader, test_loader, val_loader, _names = load_cifar10_from_execution(
        execution, batch_size=2, require_training=True
    )

    assert train_loader is not None
    assert test_loader is None
    assert val_loader is None


def test_training_inside_split_parent_dispatches():
    """A Split parent containing Training + Validation children dispatches each."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training"])
    val = _FakeBag(dataset_rid="V", dataset_types=["Validation", "Labeled"])
    parent = _FakeBag(dataset_rid="P", dataset_types=["Split"], children=[train, val])
    execution = _FakeExecution(datasets=[parent])

    train_loader, _test, val_loader, _names = load_cifar10_from_execution(
        execution, batch_size=2, require_training=True
    )
    assert train_loader is not None
    assert val_loader is not None


# ---------------------------------------------------------------------------
# Safety rail — closes the catalog-18 F40 silent-failure mode
# ---------------------------------------------------------------------------


def test_validation_only_input_raises_when_training_required():
    """Validation-only input with ``require_training=True`` raises a clear error.

    This is the catalog-18 F40 regression: a Curator hands the Developer a
    Validation-typed dataset, the runner used to silently fall through and
    produce a degenerate execution with only ``training_status.txt``. After
    the fix, the runner refuses loudly so the execution lands as
    ``Status=Failed`` rather than ``Status=Uploaded``.
    """
    val = _FakeBag(dataset_rid="DAP", dataset_types=["Validation", "Labeled"])
    execution = _FakeExecution(datasets=[val])

    with pytest.raises(RuntimeError) as excinfo:
        load_cifar10_from_execution(execution, batch_size=2, require_training=True)

    msg = str(excinfo.value)
    assert "Dataset_Type=Training" in msg
    assert "DAP" in msg  # diagnostic includes input bag RID
    assert "Validation" in msg  # and its actual types


def test_validation_only_input_is_fine_when_training_not_required():
    """``require_training=False`` (e.g. test_only mode) doesn't trip the safety rail."""
    val = _FakeBag(dataset_rid="DAP", dataset_types=["Validation", "Labeled"])
    execution = _FakeExecution(datasets=[val])

    train_loader, test_loader, val_loader, _names = load_cifar10_from_execution(
        execution, batch_size=2, require_training=False
    )
    assert train_loader is None
    assert test_loader is None
    assert val_loader is not None


def test_unrecognized_role_emits_warning_and_is_skipped():
    """A bag with no recognized role term warns rather than silently dropping."""
    train = _FakeBag(dataset_rid="T", dataset_types=["Training"])
    weird = _FakeBag(dataset_rid="CAL", dataset_types=["Calibration"])
    execution = _FakeExecution(datasets=[train, weird])

    with pytest.warns(RuntimeWarning, match="Calibration"):
        train_loader, test_loader, val_loader, _names = load_cifar10_from_execution(
            execution, batch_size=2, require_training=True
        )

    assert train_loader is not None
    assert test_loader is None
    assert val_loader is None
