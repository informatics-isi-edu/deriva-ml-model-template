"""Smoke tests for src/scripts/_cifar10_datasets.py.

Stage 3 needs a live Deriva catalog for its dataset-creation work,
so the orchestrator-level tests are sparse — end-to-end behavior
is exercised in the load-cifar10 smoke test in Task A13. The
pure RID-level stratified-sampling helper is tested directly here.
"""

from __future__ import annotations

from collections import Counter

import pytest


def _make_rid_corpus(per_class: int) -> tuple[list[str], list[str]]:
    """Build (rids, classes) with per_class items in each of 10 classes."""
    classes_names = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    rids: list[str] = []
    classes: list[str] = []
    counter = 0
    for cls in classes_names:
        for _ in range(per_class):
            rids.append(f"R-{counter:06d}")
            classes.append(cls)
            counter += 1
    return rids, classes


def test_module_exposes_expected_api():
    from scripts._cifar10_datasets import (
        create_dataset_hierarchy,
        run_datasets_phase,
        stratified_sample_rids,
    )

    for fn in (create_dataset_hierarchy, run_datasets_phase, stratified_sample_rids):
        assert callable(fn)


def test_stratified_rid_sample_balances_partition():
    """Feed a class-balanced 100-RID set, sample 50: 5 per class."""
    from scripts._cifar10_datasets import stratified_sample_rids

    rids, classes = _make_rid_corpus(per_class=10)
    sample = stratified_sample_rids(rids, classes, sample_size=50, seed=42)
    rid_to_class = dict(zip(rids, classes))

    assert len(sample) == 50
    counts = Counter(rid_to_class[r] for r in sample)
    assert all(n == 5 for n in counts.values()), counts


def test_stratified_rid_sample_handles_imbalanced_source():
    """Skewed source (most images bird/ship): result still spreads per quota."""
    from scripts._cifar10_datasets import stratified_sample_rids

    classes = ["bird"] * 200 + ["ship"] * 50 + ["airplane"] * 5 + ["truck"] * 5
    rids = [f"R-{i:06d}" for i in range(len(classes))]
    sample = stratified_sample_rids(rids, classes, sample_size=12, seed=42)

    rid_to_class = dict(zip(rids, classes))
    counts = Counter(rid_to_class[r] for r in sample)
    # Base quota 3 per class (4 classes), remainder 0. So each class
    # contributes exactly 3 — not 6 bird-skewed copies.
    assert len(sample) == 12
    assert all(n == 3 for n in counts.values()), counts


def test_stratified_rid_sample_ignores_none_class_entries():
    from scripts._cifar10_datasets import stratified_sample_rids

    rids = ["A", "B", "C", "D"]
    classes = ["x", None, "y", None]
    sample = stratified_sample_rids(rids, classes, sample_size=2, seed=42)

    assert set(sample) == {"A", "C"}


def test_stratified_rid_sample_empty_inputs_return_empty():
    from scripts._cifar10_datasets import stratified_sample_rids

    assert stratified_sample_rids([], [], sample_size=5, seed=42) == []
    assert stratified_sample_rids(["A"], ["x"], sample_size=0, seed=42) == []


# --- _require_small_variant_distinct --------------------------------------
# Regression coverage for curator/01 (2026-05-26 e2e): refuse to build a
# small Toronto split family that would be byte-identical to the full split.


def test_require_small_variant_distinct_accepts_large_pools():
    """Pools strictly larger than SMALL_*_SIZE pass without raising."""
    from scripts._cifar10_datasets import (
        SMALL_TEST_SIZE,
        SMALL_TRAIN_SIZE,
        _require_small_variant_distinct,
    )

    # Just above threshold in both partitions — must not raise.
    _require_small_variant_distinct(
        train_pool=SMALL_TRAIN_SIZE + 1, test_pool=SMALL_TEST_SIZE + 1
    )
    # Comfortably above threshold (Toronto full sizes) — must not raise.
    _require_small_variant_distinct(train_pool=50_000, test_pool=10_000)


def test_require_small_variant_distinct_rejects_curator_01_scenario():
    """At --num-images 500 the bootstrap loads 250 train + 250 test.

    That is the exact catalog state Curator/01 flagged: the small
    variant ends up byte-identical to the full variant. We must raise
    with a message that names the threshold and the alternative.
    """
    from scripts._cifar10_datasets import (
        SmallVariantDegenerateError,
        _require_small_variant_distinct,
    )

    with pytest.raises(SmallVariantDegenerateError) as excinfo:
        _require_small_variant_distinct(train_pool=250, test_pool=250)

    msg = str(excinfo.value)
    # The operator needs to know how to recover; the message must
    # surface a concrete --num-images suggestion and point at the
    # labeled-split alternative.
    assert "--num-images" in msg
    assert "labeled-split" in msg
    # The threshold the message suggests must, when applied, clear
    # the guard — i.e. it's a working remediation, not just a number.
    # SMALL_*_SIZE is 500, so the suggested minimum is 2 * 501 = 1002.
    assert "1002" in msg


def test_require_small_variant_distinct_rejects_boundary_case():
    """Equality (pool == SMALL_*_SIZE) is still degenerate.

    When ``stratified_sample_rids`` is asked for exactly len(rids) it
    returns every input RID, so the small dataset would still be
    set-equal to the full one. The guard rejects equality alongside
    smaller-than-equal.
    """
    from scripts._cifar10_datasets import (
        SMALL_TEST_SIZE,
        SMALL_TRAIN_SIZE,
        SmallVariantDegenerateError,
        _require_small_variant_distinct,
    )

    with pytest.raises(SmallVariantDegenerateError):
        _require_small_variant_distinct(
            train_pool=SMALL_TRAIN_SIZE, test_pool=SMALL_TEST_SIZE
        )


def test_require_small_variant_distinct_rejects_asymmetric_shortage():
    """Either partition being short is enough to refuse — not both."""
    from scripts._cifar10_datasets import (
        SMALL_TEST_SIZE,
        SMALL_TRAIN_SIZE,
        SmallVariantDegenerateError,
        _require_small_variant_distinct,
    )

    # Plenty of train images, but test partition is short.
    with pytest.raises(SmallVariantDegenerateError):
        _require_small_variant_distinct(
            train_pool=SMALL_TRAIN_SIZE + 100, test_pool=SMALL_TEST_SIZE
        )
    # Plenty of test images, but train partition is short.
    with pytest.raises(SmallVariantDegenerateError):
        _require_small_variant_distinct(
            train_pool=SMALL_TRAIN_SIZE, test_pool=SMALL_TEST_SIZE + 100
        )


# --- _build_dataset_descriptions ------------------------------------------
# Regression coverage for curator/03 (2026-05-26 e2e): dataset descriptions
# must report the actual member count for the run, not the Toronto defaults.


def test_dataset_descriptions_reflect_num_images_500():
    """At --num-images 500 the assets phase yields 250 train + 250 test."""
    from scripts._cifar10_datasets import _build_dataset_descriptions

    d = _build_dataset_descriptions(
        train_count=250,
        test_count=250,
        small_train_count=250,
        small_test_count=250,
    )

    assert "250" in d["training"]
    assert "labeled images" in d["training"]
    assert "250" in d["testing"]
    assert "500" in d["complete"]
    assert "250" in d["complete"]
    assert "250" in d["small_training"]
    assert "250" in d["small_testing"]
    # None of the Toronto-default counts should appear.
    for desc in d.values():
        assert "50,000" not in desc
        assert "10,000" not in desc


def test_dataset_descriptions_reflect_num_images_1000():
    """At --num-images 1000 the assets phase yields 500 train + 500 test.

    The small variant caps at SMALL_*_SIZE = 500, so small_* equals the
    full train/test pool. The description must still report the actual
    count, not a Toronto-default placeholder.
    """
    from scripts._cifar10_datasets import _build_dataset_descriptions

    d = _build_dataset_descriptions(
        train_count=500,
        test_count=500,
        small_train_count=500,
        small_test_count=500,
    )

    assert "500" in d["training"]
    assert "500" in d["testing"]
    assert "1,000" in d["complete"]
    for desc in d.values():
        assert "50,000" not in desc
        assert "10,000" not in desc


def test_dataset_descriptions_reflect_toronto_default():
    """At full Toronto sizes the descriptions still use formatted commas."""
    from scripts._cifar10_datasets import _build_dataset_descriptions

    d = _build_dataset_descriptions(
        train_count=50_000,
        test_count=10_000,
        small_train_count=500,
        small_test_count=500,
    )

    assert "50,000" in d["training"]
    assert "10,000" in d["testing"]
    assert "60,000" in d["complete"]
    assert "500" in d["small_training"]
    assert "500" in d["small_testing"]


def test_dataset_descriptions_cover_all_toronto_keys():
    """Every Toronto-family dataset created in stage 3 has a description."""
    from scripts._cifar10_datasets import _build_dataset_descriptions

    d = _build_dataset_descriptions(
        train_count=400,
        test_count=100,
        small_train_count=400,
        small_test_count=100,
    )
    expected_keys = {
        "complete",
        "split",
        "training",
        "testing",
        "small_split",
        "small_training",
        "small_testing",
    }
    assert set(d.keys()) == expected_keys
    # Each description should be non-empty and contain a digit (a count).
    for key, desc in d.items():
        assert desc, key
        assert any(ch.isdigit() for ch in desc), (key, desc)


def test_labeled_split_description_reports_partition_sizes():
    from scripts._cifar10_datasets import _labeled_split_description

    desc = _labeled_split_description(250)
    assert "200" in desc
    assert "50" in desc
    assert "seed=42" in desc
    assert "80/20" in desc


def test_labeled_split_description_full_training_set():
    from scripts._cifar10_datasets import _labeled_split_description

    desc = _labeled_split_description(50_000)
    assert "40,000" in desc
    assert "10,000" in desc


def test_small_labeled_split_description_uses_400_100_at_or_above_500():
    from scripts._cifar10_datasets import _small_labeled_split_description

    desc = _small_labeled_split_description(500)
    assert "400/100" in desc
    assert "seed=42" in desc

    desc_big = _small_labeled_split_description(50_000)
    assert "400/100" in desc_big


def test_small_labeled_split_description_falls_back_below_500():
    from scripts._cifar10_datasets import _small_labeled_split_description

    desc = _small_labeled_split_description(250)
    # Fallback path is 80/20 of training_count, seed=123.
    assert "200" in desc
    assert "50" in desc
    assert "seed=123" in desc
    assert "400/100" not in desc
