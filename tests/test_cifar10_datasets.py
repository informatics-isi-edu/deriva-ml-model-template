"""Smoke tests for src/scripts/_cifar10_datasets.py.

Stage 3 needs a live Deriva catalog for its dataset-creation work,
so the orchestrator-level tests are sparse — end-to-end behavior
is exercised in the load-cifar10 smoke test in Task A13. The
pure RID-level stratified-sampling helper is tested directly here.
"""

from __future__ import annotations

from collections import Counter


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
