"""Smoke tests for src/scripts/_cifar10_assets.py.

Stage 2 needs a live Deriva catalog for its actual work, so
this test file is intentionally sparse — it verifies module
structure. End-to-end behavior is exercised in the
load-cifar10 smoke test in Task A13.
"""

from __future__ import annotations


def test_module_exposes_expected_api():
    from scripts._cifar10_assets import (
        upload_images,
        add_classification_features,
        run_assets_phase,
        class_from_filename,
    )

    for fn in (
        upload_images,
        add_classification_features,
        run_assets_phase,
        class_from_filename,
    ):
        assert callable(fn)


def test_class_from_filename_decodes_train():
    from scripts._cifar10_assets import class_from_filename

    assert class_from_filename("train_frog_42.png") == "frog"


def test_class_from_filename_decodes_test():
    from scripts._cifar10_assets import class_from_filename

    assert class_from_filename("test_cat_19.png") == "cat"


def test_class_from_filename_returns_none_for_unknown():
    from scripts._cifar10_assets import class_from_filename

    assert class_from_filename("random_image.png") is None
    assert class_from_filename("train.png") is None
