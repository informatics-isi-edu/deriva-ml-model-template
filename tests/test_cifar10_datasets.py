"""Smoke tests for src/scripts/_cifar10_datasets.py.

Stage 3 needs a live Deriva catalog for its actual work, so
this test file is intentionally sparse — it verifies module
structure. End-to-end behavior is exercised in the
load-cifar10 smoke test in Task A13.
"""

from __future__ import annotations


def test_module_exposes_expected_api():
    from scripts._cifar10_datasets import (
        create_dataset_hierarchy,
        run_datasets_phase,
    )

    for fn in (create_dataset_hierarchy, run_datasets_phase):
        assert callable(fn)
