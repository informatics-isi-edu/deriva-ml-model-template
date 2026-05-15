"""Smoke tests for src/scripts/_cifar10_schema.py.

Most of stage 1's behavior requires a live Deriva catalog, so this
test file is intentionally sparse — it verifies the module's
public API exists and is importable. The end-to-end behavior is
exercised in the load-cifar10 smoke test in Task A13 and in
Part B of the broader test plan.
"""

from __future__ import annotations


def test_module_exposes_expected_api():
    from scripts._cifar10_schema import (
        create_or_connect_catalog,
        setup_domain_model,
        setup_workflow_types,
        setup_dataset_types,
        apply_annotations,
        run_schema_phase,
    )

    # Sanity: each is callable.
    for fn in (
        create_or_connect_catalog,
        setup_domain_model,
        setup_workflow_types,
        setup_dataset_types,
        apply_annotations,
        run_schema_phase,
    ):
        assert callable(fn)
