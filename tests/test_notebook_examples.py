"""Smoke tests for the notebook-side conventions around ``run_notebook``.

Confirms two things that together protect the template's documented pattern:

1. The upstream ``deriva_ml.execution.run_notebook`` signature still accepts
   ``config_name`` as an optional argument with a ``None`` default. The whole
   point of the template's auto-derive examples is that the caller does not
   pass a config name; if upstream regressed this, the examples would break.

2. ``notebooks/roc_analysis.ipynb`` actually uses the auto-derived form — i.e.
   does not pass a positional ``"roc_analysis"`` string to ``run_notebook``.
   The DerivaML convention is that notebook ``X.ipynb`` uses the config
   registered as ``notebook_config("X", ...)``, so an explicit positional
   string in the example notebook is redundant and misleading.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import nbformat
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ROC_NOTEBOOK = REPO_ROOT / "notebooks" / "roc_analysis.ipynb"


def test_run_notebook_config_name_is_optional() -> None:
    """``run_notebook`` must accept ``config_name`` with a ``None`` default.

    The template's notebook and docs all rely on the auto-derive path. If the
    upstream signature ever stops defaulting ``config_name`` to ``None``,
    every example call (``run_notebook(workflow_type=...)`` with no positional
    string) would fail.
    """
    from deriva_ml.execution import run_notebook

    sig = inspect.signature(run_notebook)
    assert "config_name" in sig.parameters, (
        "run_notebook() lost its config_name parameter entirely; "
        "the template's auto-derive examples assume it exists."
    )
    assert sig.parameters["config_name"].default is None, (
        "run_notebook(config_name=...) default is no longer None "
        f"(got {sig.parameters['config_name'].default!r}); the template's "
        "auto-derive examples assume calling with no positional arg works."
    )


def test_roc_notebook_uses_auto_derived_config_name() -> None:
    """``notebooks/roc_analysis.ipynb`` must not pass an explicit config name.

    The auto-derive convention is the example we ship. Catching a regression
    here is cheaper than catching it in review.
    """
    if not ROC_NOTEBOOK.exists():
        pytest.skip(f"{ROC_NOTEBOOK} not present; nothing to check.")

    nb = nbformat.read(ROC_NOTEBOOK, as_version=4)

    code_cells_calling_run_notebook = [
        cell
        for cell in nb.cells
        if cell.cell_type == "code" and "run_notebook(" in cell.source
    ]
    assert code_cells_calling_run_notebook, (
        "expected at least one code cell calling run_notebook(...) in "
        f"{ROC_NOTEBOOK.name}; found none."
    )

    # ``run_notebook("..."`` would be the explicit-string form. Anything else
    # (``run_notebook()``, ``run_notebook(workflow_type=...)``, etc.) is fine.
    explicit_string_pattern = re.compile(r"run_notebook\(\s*['\"]")
    for cell in code_cells_calling_run_notebook:
        assert not explicit_string_pattern.search(cell.source), (
            f"{ROC_NOTEBOOK.name} cell still passes a positional config-name "
            "string to run_notebook(); the example should use the auto-derive "
            f"form. Cell source:\n{cell.source}"
        )
