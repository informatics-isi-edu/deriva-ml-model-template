# Task A2 — CIFAR Staging Report

**Status:** DONE_WITH_CONCERNS  
**Date:** 2026-06-23  
**Worktree:** `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip` (branch `chore/strip-cifar-to-skeleton`)  
**Staging dir:** `/Users/carl/GitHub/DerivaML/_cifar-staging/`  
**Total files staged:** 47 (44 explicitly listed + 3 inside `.github/workflows/`)

---

## Concerns

1. **`.python-version` not found in source** — the file does not exist in the worktree. Not staged. The new repo will need to add one or rely on `pyproject.toml`'s `requires-python` instead.

2. **`.github` needed `-a` flag** — initial `rsync -R .github …` silently skipped the directory contents (rsync without `-a` does not recurse into directories). Fixed by re-running with `rsync -aR .github …`. Result confirmed correct (3 files inside).

---

## Full file listing

```
/Users/carl/GitHub/DerivaML/_cifar-staging/.github/release-drafter.yml
/Users/carl/GitHub/DerivaML/_cifar-staging/.github/workflows/publish-docs.yml
/Users/carl/GitHub/DerivaML/_cifar-staging/.github/workflows/release.yml
/Users/carl/GitHub/DerivaML/_cifar-staging/.gitignore
/Users/carl/GitHub/DerivaML/_cifar-staging/CIFAR10.md
/Users/carl/GitHub/DerivaML/_cifar-staging/CLAUDE.md
/Users/carl/GitHub/DerivaML/_cifar-staging/Experiments.md
/Users/carl/GitHub/DerivaML/_cifar-staging/README.md
/Users/carl/GitHub/DerivaML/_cifar-staging/mkdocs.yml
/Users/carl/GitHub/DerivaML/_cifar-staging/notebooks/roc_analysis.ipynb
/Users/carl/GitHub/DerivaML/_cifar-staging/pyproject.toml
/Users/carl/GitHub/DerivaML/_cifar-staging/scripts/assets.toml
/Users/carl/GitHub/DerivaML/_cifar-staging/scripts/upload_assets.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/__init__.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/assets.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/base.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/cifar10_cnn.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/dataset_generation.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/datasets.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/deriva.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/experiments.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/multirun_descriptions.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/multiruns.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/roc_analysis.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/configs/workflow.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/models/cifar10_classes.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/models/cifar10_cnn.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/models/model_protocol.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/scripts/__init__.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/scripts/_cifar10_assets.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/scripts/_cifar10_datasets.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/scripts/_cifar10_schema.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/scripts/_cifar10_source.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/scripts/analyst_join.py
/Users/carl/GitHub/DerivaML/_cifar-staging/src/scripts/load_cifar10.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_analyst_join.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_cifar10_assets.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_cifar10_cnn_loaders.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_cifar10_datasets.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_cifar10_schema.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_cifar10_source.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_configs_load.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_load_cifar10_retry.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_load_cifar10_split_no_leakage.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_runner_bag_dispatch.py
/Users/carl/GitHub/DerivaML/_cifar-staging/tests/test_runner_seed.py
/Users/carl/GitHub/DerivaML/_cifar-staging/uv.lock
```

---

## Category breakdown

| Category | Files | Status |
|---|---|---|
| CIFAR configs (7) | `src/configs/cifar10_cnn.py`, `datasets.py`, `experiments.py`, `assets.py`, `workflow.py`, `multiruns.py`, `multirun_descriptions.py` | All present |
| CIFAR models (2) | `src/models/cifar10_cnn.py`, `cifar10_classes.py` | All present |
| CIFAR scripts (6) | `src/scripts/_cifar10_assets.py`, `_cifar10_datasets.py`, `_cifar10_schema.py`, `_cifar10_source.py`, `load_cifar10.py`, `analyst_join.py` | All present |
| CIFAR notebook (1) | `notebooks/roc_analysis.ipynb` | Present |
| CIFAR scripts/data (1) | `scripts/assets.toml` | Present |
| CIFAR docs (2) | `CIFAR10.md`, `Experiments.md` | All present |
| CIFAR tests (11) | All 11 test files | All present |
| Generic scaffolding (8) | `src/configs/base.py`, `deriva.py`, `__init__.py`, `dataset_generation.py`, `roc_analysis.py`; `src/models/model_protocol.py`; `src/scripts/__init__.py`; `scripts/upload_assets.py` | All present |
| Top-level docs (3) | `README.md`, `CLAUDE.md`, `mkdocs.yml` | All present |
| CI / dotfiles (4) | `.github/` (3 files), `.gitignore`, `pyproject.toml`, `uv.lock` | Present; `.python-version` absent from source |

---

## Missing files

- `.python-version` — **does not exist** in the source worktree. Not a copy failure; the file was never there. The new repo can derive Python version from `pyproject.toml`'s `requires-python`.

No excluded files (`scripts/test_bag_fk_traversal.py`, `*.egg-info/`) were staged — confirmed absent from listing.
