# Task A1 Report — Baseline Worktree Setup

**Date:** 2026-06-23
**Status:** DONE_WITH_CONCERNS

## Worktree

- **Path:** `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip`
- **Branch:** `chore/strip-cifar-to-skeleton`
- **Tracks:** `origin/main`
- **Base commit SHA:** `3c8bd8251316e1a1dbf9d8c37b31a54d0cdedce7`
- **Commit message:** `feat: relocate design docs under docs/design/<entity>/, add feature + model (#63)`

## uv sync result

`uv sync` completed successfully. Final lines:

```
+ wcwidth==0.7.0
+ webcolors==25.10.0
+ webencodings==0.5.1
+ websocket-client==1.9.0
+ widgetsnbextension==4.0.15
```

## pytest results

### Invocation: `uv run python -m pytest -q` (from repo root — discovers scripts/ too)

```
ERROR scripts/test_bag_fk_traversal.py - FileNotFoundError: [Errno 2] No such...
14 warnings, 1 error in 187.29s (0:03:07)
```

The error is a **collection failure** (not a test failure). `scripts/test_bag_fk_traversal.py`
is an eye-ai regression script that has a module-level `DerivaML(hostname="www.eye-ai.org", ...)`
instantiation — when pytest collects it without test infrastructure running, it fails with
a `FileNotFoundError`. This script is **not part of the CIFAR test suite** and is a pre-existing
issue in the repo (it belongs in the eye-ai domain, not this template).

### Invocation: `uv run python -m pytest tests/ -q` (targeting only tests/)

```
68 passed, 2 skipped, 9 warnings in 7.16s
```

**All 68 CIFAR/config tests pass. 2 skipped (expected).** The actual test suite is green.

### Invocation: `uv run python -m pytest --ignore=scripts/ -q`

```
68 passed, 2 skipped, 9 warnings in 7.16s
```

Identical result — confirms the collection error is isolated to `scripts/test_bag_fk_traversal.py`.

## `--list-configs` output (CIFAR groups confirmed present)

```
Available Hydra Configuration Groups:
==================================================

Top-level configs:
  - deriva_base
  - deriva_model
  - roc_analysis
  - roc_epoch_sweep
  - roc_lr_batch_grid
  - roc_lr_sweep
  - roc_quick_vs_extended

assets:
  - default_asset
  - no_assets

datasets:
  - cifar10_complete
  - cifar10_labeled_split
  - cifar10_labeled_testing
  - cifar10_labeled_training
  - cifar10_small_labeled_split
  - cifar10_small_labeled_testing
  - cifar10_small_labeled_training
  - cifar10_small_testing
  - cifar10_small_training
  - cifar10_split
  - cifar10_testing
  - cifar10_training
  - default_dataset
  - no_datasets
  - none

deriva_ml:
  - default_deriva

experiment:
  - cifar10_default
  - cifar10_extended
  - cifar10_extended_full
  - cifar10_quick
  - cifar10_quick_full
  - cifar10_small_default
  - cifar10_small_large
  - cifar10_test_only

model_config:
  - cifar10_extended
  - cifar10_fast_lr
  - cifar10_large
  - cifar10_quick
  - cifar10_regularized
  - cifar10_slow_lr
  - cifar10_test_only
  - default_model

workflow:
  - cifar10_cnn
  - default_workflow
  - roc_analysis

multirun:
  - epoch_sweep: Training Duration (Epochs) Sweep
  - lr_batch_grid: Learning Rate and Batch Size Grid Search
  - lr_sweep: Learning Rate Hyperparameter Sweep
  - quick_vs_extended: CIFAR-10 CNN Multi-Experiment Comparison
  - quick_vs_extended_full: CIFAR-10 Full Dataset Comparison

==================================================
Pass a choice as a Hydra override, e.g.:
  deriva-ml-run model_config=cifar10_quick
  deriva-ml-run +experiment=cifar10_quick
To inspect the RESOLVED config a run would use (Hydra):
  deriva-ml-run +experiment=cifar10_quick --cfg job
```

All CIFAR config groups are present at baseline:
- datasets: 13 CIFAR configs + 3 generic
- experiment: 8 CIFAR experiments
- model_config: 8 CIFAR model configs
- workflow: cifar10_cnn + 2 generic
- multirun: 5 CIFAR sweep configs

## Concern

**Pre-existing issue:** `scripts/test_bag_fk_traversal.py` is an eye-ai domain script that
does not belong in this template repo (it references `www.eye-ai.org` and `eye-ai` catalog).
It causes `uv run python -m pytest` (root invocation) to fail with a collection error.
The actual CIFAR test suite (`tests/`) is fully green.

**Recommendation for subsequent tasks:** The strip plan should either:
1. Delete `scripts/test_bag_fk_traversal.py` as part of the CIFAR-unrelated cleanup, OR
2. Add `testpaths = ["tests"]` to `pyproject.toml` `[tool.pytest.ini_options]` so root pytest
   invocation doesn't collect `scripts/`.

This is a pre-existing issue — not introduced by the strip work. Document it in progress.md.

## No files modified

This task is setup + baseline only. No files were created or modified in the worktree.
