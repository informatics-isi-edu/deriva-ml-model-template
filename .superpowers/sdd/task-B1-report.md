# Task B1 Report — Assemble `deriva-ml-cifar-example` (local, fresh history)

**Status:** DONE_WITH_CONCERNS (one minor lint fix applied to staged source — see Concerns)
**Repo:** `/Users/carl/GitHub/DerivaML/deriva-ml-cifar-example`
**Initial commit SHA:** `0e3aed510a75da6703f966f0118e0822a55583b3`

---

## 1. File listing (new repo, excluding `.git/`)

```
./.github/release-drafter.yml
./.github/workflows/publish-docs.yml
./.github/workflows/release.yml
./.gitignore
./CIFAR10.md
./CLAUDE.md
./Experiments.md
./README.md
./mkdocs.yml
./notebooks/roc_analysis.ipynb
./pyproject.toml
./scripts/assets.toml
./scripts/upload_assets.py
./src/configs/__init__.py
./src/configs/assets.py
./src/configs/base.py
./src/configs/cifar10_cnn.py
./src/configs/dataset_generation.py
./src/configs/datasets.py
./src/configs/deriva.py
./src/configs/experiments.py
./src/configs/multirun_descriptions.py
./src/configs/multiruns.py
./src/configs/roc_analysis.py
./src/configs/workflow.py
./src/models/cifar10_classes.py
./src/models/cifar10_cnn.py
./src/models/model_protocol.py
./src/scripts/__init__.py
./src/scripts/_cifar10_assets.py
./src/scripts/_cifar10_datasets.py
./src/scripts/_cifar10_schema.py
./src/scripts/_cifar10_source.py
./src/scripts/analyst_join.py
./src/scripts/load_cifar10.py
./tests/test_analyst_join.py
./tests/test_cifar10_assets.py
./tests/test_cifar10_cnn_loaders.py
./tests/test_cifar10_datasets.py
./tests/test_cifar10_schema.py
./tests/test_cifar10_source.py
./tests/test_configs_load.py
./tests/test_load_cifar10_retry.py
./tests/test_load_cifar10_split_no_leakage.py
./tests/test_runner_bag_dispatch.py
./tests/test_runner_seed.py
./uv.lock
```

`src/*.egg-info` was absent in staging (glob matched nothing — already clean).
`.gitignore` already contained `*.egg-info/` (line 24), so the conditional
append was a no-op; no duplicate line was added.

## 2. pyproject.toml identity edits

- `name = "deriva-ml-model-template"` → `name = "deriva-ml-cifar-example"`
  (also dropped the stale `# Change this to be...` comment).
- `description = "A test template for using DerivaML"` →
  `description = "CIFAR-10 reference example for DerivaML, built on the deriva-ml-model-template skeleton."`
- KEPT `[project.scripts]` `load-cifar10 = "scripts.load_cifar10:main"`.
- KEPT CIFAR deps — verified present in `[project].dependencies`:
  `torchvision`, `pandas`, `matplotlib`, `scikit-learn` (all came from the
  main-derived staged pyproject).

## 3. README.md reframing

- Title `# DerivaML Model Template` → `# DerivaML CIFAR-10 Example`.
- Intro rewritten to position this as the **worked CIFAR-10 reference
  example built on the `deriva-ml-model-template` skeleton**, with a link
  back to https://github.com/informatics-isi-edu/deriva-ml-model-template
  and an explicit "to start a new project, fork the template instead" note.
- Quick Start step 1 "Create Your Repository" (which told users to fork
  *this* repo as a GitHub template) → "Clone This Example" with a `git clone`
  of `deriva-ml-cifar-example`, plus a pointer to the template for new
  projects.
- All CIFAR-specific usage content kept intact (it's the worked example).
- CLAUDE.md (CIFAR-flavored, staged from the old template) kept as-is —
  it is appropriate for this repo.

## 4. Files copied from the strip worktree

**None.** Every `from configs...` / `from scripts...` import resolved
unchanged — the package layout is identical to the original. No missing
generic files were needed.

## 5. Verification output (standalone, new repo)

### `uv sync`
```
Resolved 252 packages in 7ms
Checked 180 packages in 58ms
```
(clean; exit 0)

### `uv run python -m pytest tests/ -q`
```
66 passed, 2 skipped, 9 warnings in 7.65s
```
(The CIFAR suite. The `test_cifar_canonical_partition` fixtures already use
the correct `Image.Filename` case since the source came from main. Run with
`DERIVA_ML_ALLOW_DIRTY=true` because the tree was uncommitted at verify time.)

### `uv run deriva-ml-run --list-configs` (CIFAR groups present)
```
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
```

### `uv run ruff check src tests`
```
All checks passed!
```

## 6. Concern — one lint fix applied to staged source

The staged `src/configs/__init__.py` carried a pre-existing `E731` ruff
violation:

```python
load_all_configs = lambda: load_configs("configs")
```

This is a lambda assigned to a name; ruff rejects it. The staged test
`tests/test_configs_load.py::test_load_all_configs_registers_expected_groups`
**actively imports and calls** `load_all_configs`, so it is not dead code and
could not simply be deleted. I rewrote it as a proper documented `def`
(Google-style docstring with `Returns:`/`Example:`), preserving identical
behavior. After the fix: ruff `All checks passed!`, all 66 tests still pass,
`ruff format --check` reports the file already formatted.

Note: the de-CIFAR'd strip worktree (`deriva-ml-model-template-strip`) had
already removed this lambda entirely and rewired its `__init__.py`/test to the
skeleton shape (`__all__ = ["load_configs"]`, no `load_all_configs`). That
skeleton state is correct for the *template*, not for the CIFAR example, so I
did **not** import the strip's version — the CIFAR repo keeps `load_all_configs`
(now as a `def`) because its CIFAR test depends on it. No file was copied from
the strip worktree.

## 7. Commit

```
0e3aed5 Initial commit: CIFAR-10 reference example for DerivaML
```

Full SHA: `0e3aed510a75da6703f966f0118e0822a55583b3`

Working tree after commit: **clean** (`git status --short` empty).
No git remote configured (`git remote -v` empty) — local only, as required.
