# Task A4 Report — Convert config files to generic scaffolds

Status: **DONE**

Worktree: `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip`
Branch: `chore/strip-cifar-to-skeleton` (no commit made, per task scope)

## Files changed / created (all in scope: src/configs/, pyproject.toml, tests/)

### Created
- **`src/configs/model.py`** (NEW) — registers the REQUIRED `model_config`
  group + `default_model` that `base.py`'s defaults list references. See
  approach below.

### Converted in place (CIFAR content → generic self-documenting scaffold)
- **`src/configs/datasets.py`** — generic docstring; keeps `DatasetSpecConfig`
  / `with_description` imports; live empties `no_datasets`, `none`,
  `default_dataset`; one commented `DatasetSpecConfig(rid="<your-rid>",
  version="<ver>")` example. All `cifar10_*` named configs removed.
- **`src/configs/experiments.py`** — generic docstring + `experiment_store =
  store(group="experiment", package="_global_")`; ONE live `default`
  experiment pairing `default_model` + `default_dataset` (so `--list-configs`
  shows ≥1 experiment); one commented `<your_model_config>`/`<your_dataset_group>`
  example. All 9 `cifar10_*` experiments removed.
- **`src/configs/workflow.py`** — no longer references `Cifar10CNNWorkflow` or
  `ROCAnalysisWorkflow` (deleted symbols). Live generic `DefaultWorkflow`
  (name="Model Run", type="Training") registered as `default_workflow`; one
  commented example.
- **`src/configs/assets.py`** — generic docstring; live `default_asset`,
  `no_assets`; one commented `<your-rid>` example. CIFAR/ROC names removed from
  the commented examples.
- **`src/configs/multiruns.py`** — generic docstring; ONE live `example_sweep`
  built on `+experiment=default` (runnable); one commented grid example. All
  `cifar10_*`/`roc_*` multiruns removed.
- **`src/configs/multirun_descriptions.py`** — generic docstring; one
  `EXAMPLE_SWEEP_DESCRIPTION` constant (consumed by `multiruns.py`); one
  commented example. All CIFAR sweep description constants removed.
- **`src/configs/roc_analysis.py` → `src/configs/analysis.py`** (git mv +
  rewrite) — was a notebook-config module for the deleted
  `notebooks/roc_analysis.ipynb`, still referencing the removed
  `roc_quick_vs_extended` asset group. Rewritten as a generic analysis-notebook
  scaffold: registers a single `analysis` notebook config (defaults
  `assets=no_assets`, `datasets=no_datasets`) with an `AnalysisConfig`
  dataclass + commented example. Renamed because `roc_analysis.py` registering
  an `analysis` config was stale/misleading. **In scope** (under src/configs/).
- **`src/configs/__init__.py`** — removed the unused, explicitly-deprecated
  `load_all_configs = lambda: ...` backwards-compat shim (also the sole ruff
  E731 error; verified unreferenced anywhere). Now just re-exports
  `load_configs` with `__all__`. Docstring examples were already generic.

### pyproject.toml
- `description` → "A template for building reproducible ML projects on
  DerivaML." (name kept `deriva-ml-model-template`).
- Removed the `[project.scripts]` `load-cifar10 = "scripts.load_cifar10:main"`
  block entirely (the section had only that one entry).
- Removed CIFAR/ROC-only deps from `[project].dependencies`: `torchvision`,
  `pandas`, `matplotlib`, `scikit-learn`. Verified safe: `grep -rln
  'pandas|matplotlib|sklearn|scikit|torchvision'` over `src/` matched only the
  egg-info build artifact, nothing in real source. Kept `deriva-ml`,
  `notebook`, `ipykernel`. Dev/docker/jupyter/torch/tensorflow groups
  untouched (the `torch` group still carries `torch`+`torchvision` for users
  who opt in).
- Genericized the now-stale dependency-pin comment that referenced "The
  CIFAR-10 hierarchy below".

### tests/
- **Deleted `tests/test_notebook_examples.py`** (inherited fact #3). It
  asserted that `notebooks/roc_analysis.ipynb` uses the auto-derived
  `run_notebook()` form — but that notebook was deleted by the prior task, so
  the test only protected an example that no longer ships. The companion
  upstream-signature check it also contained belongs to deriva-ml, not the
  template skeleton. No tests remain in `tests/` (all CIFAR tests were already
  removed by the prior task); a stripped skeleton with no example code has no
  smoke tests to run, which is expected.

## Generic `default_model` approach

`src/configs/cifar10_cnn.py` (deleted) was the sole registrar of the
`model_config` store group + the REQUIRED `default_model`. I authored a small
**runnable generic placeholder model** rather than a bare config stub:

- `example_model(learning_rate=1e-3, epochs=10, *, ml_instance=None,
  execution=None) -> None` — a no-op that satisfies the
  `src/models/model_protocol.py` (`DerivaMLModel`) interface: receives
  Hydra-configured fields as kwargs plus injected `ml_instance` / `execution`,
  returns `None`. Google-style docstring with Args/Returns/Example.
- `ExampleModelConfig = builds(example_model, learning_rate=1e-3, epochs=10,
  populate_full_signature=True, zen_partial=True)` — `zen_partial=True` leaves
  `ml_instance`/`execution` unbound for the runner to inject, matching the
  pattern from the deleted `Cifar10CNNConfig`.
- `model_store = store(group="model_config")`; `model_store(ExampleModelConfig,
  name="default_model")`.
- One commented variant showing an alternate hyperparameter set.

This keeps the skeleton dry-runnable end to end: `+experiment=default --cfg
job` resolves `model_config` to `configs.model.example_model` (partial, params
intact). The docstring directs users to "replace the target with your model
function (implement the src/models/model_protocol.py interface)".

## Deps removed
`torchvision`, `pandas`, `matplotlib`, `scikit-learn` from
`[project].dependencies` (none imported by remaining source). Plus the
`load-cifar10` console script.

## FULL output of the four verify commands

### 1) `uv sync 2>&1 | tail -3`
```
Resolved 246 packages in 7ms
Checked 166 packages in 28ms
```

### 2) `uv run python -c "import configs" 2>&1 | tail -5`
```
(no output; exit=0 — import succeeds)
```

### 3) `grep -rilE 'cifar' src/ 2>&1 || echo "(no cifar refs in src/)"`
```
(no cifar refs in src/)
```
(Note: the git-ignored `src/*.egg-info/` build artifact was regenerated via
`uv sync` after the pyproject edits so even the stale artifact carries no CIFAR
strings.)

### 4) `deriva-ml-run --list-configs` (warnings filtered)
```
Available Hydra Configuration Groups:
==================================================

Top-level configs:
  - analysis
  - deriva_base
  - deriva_model

assets:
  - default_asset
  - no_assets

datasets:
  - default_dataset
  - no_datasets
  - none

deriva_ml:
  - default_deriva

experiment:
  - default

model_config:
  - default_model

workflow:
  - default_workflow

multirun:
  - example_sweep: Example Sweep

==================================================
Pass a choice as a Hydra override, e.g.:
  deriva-ml-run model_config=cifar10_quick
  deriva-ml-run +experiment=cifar10_quick
To inspect the RESOLVED config a run would use (Hydra):
  deriva-ml-run +experiment=cifar10_quick --cfg job
```

## Extra verification
- `ruff check src/configs/` → All checks passed. `ruff format --check` → 12
  files already formatted.
- `+experiment=default --cfg job` resolves the full tree:
  `model_config._target_ = configs.model.example_model` (`_partial_: true`,
  learning_rate/epochs intact), `workflow.name = "Model Run"`, empty
  datasets/assets — confirming the skeleton is wired and dry-runnable.

## Concerns
- The trailing hint lines in `--list-configs` output
  (`model_config=cifar10_quick`, `+experiment=cifar10_quick`) are **hardcoded
  in the deriva-ml CLI itself**, not in this template — out of A4 scope. Worth
  a follow-up in deriva-ml if those example strings should be generic.
- `tests/` is now empty of test files. Intentional for the stripped skeleton,
  but a later task may want to add a generic `tests/test_configs_load.py` smoke
  test (the CIFAR one was deleted) so the template ships with a green example
  suite. Flagged as out-of-scope for A4.
