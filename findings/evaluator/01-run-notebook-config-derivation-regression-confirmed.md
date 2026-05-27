# `run_notebook()` config-name auto-derivation regression confirmed under `deriva-ml-run-notebook`

**Persona:** Evaluator (verification of Analyst finding `analyst/01`)
**Severity:** **High**
**Category:** **Bug** (in code that just shipped)
**Component:** `deriva-ml` — `run_notebook` + `_derive_config_name_from_notebook`

## What the Analyst reported

`run_notebook()` is documented as auto-deriving its Hydra `config_name` from the calling notebook's filename. The Analyst observed that this works interactively in Jupyter but fails under `deriva-ml-run-notebook`, because the latter calls `pm.execute_notebook(...)` programmatically and never sets `PAPERMILL_INPUT_PATH` in `os.environ`. They worked around it by passing the config name explicitly:

```python
ml, execution, config = run_notebook("roc_analysis", ...)
```

See `findings/analyst/01-run-notebook-config-derivation-fails-under-papermill.md`.

## Why I'm upgrading this to a separate evaluator finding

This isn't just persona friction the Analyst routed around. It's a real regression in code that **shipped today (PR #248 / commit `6ed68d08`)**, and the project's only ergonomic-improvement runner is exactly the one that breaks the contract the PR introduced.

Two failure modes compound:

1. **The auto-derive feature is dead for the only headless path the project ships.** `run_notebook.py:633` calls `pm.execute_notebook(input_path=..., output_path=..., parameters=..., kernel_name=..., log_output=...)`. It does NOT set `PAPERMILL_INPUT_PATH` in the environment, and `pm.execute_notebook()` (unlike papermill's CLI) does not set it either. The first auto-derive strategy (`_derive_config_name_from_notebook`, line 522 of `base_config.py`) silently misses. The second strategy (call-stack walk for `__file__` ending in `.ipynb`, lines 530–537) also misses because papermill's kernel-side frames don't expose the notebook path that way. The function raises `ValueError`, and every notebook in the repo that omits `config_name=` becomes unrunnable through `deriva-ml-run-notebook`.

2. **The docstring is actively misleading.** `base_config.py:494–498` says PAPERMILL_INPUT_PATH "is the most reliable signal because it survives Jupyter's globals-namespace abstractions." That sentence is true for `papermill <input.ipynb>` (the CLI). It's **false** for `pm.execute_notebook()` (the API), which is the only way the deriva-ml runner invokes papermill. A future agent reading the docstring will draw the wrong conclusion about why the function fails.

## Verification

I read the offending code directly and confirmed:

- `deriva_ml/run_notebook.py:618` sets `os.environ["DERIVA_ML_NOTEBOOK_PATH"]` (deriva-ml's own env var) but never `PAPERMILL_INPUT_PATH`.
- `deriva_ml/run_notebook.py:633` calls `pm.execute_notebook(...)` without parameters that would inject `PAPERMILL_INPUT_PATH` into the kernel.
- `deriva_ml/execution/base_config.py:522` reads `os.environ.get("PAPERMILL_INPUT_PATH")` — yields `None` under `deriva-ml-run-notebook`.
- The call-stack fallback (lines 530–537) does not pick up papermill's kernel frames.

The Analyst's diagnosis is exactly right.

## Why it's high (not medium)

- **It's in code that just merged today.** PR #248 added the auto-derivation as a quality-of-life improvement. The default invocation path of the feature does not work. This is a *correctness* regression in the headline ergonomic of a brand-new feature.
- **The fix is one line.** Either set `os.environ["PAPERMILL_INPUT_PATH"] = notebook_file.as_posix()` before `pm.execute_notebook(...)` in `run_notebook.py`, or have `_derive_config_name_from_notebook` also consult `os.environ.get("DERIVA_ML_NOTEBOOK_PATH")` — which `run_notebook.py:618` already sets unconditionally. The second option is arguably cleaner because it doesn't lean on papermill-internal env-var semantics.
- **Every notebook in the template repo is affected.** Anyone who follows the PR #248 ergonomic pattern (`run_notebook()` with no args) and then tries to execute the notebook via `deriva-ml-run-notebook` will hit this. The interactive Jupyter path works, so the issue won't be caught by interactive testing — it only surfaces in headless / CI / multipersona-e2e contexts.

## Reproduction

From a clean clone of this worktree:

```
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
  --allow-dirty notebooks/<name>.ipynb
```

where `<name>.ipynb` calls `run_notebook()` without an explicit `config_name=`. Hits:

```
ValueError: config_name is required when run_notebook() is called outside
a notebook context. Pass it explicitly, e.g. run_notebook('my_config').
```

## Suggested fix (sketch)

Minimal change in `_derive_config_name_from_notebook`:

```python
papermill_path = os.environ.get("PAPERMILL_INPUT_PATH")
if papermill_path:
    return Path(papermill_path).stem
# Fallback: deriva-ml-run-notebook sets DERIVA_ML_NOTEBOOK_PATH in
# run_notebook.py before calling pm.execute_notebook (papermill's
# Python API doesn't surface PAPERMILL_INPUT_PATH the way its CLI does).
deriva_path = os.environ.get("DERIVA_ML_NOTEBOOK_PATH")
if deriva_path:
    return Path(deriva_path).stem
# ... existing call-stack fallback ...
```

This is consistent with the existing convention (`run_notebook.py` already sets `DERIVA_ML_NOTEBOOK_PATH` for downstream consumers; see `execution/workflow.py:430`). The docstring at `base_config.py:494–498` should be updated to mention `DERIVA_ML_NOTEBOOK_PATH` alongside `PAPERMILL_INPUT_PATH`, and to clarify that the latter is only set by the papermill CLI, not by `pm.execute_notebook()`.

## Disposition recommendation

**Fix inline now.** The fix is trivial, the regression is in code that just shipped, and the Analyst's workaround pollutes every notebook in the repo with a redundant `config_name` argument that the PR was specifically designed to eliminate. If the fix lands, the Analyst's edit to `notebooks/roc_analysis.ipynb` can be reverted as cleanup, which restores PR #248's ergonomic improvement to the headless runner.
