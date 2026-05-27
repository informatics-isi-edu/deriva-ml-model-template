# `run_notebook()` config-name auto-derivation fails under `deriva-ml-run-notebook`

**Persona:** Analyst
**Phase:** Trying to execute `notebooks/roc_analysis.ipynb` against the
Modeler's Family A triplet via `deriva-ml-run-notebook` so the ROC analysis
is captured as a catalog execution with provenance.

## What happened

`run_notebook()` is documented (in `roc_analysis.ipynb` cell 2 and in
`src/configs/roc_analysis.py`) as auto-deriving the Hydra config name from
the calling notebook's filename. That works interactively in Jupyter and
when running notebooks through `papermill`'s **CLI**, but fails when the
notebook is launched via `deriva-ml-run-notebook` (which calls
`pm.execute_notebook(...)` programmatically).

Command:

```
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
  --allow-dirty notebooks/roc_analysis.ipynb \
  assets=modeler_familyA_triplet
```

Failure (notebook cell 2):

```
ValueError: config_name is required when run_notebook() is called outside
a notebook context. Pass it explicitly, e.g. run_notebook('my_config').
```

`_derive_config_name_from_notebook` (in
`deriva_ml/execution/base_config.py:484`) tries two strategies:

1. Read `os.environ.get("PAPERMILL_INPUT_PATH")`.
2. Walk the call stack looking for a frame whose `__file__` ends in
   `.ipynb`.

Both fail here. Strategy 1 fails because `PAPERMILL_INPUT_PATH` is set by
papermill's **CLI** (`papermill/cli.py:217`) as a *notebook parameter*
(injected into the notebook's globals), **not** as an OS env var, and only
when invoking the CLI directly. `deriva-ml-run-notebook` bypasses the CLI
and calls `pm.execute_notebook(input_path=..., ...)` (see
`deriva_ml/run_notebook.py:633`), so `PAPERMILL_INPUT_PATH` is never set
in `os.environ`. Strategy 2 fails because papermill's kernel-side execution
frames don't expose the notebook path as `__file__` on the cell's globals
when launched this way.

Net effect: the *only* runner that ships with deriva-ml for executing
analysis notebooks is exactly the one that doesn't satisfy the
auto-derivation contract. The user-facing instruction
"`run_notebook()` derives the config name from the calling notebook's
filename" (notebook cell 2 markdown) is true only for interactive use.

## Reproduction

1. From a clean clone of this worktree, `uv sync`.
2. Pick any notebook in `notebooks/` whose first cell calls
   `run_notebook()` without an explicit `config_name=`.
3. Run:

   ```
   DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
     --allow-dirty notebooks/<name>.ipynb
   ```

4. Cell 2 raises `ValueError: config_name is required when run_notebook()
   is called outside a notebook context`.

## Workaround (used during this arc)

Edited the notebook to pass the config name explicitly:

```python
ml, execution, config = run_notebook(
    "roc_analysis",
    workflow_type="ROC Analysis Notebook",
)
```

Notebook continues to work interactively and now also works under
`deriva-ml-run-notebook`. The CIFAR10.md guidance about the "rare"
case where filename and config name diverge is now the *common* case
for headless runs.

## Notes

Two possible fixes (out of scope for this arc):

1. Have `deriva-ml-run-notebook` set `os.environ["PAPERMILL_INPUT_PATH"]`
   to the absolute notebook path before calling `pm.execute_notebook`,
   and have papermill propagate the kernel env (it does by default via
   `KernelManager.start_kernel`). One-line fix in
   `deriva_ml/run_notebook.py`.

2. Pass `PAPERMILL_INPUT_PATH` as an extra parameter to
   `pm.execute_notebook(parameters=...)` -- papermill's CLI already
   does this -- and teach `_derive_config_name_from_notebook` to look
   for it in the cell's globals (it can read the frame's
   `f_globals.get("PAPERMILL_INPUT_PATH")`). Slightly heavier but
   matches papermill's own convention.

Either way, the docstring at `base_config.py:494-498` is wrong as
stated -- `PAPERMILL_INPUT_PATH` is **not** "the most reliable signal
because it survives Jupyter's globals-namespace abstractions" *when
called via `pm.execute_notebook`*; it's not set at all in that path.

Asset RIDs and execution RIDs touched while reproducing:
- catalog 2 (`localhost`, `e2e-test-20260527e`)
- assets group: `modeler_familyA_triplet` -> [Y1M, Z3P, 105R]
