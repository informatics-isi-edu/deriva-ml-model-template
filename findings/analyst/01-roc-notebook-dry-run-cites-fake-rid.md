# roc_analysis notebook fails on dry_run because cell 3 cites the fake `0000` execution RID

**Persona:** Analyst
**Phase:** Pre-flight: dry-run the ROC notebook against the Toronto predictions
asset group (`assets=toronto_predictions`) before doing the real run.

## What happened

Ran:

```
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
    notebooks/roc_analysis.ipynb \
    assets=toronto_predictions \
    dry_run=true
```

Cell 3 (the post-`run_notebook()` connection / configuration display
cell) raises `DerivaMLException: Entity RID does not exist` on this
line:

```python
display(Markdown(f"""...
**Execution:** [{execution.execution_rid}]({ml.cite(execution.execution_rid)})
..."""))
```

Trace tail:

```
HTTPError: 404 Client Error: Not Found for url:
    https://localhost/ermrest/catalog/27/entity_rid/0000
    Details: Resource not found. Detail: entity with RID=0000

KeyError: '0000'

DerivaMLException: Invalid RID 0000

DerivaMLException: Entity RID does not exist
```

Under `dry_run=true`, `run_notebook()` returns an `execution` whose
`execution_rid` is the placeholder string `"0000"` — no execution row is
created on the catalog. Calling `ml.cite("0000")` then tries to resolve
it against `https://localhost/ermrest/catalog/27/entity_rid/0000`, which
404s, and the notebook crashes before any of the analysis cells run.

The notebook is otherwise structured to handle `dry_run` cleanly — the
downstream cells just use `execution.asset_paths` and
`execution.asset_file_path(...)`, which work fine without a real
execution row. The crash is in the *informational header* cell, which
makes the dry-run path strictly less useful than a real run.

## Reproduction

```
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
    notebooks/roc_analysis.ipynb \
    assets=toronto_predictions \
    dry_run=true
```

Crashes in cell 3 with the trace above. Removing the `ml.cite(...)`
call from the cell (or guarding it with `if not config.dry_run`) would
let the rest of the notebook execute.

## Notes

Workaround the Analyst used: skipped the dry-run gate and ran the
notebook for real (`dry_run=false`, the default). The real run created
a valid execution row, `ml.cite()` resolved, and the notebook completed
end-to-end.

This is the second observed instance of the "informational cell trips on
a fake RID during dry_run" pattern; the same cell shape appears in the
roc_analysis template. A general fix would be either:

1. Make `ml.cite()` return a placeholder string (e.g. the literal RID
   or `"dry-run"`) when the RID is the dry-run sentinel `"0000"`, or
2. Guard the citation in the template notebook cell with a
   `config.dry_run` check.

Option 1 is the more durable fix because it covers every notebook
template that follows the same display-the-execution-link pattern.

Detected during the 2026-05-28 e2e run; sibling versions per the
Curator findings (deriva-ml v1.40.2, deriva-ml-mcp v0.5.9,
deriva-mcp-core latest main, deriva-skills v1.2.4,
deriva-ml-skills v1.4.11).
