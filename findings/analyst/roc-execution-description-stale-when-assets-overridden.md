# Finding: recorded ROC analysis execution description misdescribes the run when `assets=` is overridden

- **Persona:** Analyst
- **Date:** 2026-06-01
- **Catalog:** localhost / catalog 2 / schema `e2e-test-20260601`
- **Severity:** Low (provenance readability; the override suffix carries the truth, the prose half does not)
- **Category:** Template config ergonomics / execution-record readability

## What I expected

I ran the ROC notebook against the three-way comparison config:

```bash
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb assets=roc_modeler_e2e_three_way
```

I expected the resulting execution record's `Description` to describe
*what actually ran* — a three-way comparison of the smoke / regularized /
fast_lr runs on PK6 — so a future reader scanning `list_executions()`
could tell at a glance what this analysis was.

## What actually happened

The recorded analysis execution
([REJ](https://localhost/id/2/REJ@356-DD8X-BD5W)) came out with:

```
ROC curve analysis (default: quick vs extended training) [overrides: assets=roc_modeler_e2e_three_way]
```

The prose half — "ROC curve analysis (default: quick vs extended
training)" — is the literal `description=` string baked into
`notebook_config("roc_analysis", ...)` in `src/configs/roc_analysis.py`.
It is **not** what ran. There is no "quick vs extended" comparison in this
catalog at all (those are template-era asset groups; the actual assets are
`roc_modeler_e2e_three_way` = QN6/QY8/R7A). A reader who trusts the prose
and skips the bracketed override suffix is actively misled about the
analysis.

The override suffix (`[overrides: assets=roc_modeler_e2e_three_way]`) is
correct and does carry the real information — so the provenance is not
*lost*, just buried behind a misleading lead.

## Repro

```bash
cd <worktree>
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb assets=roc_modeler_e2e_three_way
# -> execution Description begins "ROC curve analysis (default: quick vs extended training)"
#    regardless of which assets= group was actually selected.
```

The root cause: the notebook calls `run_notebook(workflow_type="ROC
Analysis Notebook")` with no per-run description, so deriva-ml composes the
description from the *config's static `description=`* plus the resolved
overrides. The static string names the default asset group ("quick vs
extended") as if it were always the subject, but it's only the default —
overriding `assets=` doesn't update the prose.

## Impact

- A scan of `list_executions()` shows an analysis whose human-readable
  description names a comparison that didn't happen. The reader must parse
  the bracketed override suffix to recover the truth.
- This is the *consumer* mirror of the Modeler's
  `hydra-description-override-grammar` finding: the Modeler couldn't easily
  set a rich free-text description from the CLI; here the notebook path
  *does* set one automatically, but it's a static string that goes stale
  the moment a non-default `assets=` group is used.

## Workaround applied

None needed for the analysis itself — the metrics, figures, and lineage
are all correct, and the override suffix records the true asset group. For
this report I cite the run by RID + the actual asset group rather than by
its description text.

## Suggested direction (NOT done — out of scope for this arc)

Either (a) make the `roc_analysis` config description generic ("ROC curve
analysis of the selected prediction-probability assets") so it doesn't
falsely claim a specific comparison, or (b) register a dedicated
`notebook_config("roc_modeler_e2e_three_way", ...)` with an accurate
description and reference it from the notebook's `run_notebook("<name>")`
call — though note the CLI cannot swap the config name (the `--config*`
trap), so (b) requires editing the notebook's first cell. Option (a) is the
lower-touch fix and removes the false-specificity for *all* override
groups at once.
