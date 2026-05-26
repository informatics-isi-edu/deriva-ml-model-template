# Notebook execution row carries the notebook-config description, ignoring the asset override

**Persona:** Analyst
**Phase:** Cross-channel verification of analysis execution F6C, 2026-05-26
**Severity:** Low
**Component:** `deriva-ml` notebook runner (`run_notebook` /
`deriva-ml-run-notebook`) and the
`Execution.description` it writes to the catalog

## What happened

The Analyst executed `notebooks/roc_analysis.ipynb` with a Hydra
override pointing at the new `roc_all_six` asset config:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
    --allow-dirty notebooks/roc_analysis.ipynb assets=roc_all_six
```

The notebook itself calls `run_notebook("roc_analysis", ...)` —
`"roc_analysis"` is the notebook config name, whose default
description (in `src/configs/roc_analysis.py`) is:

```python
notebook_config(
    "roc_analysis",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "roc_quick_vs_extended", "datasets": "no_datasets"},
    description="ROC curve analysis (default: quick vs extended training)",
)
```

After execution `F6C` completed, both channels report the same
description string:

```text
mcp__dev-localhost__deriva_ml_get_execution(catalog_id=18, execution_rid="F6C")
=> {"rid":"F6C", "description":"ROC curve analysis (default: quick vs extended training)", ...}
```

But the actual asset config that ran was **`roc_all_six`** — all
six Developer training runs. The Chaise UI for `F6C` therefore
displays a description that's a literal lie about the work the
execution did. Anyone looking at the catalog row to figure out
"what is this analysis" will be misdirected.

## Reproduction

```bash
# Against catalog 18, ROC notebook + a non-default asset override:
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
    --allow-dirty notebooks/roc_analysis.ipynb assets=roc_lr_sweep
# Execution row gets description = "ROC curve analysis (default:
# quick vs extended training)" even though it ran the lr_sweep.
```

The description is captured from the notebook config at execution
*open* time, before Hydra overrides are applied to the asset
group. It records the registered config-level description, not the
resolved-after-overrides description.

## Impact on the persona's work

Not blocking — the catalog data is correct (asset RIDs, output
files, configuration.json all reflect the actual run). Only the
human-readable description is misleading.

But: this is *exactly* the kind of catalog metadata an Analyst
reaches for when sweeping `find_executions(workflow_type="ROC
Analysis Notebook")` to decide which prior analysis is the right
one to compare against. A stale description directs the wrong
analyses into the comparison.

The Analyst's deliverable `docs/reports/2026-05-26-multipersona-analysis.md`
identifies the F6C execution by RID and asset-config name
explicitly, so this report is self-correcting; future analyses
won't be.

## Suggested classification

Bug / minor. Description capture should happen *after* Hydra
override resolution, or carry the resolved asset group name in
addition to the config-level description.

## Notes for the fix-pass

- Code site: `deriva-ml/src/deriva_ml/execution/run_notebook.py`
  (or wherever `Execution.description` is set from the
  `notebook_config` registration). Look for the point where the
  Hydra config is finalized — `description` should be derived
  from the *resolved* config dict, not the registration-time
  default.
- A reasonable description template:
  `f"{base_description} | assets={cfg.assets._name_} datasets={cfg.datasets._name_}"`.
- Sub-question: should the per-experiment Hydra `description`
  field (set via `with_description(...)`) be available in
  `cfg.assets.description`? If so, the runner could prefer that
  over the notebook-config-level default. The `roc_all_six`
  asset entry has its own description ("All 6 viable
  training-run predictions on CSA test set ..."); that's the
  string this execution row should be carrying.

## Related

- `findings/analyst/01-describe-vs-run-include-tables.md`
  (different surface, same flavor: a registration-time validation
  or string differs from a runtime resolution).
