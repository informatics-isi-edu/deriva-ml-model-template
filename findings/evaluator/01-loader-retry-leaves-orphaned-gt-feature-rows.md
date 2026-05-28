# Loader retry leaves orphaned Image_Classification rows; downstream landmines confirmed in two places

**Persona:** Evaluator
**Severity:** High
**Category:** Bug
**Phase:** Cross-arc synthesis (Curator-01 + Analyst-02 share a root cause)

## What happened

The Curator filed two findings (`findings/curator/01`,
`findings/curator/02`) and the Analyst filed one
(`findings/analyst/02`) that all trace back to a single upstream
defect in `load-cifar10`: **when the loader's `datasets` phase fails
after the `images` phase has already succeeded, retrying with new
`--num-images` settings does not delete the prior pass's
`Image_Classification` feature rows.** On this catalog the first
attempt (execution `854`, `--num-images 500`) left 500 rows behind;
the successful retry (`HSR`, `--num-images 1100`) then wrote a
*fresh* 1100 rows over the same images, producing 1600 GT rows for
1100 unique images (`1600 = 500 + 1100`; verified directly:
`count(Execution=854)=500`, `count(Execution=HSR)=1100`).

That cost was paid in two non-obvious places downstream:

1. **`_cifar10_datasets.py`'s `split_dataset(row_per=feature_table)`**
   produced TCM/TCY with 33 image-RID overlap and VAY/VB8 with 24 —
   100% of the overlapping images were exactly the doubly-tagged
   set. The Curator (`findings/curator/02`) correctly identified
   the proximate mechanism (the splitter partitions feature *rows*
   when `row_per` points at a feature table, so an image with two
   feature rows can land on both sides) and the durable fix
   (`row_per="Image"` + dedupe upstream); the root cause is still
   the loader leaving orphaned rows.
2. **`notebooks/roc_analysis.ipynb`'s GT-selection heuristic**
   (`with_confidence == 0` then `.index[0]`) picked `854` instead
   of `HSR` and analysed only 250/550 M1G test images — a 55%
   sample loss that was visible in the per-model headings but
   silently propagated through ROC curves, confusion matrices,
   and the catalog-stored `roc_analysis.md` export. Analyst routed
   around with `scripts/build_joined_wide_table.py`. Reported in
   `findings/analyst/02`.

The Curator named the trap in `tacit-knowledge.md` `tk-001` before
the Modeler picked a dataset, which is exactly how the platform is
*supposed* to absorb this kind of upstream defect — but a future
catalog won't have a Curator arc, and even on *this* catalog the
roc_analysis notebook tripped on it anyway.

## Why upgraded to High / Bug

- **Two persona findings + one tacit-knowledge entry tracing to one
  cause** is the load-bearing definition of a High finding: a
  primary deliverable (the roc_analysis notebook the project ships
  as its canonical analysis template) is materially compromised on
  any catalog that has ever seen a loader retry, and the framework
  the platform *encourages* (split_dataset on a feature table) is
  unsafe by construction in that same scenario.
- **The hazard is silent on a fresh-load catalog and only surfaces
  after a retry.** The first run looks clean; the second run is
  where the landmines arm themselves. That is the worst kind of
  data-quality bug to ship — it only fires for users whose
  workflow includes the very recovery path the loader was supposed
  to enable.
- **Workaround exists but is fragile.** "Filter to a known-good
  execution RID" is fine when a human has already characterised
  the catalog; it's not actionable when a downstream tool
  (notebook, MCP query, third-party script) is doing the read.

## Reproduction

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id=27)
ds = ml.catalog.getPathBuilder().schemas["e2e-test-20260528"]
feat = ds.Execution_Image_Image_Classification.entities().fetch()
# Direct counts:
#   - total feature rows: 1600 (Curator finding; before Modeler arc)
#                          OR 3250 today (after 3 Modeler training runs added 1650 prediction rows)
#   - rows from Execution=854: 500    (orphaned)
#   - rows from Execution=HSR: 1100   (canonical)
#   - unique Image RIDs touched: 1100
```

## Suggested fix

**Loader-side (the root):** before re-writing
`Image_Classification` rows in a `--phase images` run, delete prior
rows for the schema-and-feature pair. Two viable shapes:

1. **Truncate-and-replace:** treat `--phase images` as a fresh
   write — delete *all* prior `Image_Classification` rows for the
   target schema before inserting the new pass's rows. Simplest;
   loses no information that the new pass doesn't already replace.
2. **Skip-if-present:** check per-image whether a class has already
   been recorded; skip those rows. Preserves both passes as
   provenance; relies on `Image_Classification` not being
   semantically "per execution" (it isn't — see tk-001).

(1) is the cleaner fix for an idempotent loader; (2) is the
backwards-compatible fix for a project that wants the failed-attempt
trail preserved for forensic reasons.

**Notebook-side (defence in depth, even if loader is fixed):**
when multiple GT-candidate executions exist, pick the one with the
most rows (Analyst's one-line fix in `findings/analyst/02`) or take
an explicit `gt_execution=...` parameter on the notebook config.

Both fixes belong in a fix-pass.

Detected during the 2026-05-28 e2e run with sibling versions:
deriva-ml v1.40.2, deriva-ml-mcp v0.5.9, deriva-mcp-core latest main,
deriva-skills v1.2.4, deriva-ml-skills v1.4.11.
