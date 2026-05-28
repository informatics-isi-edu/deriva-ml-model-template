# roc_analysis notebook silently picks the wrong GT execution and analyses 250/550 of M1G

**Persona:** Analyst
**Phase:** Run the ROC notebook against the Toronto-pair predictions
(`assets=toronto_predictions`); cross-check the numbers against the
Modeler's emission-time accuracies.

## What happened

The notebook's ground-truth-execution heuristic (cell 9) picks the
**first execution it finds with no confidence scores**:

```python
gt_mask = exec_summary['with_confidence'] == 0
if gt_mask.any():
    gt_execution = exec_summary[gt_mask].index[0]
```

On this catalog (catalog 27) there are **two** ground-truth executions
of `Image_Classification` (Curator [tk-001]):

| Execution | Rows | Status | Notes |
|---|---|---|---|
| `854` | 500 | Uploaded | First failed `load-cifar10 --num-images 500` attempt; loader-images phase succeeded, datasets phase failed |
| `HSR` | 1100 | Uploaded | Successful retry at `--num-images 1100`; the canonical full GT execution |

The heuristic picked `854` (the partial 500-row attempt). Consequence:
the notebook joined the three prediction CSVs (W96 / XEE / YKP, each
covering all 550 M1G test images) against the 500-row GT subset and
discarded the 300 images that don't have a row in `854`. **Final
analysed sample size: 250/550 per model, not 550/550.**

Reported numbers:

| Run | Reported acc (n=250) | Modeler emission-time acc (n=550) |
|---|---|---|
| W76 (cifar10_quick)   | 25.20% | 24.00% |
| XCE (default_model)   | 39.20% | 37.82% |
| YHP (cifar10_large)   | 44.40% | 41.09% |

The numbers are similar (within ~3 points) but they're **measured on
different test sets**. A model evaluated on a 250-image subset has
~1.6× the per-class sample noise of the full 550-image evaluation, and
the per-class AUC scores in the notebook are computed on the same
truncated subset.

The notebook's heading on cell 11 reads:

> - **cifar10_quick**: 250/550 matched, accuracy 25.2%

So the discrepancy *is surfaced* — a careful reader sees `250/550
matched` and knows half the data was discarded. But the failure mode
is silent in the sense that the notebook proceeds, the ROC curves are
plotted on the truncated set, and the catalog stores a confidently
labelled analysis JPEG that under-uses the test set by 55%.

## Reproduction

```
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \
    assets=toronto_predictions
```

Inspect cell 9 output:

> **Ground Truth:**
> - Execution: [854](https://localhost/id/27/854@355-RZK1-XCXG)
> - Total labels: 500

vs the intended ground truth (per Curator tk-001):

> Execution `HSR`, 1100 rows, covers all 1100 Image rows on this
> catalog including the full 550-image M1G test partition.

## Notes

The Curator's `tacit-knowledge.md` [tk-001] explicitly documents this
trap and prescribes the durable fix: filter by `Execution == "HSR"`,
not by `Confidence IS NULL`. The notebook predates the discovery and
uses the `Confidence IS NULL` filter that mostly works in single-load
catalogs but breaks under loader-retry double-tagging.

Two fix paths:

1. **Notebook-side:** when multiple GT-candidate executions exist
   (`with_confidence == 0`), pick the one with the **most rows** rather
   than the first by index order. On this catalog that would have
   selected HSR (1100 > 500). One-line fix:

   ```python
   gt_execution = exec_summary[gt_mask]['num_images'].idxmax()
   ```

   instead of `exec_summary[gt_mask].index[0]`.

2. **Loader-side:** delete prior `Image_Classification` rows on retry
   (or skip images that already have a class recorded). This is the
   same fix Curator finding 01 recommends for the upstream cause.

Option 1 is the notebook-template-touching fix; option 2 prevents the
problem from existing in the first place. Both belong in a fix-pass,
not in the Analyst arc.

**Workaround the Analyst used:** built a standalone analysis script
(`scripts/build_joined_wide_table.py`) that filters ground truth by
`Execution == "HSR"`, materialises the joined wide table over all 550
M1G test images, and recomputes the analysis numbers on the full set
for the report.

Detected during the 2026-05-28 e2e run; sibling versions per the
Curator findings.
