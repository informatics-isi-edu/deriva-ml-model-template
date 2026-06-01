# Finding: two small numeric slips in the analysis report prose

- **Persona:** Evaluator (cross-persona read)
- **Date:** 2026-06-01
- **Catalog:** localhost / catalog 2 / schema `e2e-test-20260601`
- **Severity:** Low
- **Component:** `docs/reports/2026-06-01-analysis.md` (Analyst deliverable, not platform code)
- **Category:** Polish (report accuracy)

## What I expected

A cold reader trusts the report's prose numbers. Each stated figure should match
what the catalog assets (raw prediction CSVs QN6/QY8/R7A and the metrics CSV RH4)
actually contain.

## What actually happened

Independently recomputing from the raw prediction CSVs against PK6 ground truth
(sklearn, not reading RH4), the leaderboard, collapse signatures, confusion
table, and component calibration means all reproduce **to the digit**. Two
prose statements are slightly off:

1. **§5 calibration table — Regularized gap.** The report shows the Regularized
   run's confidence means as 0.67 (correct) / 0.59 (wrong) and states the gap as
   **+0.08**. The true means are 0.6676 / 0.5939, gap = **0.0738**, which rounds
   to **+0.07**, not +0.08. The component values round correctly; only the
   subtraction is mis-rounded.

2. **§4 — QWA per-class AUC minimum.** The report says the Regularized run has
   "no class below cat's 0.657." But the metrics CSV (RH4) shows QWA's **deer
   AUC = 0.6533**, which is *below* cat's 0.6567. Cat is therefore not the
   per-class minimum — deer is. The broader claim (QWA's per-class AUC is far
   more even than Fast-LR's, which has multiple near-0.5 classes) remains true.

## Repro

```python
import warnings, tempfile; warnings.filterwarnings("ignore")
import pandas as pd
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id="2")
d = tempfile.mkdtemp()
fv = [r.model_dump() for r in ml.feature_values("Image", "Image_Classification")]
gt = {r["Image"]: r["Image_Class"] for r in fv if r.get("Confidence") in (None, "")}
pk6 = {m["RID"] for m in ml.lookup_dataset("PK6").list_dataset_members().get("Image", [])}
df = pd.read_csv(ml.lookup_asset("QY8").download(d))
df = df[df["Image_RID"].isin(pk6)]
df["true"] = [gt[r] for r in df["Image_RID"]]
df["correct"] = df["true"] == df["Predicted_Class"]
c, w = df[df.correct].Confidence.mean(), df[~df.correct].Confidence.mean()
print(round(c, 4), round(w, 4), round(c - w, 4))   # -> 0.6676 0.5939 0.0738  (rounds to +0.07)
# RH4 QWA per-class: cat=0.656667, deer=0.653333  -> deer is the min, not cat
```

## Impact

Cosmetic. The conclusions, ranking, and all figures on the catalog are correct;
these are authoring nits in the human-readable report. A cold reader who trusts
the prose over the catalog figures would carry one slightly-wrong gap value and
one slightly-wrong "minimum class" attribution. No platform defect is implicated.

## Suggested direction (NOT done — evaluator does not edit deliverables)

Two one-line edits to `docs/reports/2026-06-01-analysis.md`: change "+0.08" →
"+0.07" in the §5 table, and reword the §4 "no class below cat's 0.657" sentence
to reflect that deer (0.653) is QWA's actual per-class AUC floor.
