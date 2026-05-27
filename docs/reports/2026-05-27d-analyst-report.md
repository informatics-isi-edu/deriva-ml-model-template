# CIFAR-10 CNN Analyst Report (2026-05-27d)

**Author:** Analyst persona, e2e multipersona run on catalog 96 (`e2e-test-20260527d`)
**Date:** 2026-05-27
**Worktree:** `deriva-ml-model-template-e2e/` (branch `e2e-test/2026-05-27-d`)
**Analysis execution:** `Y90` — [https://localhost/id/96/Y90](https://localhost/id/96/Y90)
**Notebook:** `notebooks/roc_analysis.ipynb`, executed via `deriva-ml-run-notebook`
**Input runs analysed:** XDP (`cifar10_quick`), XPR (`default_model`), XZT (`cifar10_extended`)
**Held-out test set:** `cifar10_small_labeled_testing` (WDM, 100 images, 10/class)

---

## TL;DR for a non-ML collaborator

The Modeler produced three CIFAR-10 models on a small (400-image)
training set. The biggest model with the most training was best on
top-1 accuracy (41%) — but only just (3 percentage points above the
mid-size model at 38%), and at four times the parameter count and
five times the training cost. **The mid-size model (XPR) is the
right operating point.** The big model (XZT) shows classic overfit
symptoms: it is more confident than the mid-size model, but its
confidence is not earned — it's confident at the same rate when it
is wrong. The smallest model (XDP) hasn't learned to classify yet;
it's the smoke test, not a candidate.

All three models share the same blind spot: *frog is a magnet
class*. When they don't know, they reach for frog. Cat is the most
brittle class across all three runs — none of them learned to
recognize it reliably.

These are not publishable numbers (small training set, small test
set, ten classes, classes-as-words-not-pictures). Treat them as a
characterisation of *how the models behave*, not as evidence of
*how well they would perform in production*.

---

## 1. Inputs and how they were chosen

Per `tacit-knowledge.md` tk-003, the Modeler ran three training
executions, all sharing the same workflow (XDG) and the same
dataset (WD2, `cifar10_small_labeled_split`, 400 train / 100 test).
Holding the data fixed and varying only `model_config` is the
cleanest controlled experiment the catalog supports, and it lets
the same ROC/confusion-matrix surface compare the three models
without any per-image confounds.

| Exec | Asset (preds) | model_config       | Epochs | Arch              | Reg                 | Weights |
|------|---------------|--------------------|-------:|-------------------|---------------------|--------:|
| XDP  | XFM           | `cifar10_quick`    | 3      | 32→64ch, 128h     | none                | 6.5 MB  |
| XPR  | XRP           | `default_model`    | 10     | 32→64ch, 128h     | none                | 6.5 MB  |
| XZT  | Y1R           | `cifar10_extended` | 50     | 64→128ch, 256h    | dropout 0.25, wd 1e-4 | 26 MB |

Catalog audit (tk-001) confirmed every Image in WD2 carries a
ground-truth `Image_Classification` label, and class balance is
exactly 10/class on the held-out side. So we can read accuracy
straight from the prediction-vs-GT join without weighting or
filtering.

## 2. Independent verification of the Modeler's numbers

Before trusting any further analysis, I re-derived top-1 accuracy
from each prediction CSV joined to the GT feature, via the
`deriva-ml` Python API (`analysis-scratch/rank_runs.py`). All three
runs matched the emission-time numbers in tk-003 exactly:

| Exec  | Modeler reported | Re-derived | Match |
|-------|------------------|------------|-------|
| XDP   | 24.00%           | 24.00%     | yes   |
| XPR   | 38.00%           | 38.00%     | yes   |
| XZT   | 41.00%           | 41.00%     | yes   |

Prediction count = 100 in all three runs, 100/100 matched to
ground truth. No missing or extra labels.

## 3. Ranking and ROC analysis

Ranked by top-1 accuracy, then by micro-AUC. The same ordering
holds either way:

| Rank | Exec | Accuracy | Micro-AUC | Macro-AUC |
|------|------|---------:|----------:|----------:|
| 1    | XZT  | 41.00%   | 0.7979    | 0.8046    |
| 2    | XPR  | 38.00%   | 0.7874    | 0.7959    |
| 3    | XDP  | 24.00%   | 0.6773    | 0.7124    |

The notebook produced ROC curves for each run, a micro-AUC overlay,
and confusion matrices. All eight JPEGs + `roc_metrics.csv` were
committed as output assets on Y90 — see §6 for the asset list.

**Why the AUC ordering matters.** Accuracy alone treats every
mis-classification the same and only looks at the top-1 prediction.
AUC asks: if I had to score *every* image's *every* class as
"probably this class / probably not this class", how often is the
model's score for the true class higher than its score for some
random wrong class? It uses the entire probability distribution,
not just the argmax. The fact that XZT only edges out XPR by 1
point of micro-AUC (0.798 vs 0.787) — despite XZT having 4× the
parameters and 5× the training — tells you the *probability
landscape* the two models learned is broadly similar. The size and
training cost bought 3pp of top-1 accuracy and almost nothing in
AUC. That is not a good return.

## 4. Confusion-matrix narrative

This is where the three models actually differ — top-1 obscures
the shape of the errors.

**XDP (3 epochs, 24% acc) — "model has barely started learning".**
It outputs `frog` for 39% of all test images and `truck` for 19%.
Seven of ten classes have **zero** recall (0/10 correct). The two
classes with high recall (`airplane` 70%, `frog` 80%, `truck` 80%)
are not signs of learning — they're signs of bias toward those
output classes during the chaotic first few epochs. This is not a
candidate model; it's the smoke-test that confirms the pipeline
trains *something*.

**XPR (10 epochs, 38% acc) — "broadly competent".** Recall is
non-zero for 9 of 10 classes (only `deer` is at 0%). Per-class
accuracy ranges from 20% (`bird`) to 70% (`automobile`). The most
common confusions are domain-sensible:
- `truck → automobile` (5 of 10) — both are wheeled vehicles seen
  from the side. A model with low resolution and few epochs will
  reasonably confuse them.
- `dog → cat` (4 of 10) — both are small furry quadrupeds.
- `deer → bird` (5 of 10) and `cat → bird` (5 of 10) — `bird` is
  acting as a low-confidence "small natural thing" attractor, the
  same role `frog` plays in XDP and XZT.

This is what "the model is starting to learn the structure of the
classes" looks like.

**XZT (50 epochs, 41% acc) — "learned more, but lost some
classes".** It's better than XPR on `frog` (60% vs 40%), `horse`
(60% vs 40%), `ship` (30% vs 40% — actually slightly worse), and
`truck` (60% vs 40%). But it **completely loses `bird`** (0% vs
20%) and *nearly* loses `cat` (10% vs 30%). Net: +3 points of
accuracy. The same `cat → frog` confusion that XDP showed (8/10
cats predicted as frogs) is still present in XZT (5/10) but
absent from XPR (0/10). The "extended" config did not heal the
attractor-class issue; it just made the model more confident
about the *frogs* than the *cats*.

**Common pattern across all three: frog and truck attract.**
- XDP: predicts `frog` for 39%, `truck` for 19% of all 100 images.
- XPR: more uniform; `automobile` is the dominant prediction at 18%.
- XZT: `frog` at 25%, `automobile` at 16%, `truck` at 15%.

CIFAR-10's "frog" images are visually low-contrast and round-bodied;
they sit in the same shape neighbourhood as cats, deer, and birds at
32×32 resolution. The models are picking up on the shape, not on
species-level features.

## 5. Confidence calibration: the most important finding

This is what makes the rank-1 model (XZT) genuinely concerning.

Average softmax max-probability ("how confident was the model in
its top guess?"), split by whether the guess was right:

| Exec | Confidence when correct | Confidence when wrong | Gap |
|------|-------------------------|-----------------------|-----|
| XDP  | 23.5%                   | 17.8%                 | 5.7pp |
| XPR  | 50.8%                   | 41.8%                 | 9.0pp |
| XZT  | **88.6%**               | **77.8%**             | **10.8pp** |

XZT is *aggressively confident* on every image — 78% confident
even when wrong. A non-ML reader should think of this as: imagine
a colleague who answers every question with "I'm 78% sure" — but
40% of the time, they're wrong. That's a worse signal than a
colleague who says "I'm 42% sure" but is right 38% of the time.

The XPR gap (50.8% vs 41.8%, a 9-point spread between
correct-confidence and wrong-confidence) is the best
calibration of the three. **XPR is the model whose probability
outputs are most informative.** XZT's outputs are saturated by the
50-epoch training on 400 images — it has memorised the training
set and projects that memorisation onto every test image with high
confidence regardless of whether it actually recognises it.

This is the *exact* overfit signature tk-003 anticipated from
training/test loss divergence; it shows up here as a confidence
gap on the held-out predictions.

## 6. Catalog state (analyst-produced artifacts)

All artifacts are linked to Execution `Y90` on catalog 96. Verified
cross-channel: `analysis-scratch/verify_y90.py` lists 13 linked
`Execution_Asset` rows (3 inputs + 10 outputs).

**Inputs (downloaded by the runner):**
- XFM (XDP's predictions) — 15 KB
- XRP (XPR's predictions) — 16 KB
- Y1R (XZT's predictions) — 17 KB

**Outputs (produced by the notebook, all `Uploaded`):**

| Asset | Filename                                         | Bytes   |
|-------|--------------------------------------------------|--------:|
| YB4   | `roc_curves_cifar10_quick_XFM.jpg`               | 144,416 |
| YB6   | `roc_curves_default_model_XRP.jpg`               | 134,622 |
| YB8   | `roc_curves_cifar10_extended_Y1R.jpg`            | 133,790 |
| YBA   | `roc_comparison_Y90.jpg`                         | 106,610 |
| YBC   | `confusion_matrix_cifar10_quick_XFM.jpg`         | 145,662 |
| YBE   | `confusion_matrix_default_model_XRP.jpg`         | 147,553 |
| YBG   | `confusion_matrix_cifar10_extended_Y1R.jpg`      | 148,225 |
| YBJ   | `roc_metrics.csv`                                |     571 |
| YCT   | `roc_analysis.ipynb` (papermill-executed)        | 960,497 |
| YCW   | `roc_analysis.md` (markdown export)              | 910,303 |

A local copy of all 10 output assets is in
`analysis-scratch/y90_outputs/`, downloaded via
`fetch_outputs.py`, so the Evaluator can review the artifacts
without having to re-download from the catalog.

## 7. Recommendations for downstream readers

For the use case represented by this catalog (a small CIFAR-10
demo, not a production system), the analytical answer is:

1. **Pick XPR** (`default_model`, 10 epochs) for any further
   downstream work. It produces the best-calibrated probability
   outputs of the three runs — and on this dataset, calibration
   is more useful than the 3 extra points of top-1 accuracy XZT
   buys.
2. **Don't pick XZT for "best model".** Its 41% top-1 accuracy is
   real but bought with a model 4× larger and saturated-confidence
   outputs. On a held-out distribution shift it would be very
   confidently wrong.
3. **Don't pick XDP at all.** It's the smoke test.

For a real version of this exercise (not a 100-image test set, not
a 400-image train set), you would want:
- Larger held-out test partition. 10/class doesn't pin down per-class
  recall to a stable interval — moving one image changes a class
  recall by 10 absolute points.
- Train on the full `cifar10_labeled_training` (TX8, 600 images) or
  the full `cifar10_training` (JZT, 750) rather than the
  small-labeled (WDA, 400). The Modeler's choice of WD2 was correct
  for *this* exercise (controlled comparison) but limits the
  conclusions about absolute performance.
- A calibration plot (reliability diagram) per model, not just the
  correct/wrong mean confidence numbers above. The notebook does not
  produce these today; that's a candidate extension.

## 8. Process observations (for the Evaluator)

Things that went well:
- The asset-RIDs-in-config pattern that the Modeler used
  (`roc_quick_vs_extended` in `src/configs/assets.py`) plugged
  straight into the notebook without any further wiring. One
  `deriva-ml-run-notebook notebooks/roc_analysis.ipynb` invocation
  produced the entire analysis with provenance.
- Cross-channel verification (`PathBuilder` query of
  `Execution_Asset_Execution` filtered to Execution=Y90) confirmed
  all 13 assets landed in the catalog with the expected MIME types
  and byte counts. No drift between what the notebook said it saved
  and what the catalog ended up holding.
- The notebook's per-asset-RID suffix on output filenames
  (`roc_curves_<model_config>_<asset_rid>.jpg`) protected the
  outputs from collision even though XPR and XDP share a Hydra
  `model_config` shape. (A pure-`model_config`-naming scheme would
  have overwritten one of them.)

Things that took a beat to figure out:
- I initially tried `uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb roc_analysis`
  expecting the trailing arg to be a config selector; that's not
  how it works (`roc_analysis` is the default config the notebook
  asks for internally via `run_notebook("roc_analysis", ...)`). The
  hydra parser raised `missing EQUAL at '<EOF>'` which is
  technically correct but doesn't point at the actual problem (an
  unexpected positional). Not blocking; not filing a finding for
  it. CLAUDE.md and the notebook docstring already document this
  pattern.

Nothing I want to flag as a `findings/analyst/` entry — the
analysis ran cleanly end to end, the catalog matches what the
notebook claimed it did, and the friction I encountered is the
documented kind.

---

## Pointers for the Evaluator

- `tacit-knowledge.md` entries: tk-001 (Curator audit), tk-002
  (leakage map), tk-003 (Modeler runs), tk-004 (this analysis —
  added in the same commit as this report).
- Independent ranking script: `analysis-scratch/rank_runs.py`
  (re-runnable; outputs `analysis-scratch/rankings.json` with the
  full per-class breakdown that informed §4).
- Cross-channel verifier: `analysis-scratch/verify_y90.py`.
- Local copies of the 10 output assets: `analysis-scratch/y90_outputs/`.
- The catalog itself: `https://localhost/id/96/Y90` (the analysis
  execution; all linked assets visible via Chaise).
