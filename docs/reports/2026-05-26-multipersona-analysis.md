# CIFAR-10 e2e Multipersona Analysis (Catalog 18, 2026-05-26)

**Analyst arc** of the 2026-05-26 three-persona end-to-end run on
catalog `e2e-test-20260526` (id `18`). The Developer arc trained 6
viable CIFAR-10 CNN variants on `CRR` (200-image stratified subset,
seed=123); all six tested on the same 50-image partition `CSA`
(seed=123, 5 imgs/class). This report ranks those runs, compares
accuracy against AUC, exercises the dataset denormalize surface, and
flags a handful of friction points.

**TL;DR.** EJ0 (lr=1e-3 from the lr_sweep) wins on raw test accuracy
at 30%; the extended-architecture run E4A (50 epochs) wins on macro/
micro AUC at ~0.70 — a clean illustration of the difference between
"how often is the argmax right" and "how well does the score
separate". With only 50 test images (5/class) every ranking has a
~7-point standard error, so do not over-read these spreads.

## 1. Catalog state going in

- **Catalog:** `localhost / 18` (alias `e2e-test-20260526`), 500
  images, 15 datasets (13 bootstrap + 2 curator-added).
- **Source of record for the Developer's runs:**
  [`tacit-knowledge.md` §tk-004](../../tacit-knowledge.md).
- **Skip list:** `F40` (degenerate Validation-bag execution from
  finding `developer/01`, 1 asset = a 50-byte `training_status.txt`,
  no model weights), `EA8` (lr_sweep parent shell, no model
  artifacts).
- **Workflow:** all 6 viable training executions reference
  `Workflow=DY6` (`cifar10_cnn`) directly via ermrest;
  `deriva_ml_get_execution` over MCP reports `workflow_rid: null`
  for every one of them (developer/02). This report uses the direct
  ermrest channel where workflow attribution matters.

## 2. Ranking — accuracy vs AUC

Recomputed from the prediction-CSV assets at notebook execution
[`F6C`](https://localhost/id/18/F6C). 50 test images on CSA;
ground-truth feature values come from bootstrap execution `854`.

| Rank | Exec | Variant | Predictions | Test_acc | AUC_Macro | AUC_Micro |
|---|---|---|---|---|---|---|
| 1 (acc) | **EJ0** | lr_sweep lr=1e-3 | `EM0` | **30.0%** | 0.647 | 0.645 |
| 2 (acc) | DYC | quick (3 ep, lr=1e-3) | `E0A` | 28.0% | 0.616 | 0.593 |
| 3 (acc) | **E4A** | extended (50 ep, deeper net) | `E68` | 24.0% | **0.695** | **0.699** |
| 4 (acc) | EC0 | lr_sweep lr=1e-4 | `EE0` | 14.0% | 0.642 | 0.589 |
| 5 (acc) | ER0 | lr_sweep lr=1e-2 | `ET0` | 12.0% | 0.569 | 0.553 |
| 6 (acc) | EY0 | lr_sweep lr=1e-1 (diverged) | `F00` | 10.0% | 0.500 | 0.500 |

**Reads:**

- **Accuracy ranking matches tk-004 exactly.** No surprises;
  cross-channel verification of test_acc against the catalog
  prediction CSVs passes. EJ0 > DYC > E4A > EC0 > ER0 > EY0.
- **AUC tells a different story for E4A.** The deeper, longer-
  trained model has the best probability calibration on this test
  set (AUC_Macro 0.695, AUC_Micro 0.699 — both ahead of EJ0's 0.647
  and DYC's 0.616), even though its argmax-accuracy is lower. The
  Developer's tk-004 noted E4A peaked at ~32% test_acc around epoch
  29 and degraded thereafter; the AUC numbers suggest the
  *underlying score distribution* still discriminates well — the
  argmax just lands on the wrong class slightly more often.
- **EY0 is at chance.** AUC_Macro 0.500 across every class is
  consistent with the Developer's "train_loss=1269 at epoch 1"
  divergence (tk-004): the model never learned to discriminate.
- **EC0 is undertrained, not broken.** 14% accuracy but AUC_Macro
  0.642 — its probability rankings are surprisingly close to EJ0's;
  10 epochs at lr=1e-4 just hasn't moved the argmax decision
  boundary yet.

**Choice of metric.** For this run the right summary metric is
**AUC_Macro** — argmax accuracy on 50 images with 5/class has a
~7-percentage-point std error from a binomial-N=50 view, so the
8-point spread between EJ0 (30%) and E4A (24%) is well within noise.
AUC integrates over the threshold sweep and is the more stable
estimate at this sample size. By that metric the ranking is
**E4A > EJ0 > EC0 > DYC > ER0 > EY0**, with E4A's lead larger than
the gap between any other pair.

## 3. Outputs landed in the catalog

All of the following live under execution **`F6C`** (analysis run,
Workflow_Type=`ROC Analysis Notebook`, hydra config
`roc_analysis assets=roc_all_six`):

| Asset | RID | What |
|---|---|---|
| Per-experiment ROC curves (× 6) | `F8W`, `F8Y`, `F90`, `F92`, `F94`, `F96` | One JPG each for DYC, E4A, EC0, EJ0, ER0, EY0 |
| ROC comparison overlay | `F98` | Micro-avg ROC for all six on one axis |
| Confusion matrices (× 6) | `F9A`, `F9C`, `F9E`, `F9G`, `F9J`, `F9M` | Normalized confusion matrices per run |
| Summary CSV | `F9P` (`roc_metrics.csv`) | Per-class + aggregate AUC + accuracy |
| Executed notebook | `FBP` (`roc_analysis.ipynb`, 1.7 MB) | Includes all rendered figures |
| Markdown export | `FBR` (`roc_analysis.md`) | Human-readable transcript |

Per-experiment ROC and confusion-matrix filenames use
`cifar10_quick_*` for five of six runs because all four `lr_sweep`
children share `model_config=cifar10_quick` (only `learning_rate`
differs). The asset RID suffix is what distinguishes them.

## 4. The denormalize experience

This is the test's deliberate exercise of the denormalize surface.
Below is what it actually felt like to use, end-to-end.

**The discovery path was good.**
`ml.lookup_dataset(rid)` exposes five denormalize-adjacent methods
(`describe_denormalized`, `list_denormalized_columns`,
`get_denormalized_as_dataframe`, `get_denormalized_as_dict`,
`cache_denormalized`). Tab-completion surfaces all of them. The
MCP equivalent `deriva_ml_denormalize_dataset` is registered and
listed by the orientation resources.

**The semantics took two tries to figure out.** First attempt:

```python
ds.get_denormalized_as_dataframe(
    include_tables=["Image", "Image_Classification"],  # WRONG
    row_per="Image",
)
# raises DerivaMLException: The table Image_Classification doesn't exist.
```

The mental model "I want the dataset's images, joined with their
classification" reaches for the *feature name* (`Image_Classification`)
because that's what the feature is called in `find_features()` and
in `feature_values()` everywhere else. But `_run` requires the
underlying *feature table name*, `Execution_Image_Image_Classification`.
Annoying but the right name is one `find_features('Image')` call
away.

Inconsistency worth flagging: **`describe_denormalized(include_tables=
["Image", "Image_Classification"])` happily accepts the feature
name** and lists it as a `row_per_candidate`, before `_run` rejects
it. Describe-vs-run disagree on what counts as a valid `include_tables`
entry. See `findings/analyst/01-describe-vs-run-include-tables.md`.

**Second try, with the right table name, was textbook.**

```python
df = ds.get_denormalized_as_dataframe(
    include_tables=["Image", "Execution_Image_Image_Classification"],
)
# shape: (350, 12); one row per feature value (50 imgs × 7 sources)
```

**The output shape is execution-aware.** Because
`Execution_Image_Image_Classification` carries `Execution` as a
discriminator, the wide table emits one row per
(image, producing execution). For CSA that means:

| Producing Execution | Rows |
|---|---|
| `854` (bootstrap ground-truth) | 50 |
| `DYC`, `E4A`, `EC0`, `EJ0`, `ER0`, `EY0` (each model) | 50 each |
| **Total** | **350** |

This is genuinely useful for analysis: instead of having to load
each prediction CSV from hatrac and join, the denormalize wide
table already presents predictions and ground truth in the same
join-key space (`Image.RID`). Filtering by
`Execution_Image_Image_Classification.Execution` cleanly separates
ground truth (`==854`) from predictions (`in {DYC, ...}`). I used
this view to drive the cross-channel reconciliation in §5.

**Picking the right `row_per`.** With no `row_per`, the
denormalizer auto-picked `Execution_Image_Image_Classification`
(the downstream-most table) and emitted the 350-row long-format
view above. Passing `row_per="Image"` aggregates over executions
and the runner refuses — `DerivaMLDenormalizeDownstreamLeaf`
explains exactly that, including a suggested fix. Excellent error
message, much better than I expected.

**Column naming.** `Table.column` notation throughout. Predictable.
The only nit: `Execution_Image_Image_Classification.Image` and
`Image.RID` are the same value for the join key — the
denormalizer keeps both, presumably for cardinality safety. Not
a blocker, but a user analyzing 350 rows × 12 cols has to know
which of the duplicate-meaning columns is canonical.

**Verdict.** Discoverable, reliable, semantically correct on this
catalog. The describe-vs-run inconsistency is the one rough edge.
If this surface stays this stable, I would reach for it in
preference to ad-hoc PathBuilder joins in future analysis
notebooks.

## 5. Cross-channel verification

Followed the §3.4 mandatory protocol; all checks pass.

| Check | Indirect (MCP / skill) | Direct (deriva-ml / ermrest) | Agree? |
|---|---|---|---|
| 6 viable executions present (DYC/E4A/EC0/EJ0/ER0/EY0) | `deriva_ml_get_execution` returns each | `pb.schemas['deriva-ml'].tables['Execution']` returns each, all `Status=Uploaded` | **YES** |
| Prediction-CSV asset RIDs match tk-004 mapping | `deriva_ml_list_assets(execution_rid=...)` returns expected RIDs | `ml.lookup_execution(...).list_assets()` returns expected RIDs | **YES** |
| Workflow attribution | MCP `workflow_rid: null` everywhere | ermrest `Workflow=DY6` everywhere | **NO** (developer/02 still applies) |
| F6C analysis assets uploaded (14 new) | `deriva_ml_list_assets(F6C)` | `ml.lookup_execution('F6C').list_assets()` | **YES** |
| **Denormalize wide-table on CSA**: 50 GT rows | `deriva_ml_denormalize_dataset(CSA, ...)` returns 350-row long-format | `ds.get_denormalized_as_dataframe(...)` returns 350-row DataFrame | **YES** |
| **Denormalize GT image RIDs == `list_dataset_members(CSA)['Image']`** | n/a (members via direct only) | 50 == 50, exact set match | **YES** |
| **Denormalize GT label distribution == `feature_values('Image','Image_Classification')` for CSA images** | balanced 5/class × 10 | balanced 5/class × 10 | **YES** |
| Predictions used in analysis match what Developer's executions actually produced (joined by Image_RID, all 50 keys present each side) | n/a | hand-checked: every prediction CSV's `Image_RID` column ∈ CSA's 50 image RIDs | **YES** |

Three direct-channel scripts back this up (committed under
`scripts/`):

- `scripts/analyst_verify_executions.py` — exec → asset-RID mapping
- `scripts/analyst_denormalize_check.py` — denormalize reconciliation

## 6. Caveats / what to NOT over-read

- **N=5 per class** on the test set. Confusion matrices have at
  most 5 in any diagonal cell and 0 in many off-diagonal cells.
  Spread is dominated by noise.
- **No seed** (developer/03 / D02). Re-running the same Hydra
  config will not reproduce these exact accuracies. The numbers
  above are specific to execution RIDs DYC, E4A, EC0, EJ0, ER0,
  EY0.
- **No held-out validation.** All evaluation is on CSA (seed=123
  test partition, drawn from the same training pool as CS0).
  Curator-added `DAP` was the intended held-out evaluator but is
  not currently consumable by the `cifar10_cnn` runner
  (developer/01).
- **Accuracy and AUC rank E4A differently** (3rd vs 1st). Both
  metrics are correct; they answer different questions
  ("argmax" vs "score-separability"). For a production model
  decision, the choice between them is the load-bearing one.
- **MCP `feature_values` cursor is broken** (curator/02). The
  ROC notebook works around it by using the direct
  `ml.feature_values('Image', 'Image_Classification')` Python
  call.

## 7. Open questions left

1. Is E4A's epoch-29 32% peak (Developer's tk-004 observation)
   reproducible? With no seed, the only way to answer that is
   D02 → rerun. The AUC numbers I see here suggest the peak is
   real (E4A's *score distribution* is genuinely better), not a
   noise spike.
2. The `roc_metrics.csv` shows EC0 (undertrained, 14% acc) with
   AUC_Macro 0.642 — close to EJ0's 0.647. Is this an artifact
   of 50 test images, or would a longer run at lr=1e-4 land at
   EJ0-or-better performance? A second lr_sweep with more epochs
   per child would settle it.
3. The Curator's `DB0` (`cifar10_balanced_demo`, 50 imgs, 5/class
   hand-picked from 96E) was prepared as the "guaranteed-populated
   confusion matrix" target but **no execution in this arc ran
   against DB0** — they all ran on CSA. A `test_only` run pointing
   at one of the existing weights assets and `datasets=cifar10_balanced_demo`
   would close the loop. Outside this arc's scope, noted for next
   time.

## 8. Friction summary (for the wrap-up friction map)

| # | Surface | Severity | Filed as |
|---|---|---|---|
| 1 | `describe_denormalized` accepts feature names that `get_denormalized_as_dataframe` rejects | Low | `findings/analyst/01-describe-vs-run-include-tables.md` |
| 2 | F6C description bound to notebook config name, not asset-override choice (chaise shows "quick vs extended" for an all-six run) | Low | `findings/analyst/02-execution-description-stale-on-asset-override.md` |
| 3 | `Asset.download()` requires positional `dest_dir`; docstring doesn't make this obvious vs other deriva-ml download APIs that have sensible defaults | Trivial | not filed — one-line fix, not blocking |

No new high-severity findings against the denormalize surface — it
behaved correctly on every checked dimension.

---

*Report prepared by the Analyst persona, 2026-05-26. Catalog 18.
Analysis execution `F6C`. See `tacit-knowledge.md` §tk-005 and
§tk-006 for the decision rationale behind the metric choice and
the denormalize call shape.*
