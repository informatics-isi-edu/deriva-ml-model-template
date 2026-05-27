# Tacit Knowledge

This file records **tacit knowledge** — the *why*, the *intent*, and the
*background* behind decisions made about this project's models and data.

The **catalog** is the source of record for everything else: data contents,
RIDs, dataset versions, workflow URLs and checksums, executions, lineage.
Don't replicate catalog-stored facts here. Don't ask this file what's in
the catalog — query the catalog directly (resources first, tools next).
When this file *needs* to reference a catalog entity, link to it
(`deriva://catalog/{host}/{cat}/ml/...`) instead of inlining its contents.

Each entry captures a decision: what was chosen, what alternatives were
considered, what was rejected and why, and any background context a future
reader would need to evaluate whether the decision still holds.

---

<a id="tk-001"></a>
### tk-001 — Substrate audit: catalog ships with clean, perfectly balanced GT ([dataset JZ8 v0.1.0.post1.dev3](https://localhost/id/2/JZ8@355-KW8K-DXSC))
**When:** 2026-05-27T17:06:00+00:00
**By:** Carl Kesselman (carl@isi.edu)

Audited the freshly-bootstrapped catalog ([catalog 2](https://localhost/chaise/recordset/#2),
`e2e-test-20260527e`) before any modelling work. Question: does the data make sense, and are
there any hidden land mines for the Modeler and Analyst?

The result was a clean substrate. Worth recording because the cleanliness is
*specific to a freshly-loaded catalog* — most of these claims will not survive
the first training execution (see [tk-003](#tk-003) for the Image_Classification
convention).

Findings (audit-time snapshot, not durable beyond the next mutation):

- **Image_Classification ground-truth layer is whole and unambiguous.** 1500
  feature value rows on the [Image table](https://localhost/id/2/Image), exactly
  one row per Image RID, every row written by the loader execution
  [FZC](https://localhost/id/2/FZC). `Confidence` is NULL on every GT row (the
  loader doesn't set it — ground truth has no confidence). 10 classes, each with
  exactly 150 images. No `<unlabeled>` images and no images with multiple GT
  rows.
- **Canonical split is a clean partition.** [JZT v0.1.0.post1.dev2](https://localhost/id/2/JZT@355-KW8K-DXSC)
  (Training, 750) ∪ [K04 v0.1.0.post1.dev2](https://localhost/id/2/K04@355-KW8K-DXSC)
  (Testing, 750) equals [JZ8](https://localhost/id/2/JZ8@355-KW8K-DXSC) (Complete,
  1500), with zero overlap. Each partition is internally class-balanced (75 per
  class on each side). The loader populated these from the canonical Toronto
  CIFAR-10 distribution so this is the source-faithful split.
- **All derived splits are well-formed.** TX8/TXJ and WDA/WDM (the labeled
  stratified families) are disjoint and entirely subset of JZT; K0W/K16 (the
  Small Toronto family) are disjoint and respect the JZT/K04 partition. All
  derived splits are per-class balanced.

Implications for collaborators: the Modeler can train against any of the labeled
training partitions without auditing class balance themselves; the Analyst can
treat ground-truth as the loader-execution rows (filter `Confidence IS NULL` or
by execution RID [FZC](https://localhost/id/2/FZC)) and get a clean 1500-image
GT layer. Class imbalance is not a concern at this scale of catalog. The
*pattern* that makes these claims durable — the loader vs. prediction
write-side of the Image_Classification feature — lives in [tk-003](#tk-003);
the raw counts above will silently rot the first time a model writes
predictions, so reach for [tk-003](#tk-003) when reading this feature in any
post-training session.

---

<a id="tk-002"></a>
### tk-002 — Two small-variant families, two different test-partition sources ([dataset TX0 v0.1.0.post1.dev1](https://localhost/id/2/TX0@355-KW8K-DXSC))
**When:** 2026-05-27T17:06:30+00:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (substrate audit established the partition shape this entry distinguishes)

The catalog ships with two parallel small-variant Split families, and they are
**not interchangeable** for evaluation purposes. The distinction is which side
of the canonical Toronto split the test partition was sampled from. The
template's [CIFAR10.md](CIFAR10.md) §"Dataset Types" notes this distinction;
this entry pins it to the actual RIDs in *this* catalog and spells out the
modelling consequence.

**Family A — sampled from real Toronto train/test** ([dataset K0M v0.1.0.post1.dev1](https://localhost/id/2/K0M@355-KW8K-DXSC),
Split parent):
- K0W (Training, 500): K0W ⊆ JZT — drawn from the official 750 training images.
- K16 (Testing, 500): K16 ⊆ K04 — drawn from the official 750 testing images.
- Test partition is **genuinely held out** from training images.

**Family B — stratified 80/20 split *of the training side only*** ([dataset WD2 v0.1.0.post1.dev1](https://localhost/id/2/WD2@355-KW8K-DXSC)
and [dataset TX0 v0.1.0.post1.dev1](https://localhost/id/2/TX0@355-KW8K-DXSC),
both Split parents):
- WDA (Training, 400) + WDM (Testing, 100): both ⊆ JZT.
- TX8 (Training, 600) + TXJ (Testing, 150): both ⊆ JZT.
- WDA ∩ WDM = ∅ and TX8 ∩ TXJ = ∅, so each family is internally valid as a
  hold-out for *its own* training partition — but the "test" partitions
  WDM and TXJ are **drawn from the same image pool as the Toronto training
  set**. They do **not** represent the canonical CIFAR-10 testing partition
  (K04).

Implications for collaborators:

- **For pipeline-validation runs and fast iteration** (the modelling-pipeline
  question the Modeler may want to answer): either family is fine. The point
  there is "does the training loop converge against a coherent
  training/testing pair," and both families give a coherent pair.
- **For results that should generalize to held-out data the model has never
  seen, including ROC analysis against ground truth not in the training
  pool**: use Family A (`cifar10_small_split` = K0M) or, at full scale, the
  canonical [JZJ](https://localhost/id/2/JZJ@355-KW8K-DXSC) split (Training=JZT,
  Testing=K04). The default-pinned `default_dataset` and `cifar10_quick`
  experiment both currently point at Family B
  ([WD2](https://localhost/id/2/WD2@355-KW8K-DXSC)) — that's a reasonable
  smoke-test default but not a held-out evaluation default. The Analyst doing
  ROC work should pick consciously and the Modeler should produce predictions
  against at least one Family-A or canonical-split dataset if Analyst-grade
  evaluation is in scope.

Weighed alternatives:

- **Curate a new "really held-out" labeled small split** (e.g., a stratified
  subset of K04 paired with stratified K0W). Considered, declined for this
  arc: K0M (Family A) already covers this need at 500/500 — both partitions
  are class-balanced and disjoint from each other. Adding a third labeled
  small family would dilute the catalog without solving a real problem. If
  the Modeler decides they need a tighter analogue of WDA/WDM with K04-side
  test images, that's their call to make from a position of knowing what
  they're modelling.
- **Re-tag WD2/TX0 with a more specific Dataset_Type** (e.g., a new
  `Training_Holdout_Split` term) so consumers dispatching on type can tell
  them apart from real held-out splits. Declined for this arc: the existing
  `Dataset_Type` vocabulary doesn't model "what side of a parent split this
  came from," and adding a term just for this catalog would be an
  over-correction without a downstream consumer asking for it. The
  description on each dataset already names the seed and source partition
  (see `cifar10_labeled_split` description — "stratified 80/20 from training
  images, seed=42"). The Modeler reads descriptions before training; this
  entry makes sure they don't have to read all 13 to find the distinction.

---

<a id="tk-003"></a>
### tk-003 — Convention — Image_Classification is dual-purpose (ground truth + predictions)
**When:** 2026-05-27T17:07:00+00:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (audit established the loader-execution shape this convention scopes)

The [Image_Classification feature on Image](https://localhost/id/2/Execution_Image_Image_Classification)
is written by two distinct kinds of execution and the rows are **not**
distinguishable by table membership alone:

- The **loader execution** ([FZC](https://localhost/id/2/FZC)) wrote 1500
  ground-truth rows with `Confidence IS NULL` (one per Image RID, the canonical
  Toronto label).
- **Training/prediction executions** (none yet at the time of writing this
  entry) will write rows with `Confidence` populated — typically one row per
  (Image, Execution) pair, so the same image will carry multiple label rows
  after each training run.

Implications for collaborators:

- **When reading this feature as ground truth**, filter by execution
  (`Execution == 'FZC'`) or by `Confidence IS NULL`. Either is correct *now*;
  the execution filter is more durable (NULL `Confidence` is a loader-side
  convention that could change with a future loader version).
- An unfiltered `ml.feature_values("Image", "Image_Classification")` returns
  GT + every recorded prediction interleaved. After the Modeler runs even one
  training execution, that's already not what you want for evaluation.
- The `newest` selector is **not** a safe substitute for "ground truth" — it
  resolves to "whichever execution last wrote a row for this image," which
  after training is the most recent prediction, not the GT.
- Dataset-level auditing that counts rows in this feature table will need to
  scope by execution too — a parity check like "1500 rows, no duplicates"
  was true at audit time but goes false after the first training run. The
  durable shape is "1500 rows scoped to execution [FZC](https://localhost/id/2/FZC),
  one per image" — that count survives.

Why this convention exists: the catalog reuses one Feature_Name across the
GT and prediction roles instead of carving out a separate `Image_GT` feature.
That makes provenance cleaner (every label row, whether GT or prediction,
traces to a producing execution) at the cost of needing this filter discipline
on the read side. The right place to filter is in the consumer code, not by
splitting the feature.

---

<a id="tk-004"></a>
### tk-004 — Modeler arc: ROC-ready triplet on Family A K16 ([Execution XZP](https://localhost/id/2/XZP), [Z1R](https://localhost/id/2/Z1R), [103T](https://localhost/id/2/103T))
**When:** 2026-05-27T17:15:00+00:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (clean substrate), [tk-002](#tk-002) (Family A vs B distinction — drove the dataset choice), [tk-003](#tk-003) (Image_Classification dual-purpose — drove the filter discipline in the predictions written here)

Stress-tested the modelling pipeline against the freshly-curated catalog and
produced a coherent triplet of training runs the Analyst can compare on
identical genuinely-held-out test data. The dataset choice was load-bearing:
per [tk-002](#tk-002), only Family A (K0M, with K0W training / K16 testing) has
a test partition drawn from the Toronto official test_batch — i.e. data the
training side has never seen. The default-pinned `cifar10_quick` experiment
runs against Family B WD2, which is a stratified holdout of the training
images themselves; useful as a smoke test but not the right thing for the
Analyst to ROC against.

The three Family-A runs (all same data, all `seed=42`):

| Execution | Model config | Epochs | Channels | Test acc (epoch_final) | Predictions CSV | Weights |
|-----------|--------------|--------|----------|------------------------|-----------------|---------|
| [XZP](https://localhost/id/2/XZP) | `cifar10_quick` | 3 | 32→64 | 25.20% | [Y1M](https://localhost/id/2/Y1M) | [Y1G](https://localhost/id/2/Y1G) |
| [Z1R](https://localhost/id/2/Z1R) | `default_model` | 10 | 32→64 | 36.00% | [Z3P](https://localhost/id/2/Z3P) | [Z3J](https://localhost/id/2/Z3J) |
| [103T](https://localhost/id/2/103T) | `cifar10_large` | 20 | 64→128 | 36.80% | [105R](https://localhost/id/2/105R) | [105M](https://localhost/id/2/105M) |

A fourth smoke run on Family B WD2 ([XDP](https://localhost/id/2/XDP),
`+experiment=cifar10_quick`, 100 test images, 24.00% test_acc) lives in the
catalog too — useful only as evidence that the default experiment preset
runs end-to-end. The Analyst should ignore it for ROC unless they explicitly
want a Family-B comparison point.

Things worth knowing for the Analyst:

- **The triplet is wired into [src/configs/assets.py](src/configs/assets.py)**
  as `modeler_familyA_triplet` (all three prediction CSVs) and as
  per-run `modeler_{quick,default,large}_weights`. Use the triplet name
  in the notebook's `assets=...` override; this avoids hand-pasting RIDs.
- **Training-time accuracy ≠ ROC accuracy.** The `epoch_N` accuracy logged
  above is what the model said about K16 *at the moment training stopped*.
  The Analyst's notebook will recompute accuracy from the committed CSV
  joined against GT — that's the durable number. The emission-time note
  printed at run end (`Emission-time accuracy: ...`) gives the expected
  value so the Analyst can spot-check round-trip integrity.
- **103T shows textbook overfitting.** Train accuracy climbs from 16% (epoch 1)
  to 100% (epoch 20) while test accuracy plateaus around 37% and test loss
  *increases* monotonically after epoch 9. This is a feature for the Analyst,
  not a bug: it gives them per-class behaviour where the model is highly
  confident on examples it's memorised, and tells a real story about model
  capacity vs. dataset size. Peak test accuracy across the triplet was
  ~39.8% at 103T epoch 9.
- **All three runs use `seed=42`**. Run-to-run variance is not what this
  triplet exercises — model/training-budget variation is. If the Analyst
  wants variance estimates, that's a follow-up Modeler arc with
  `model_config.seed=` overrides.
- **[tk-003](#tk-003) is now load-bearing.** `Image_Classification` carries
  1500 GT rows (Execution=`FZC`) + 1700 prediction rows across the five
  training executions in the catalog (3×500 Family-A + 2×100 Family-B).
  Filter by execution RID — `feature_values(..., execution_rids=[...])` —
  before joining to ground truth. The `newest` selector points at whichever
  prediction last touched a given image and is *not* a safe GT shortcut.

Weighed alternatives:

- **Run the same triplet on the canonical JZJ split (JZT train / K04 test,
  750/750).** Considered, declined for this arc. JZJ is 1.5× the data of K0M
  and would have produced very similar comparative findings at higher
  CPU cost; the K16 partition is already 500 fully-class-balanced held-out
  test images, which is enough to produce meaningful per-class ROC. If the
  Analyst wants more confidence in absolute accuracy numbers (vs. cross-run
  *comparisons*), a JZJ follow-up is a one-line override
  (`datasets=cifar10_split`).
- **Use a Family-B experiment for at least one of the substantive runs.**
  Considered, declined. The whole point of the differentiated triplet is to
  let the Analyst compare on identical test data. Mixing in a Family-B run
  would dilute the comparison (different test images, different difficulty
  profile) and the Family-B smoke at [XDP](https://localhost/id/2/XDP) already
  exercises the experiment-preset path.
- **Vary `seed` rather than capacity.** Considered, declined for the
  first-pass arc. Variance across seeds answers a *different* question
  (is training stable?) than variance across configs (does capacity help?).
  The Analyst is more likely to ask the latter — they want to see whether
  the pipeline tells a story about model choice. Seed variation is a clean
  follow-up if they ask for it.
- **Run the `quick_vs_extended` multirun preset.** Considered, declined: it
  pins to Family B `cifar10_small_labeled_split` (WD2). For Analyst-grade
  ROC against truly held-out test data, the right move is the same-model-config
  presets composed with `datasets=cifar10_small_split` overrides, which is
  what the triplet does.

One pipeline gotcha worth pinning, since it surfaced during this arc:
**`description=` is not a free-form Hydra override** — embedded spaces fail
the override grammar with `mismatched input ' '`. The auto-composed
`Execution.description` from PR #46 covers the `+experiment=` path
(see XZP description above), but bare `model_config=/datasets=` overrides
default to "Simple model run" (see Z1R, 103T). Not a bug; just worth
knowing if rich descriptions matter for downstream filtering. The
work-around is to add a small experiment preset rather than reach for
`description=`.

---

<a id="tk-005"></a>
### tk-005 — Analyst arc: capacity helps on textured classes, hurts on `airplane`, and the K16 floor is shared ([Execution 11AY](https://localhost/id/2/11AY), [report](docs/reports/2026-05-27-e-analysis.md))
**When:** 2026-05-27T17:25:00+00:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (clean GT 1500 rows), [tk-002](#tk-002) (Family A = genuinely held-out K16), [tk-003](#tk-003) (filter `Execution=FZC AND Confidence IS NULL` for GT), [tk-004](#tk-004) (the triplet under analysis)

Ranked the Modeler's [Family-A triplet](#tk-004) against K16
ground truth and produced a single wide joined table
([`findings/analyst/wide_joined_K16.csv`](findings/analyst/wide_joined_K16.csv),
500 rows × 35 cols: Image_RID, True_Class, and each model's
Predicted_Class plus all 10 per-class probabilities) from which every
number in [`docs/reports/2026-05-27-e-analysis.md`](docs/reports/2026-05-27-e-analysis.md)
can be re-derived. Cross-channel verified the catalog-resident
`roc_metrics.csv` ([Asset 118J](https://localhost/id/2/118J)) against
a standalone pandas derivation
([`findings/analyst/rank_and_join.py`](findings/analyst/rank_and_join.py));
the two agree to all printed digits.

**Ranking (consistent across Top-1 and Micro-AUC):**

| Run | Top-1 | Micro-AUC | Macro-AUC |
|-----|-------|-----------|-----------|
| [103T](https://localhost/id/2/103T) `cifar10_large` | 36.8% | 0.817 | 0.817 |
| [Z1R](https://localhost/id/2/Z1R)  `default_model` | 36.0% | 0.795 | 0.801 |
| [XZP](https://localhost/id/2/XZP)  `cifar10_quick` | 25.2% | 0.722 | 0.732 |

Findings worth pinning for the next analysis (or next Modeler arc on
this catalog):

- **AUC discriminates better than Top-1 between Z1R and 103T.** The
  accuracy gap is 0.8pp; the Micro-AUC gap is 0.022. If a future
  iteration of this work uses Top-1 alone, it will report the
  triplet as "essentially tied at the top," which understates what
  capacity is buying.
- **Capacity helps most on textured classes.** Per-class AUC jumps
  from `quick` → `large` are largest on `automobile` (+0.157),
  `bird` (+0.137), and `horse` (+0.119); smallest on the smooth
  silhouettes (`airplane`, `ship`, `truck`). Interpretation: the
  larger feature pyramid earns its keep on classes with busy local
  texture; on classes that segregate on global silhouette, the
  shallow features already saturate.
- **`airplane` regresses with capacity.** The large model is
  slightly *worse* than the default on airplane AUC (0.816 vs 0.847).
  Read as a capacity-vs-data signal: with 50 airplane training
  images, the larger model has room to overfit background features
  (sky vs water) that don't transfer K0W→K16. Worth a follow-up
  Modeler arc with augmentation if airplane recall matters.
- **41.4% of K16 is missed by all three models simultaneously.**
  46 / 500 (9.2%) get the same correct prediction from all three;
  207 / 500 (41.4%) get a wrong prediction from all three. The
  41% floor is the *shared difficulty* of K16 — the next modelling
  lever is augmentation / more data / different representation, not
  capacity within this architecture.
- **103T overfits on training but still leads on test AUC.** The
  Modeler's note ([tk-004](#tk-004)) about training accuracy climbing
  to 100% while test loss rises after epoch 9 is real. It doesn't
  flip the AUC ranking because AUC measures ranking, not calibration.
  If a downstream consumer needs calibrated probabilities, 103T is
  the *worst* choice of the three; for pure score-based ranking it's
  still the best. Peak test accuracy from training was ~39.8% at
  epoch 9 — early stopping would have been the right operating point.

Weighed alternatives:

- **Re-run the triplet on JZJ (750/750) for absolute numbers.**
  Considered, declined for this arc. K16 at 500 images is enough to
  *rank* models, and the ranking is what the Analyst owes the next
  reader. JZJ is the right move if anyone wants confidence intervals
  on absolute accuracy; one-line override
  (`datasets=cifar10_split`) and a Modeler re-run.
- **Include per-image confidence calibration plots (reliability
  diagrams).** Considered, declined. The triplet has only 500 test
  images; reliability bins would be ~25 images each, too noisy to
  be load-bearing. If the team adds more runs (seed sweep, longer
  training), reliability becomes worth doing.
- **Use the macro-AUC as the primary metric.** Considered, declined.
  Macro and Micro agree on ranking on this dataset (the class
  balance is exact, so they're roughly the same number). Reported
  both, defaulted reasoning to Micro because it's the conventional
  CIFAR comparison metric. If a future dataset is class-imbalanced,
  Macro is the right primary.
- **Run a confusion-matrix-driven follow-up analysis using
  feature-level joins** (e.g., partition K16 by Image_Class subtype
  if such a vocab existed). Considered, not relevant for this
  catalog: `Image_Classification` is the only label feature and
  there's no subclass vocabulary to slice by. If a future catalog
  adds richer per-image metadata, this becomes a viable next
  analysis tier.

One pipeline friction worth pinning (filed as
[`findings/analyst/01-run-notebook-config-derivation-fails-under-papermill.md`](findings/analyst/01-run-notebook-config-derivation-fails-under-papermill.md)):
**`run_notebook()` auto-derives its Hydra config name from the
notebook filename interactively, but fails under
`deriva-ml-run-notebook`** because the latter calls
`pm.execute_notebook(...)` programmatically and never sets
`PAPERMILL_INPUT_PATH` in `os.environ`. The result is a
`ValueError` from `_derive_config_name_from_notebook` in
[base_config.py:539](deriva_ml/execution/base_config.py). Workaround
applied to `notebooks/roc_analysis.ipynb` cell 3: pass
`run_notebook("roc_analysis", ...)` explicitly. The
`CIFAR10.md`-style guidance about the explicit name being needed
"only when the notebook filename and config name diverge" is wrong
for the only headless runner the project ships — for that runner,
the explicit name is the *common* case. Either
`deriva-ml-run-notebook` should set `PAPERMILL_INPUT_PATH` before
invoking papermill, or the derivation function should also check
papermill's parameter-injected `PAPERMILL_INPUT_PATH` global (which
the CLI does set). Out of scope for this arc — routed around by
editing the notebook.

