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
### tk-001 — Convention — `Image_Classification` carries both loader-retry rows and (after training) prediction rows ([feature Execution_Image_Image_Classification](https://localhost/id/27/HSR@355-RWN7-R3D8))
**When:** 2026-05-28T10:30:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

`Image_Classification` on `Image` is a single feature definition with
two consumers in this project: the **loader** writes ground-truth
rows (one per image per `load-cifar10` images-phase run), and **training
executions** will write prediction rows. The rows are not
distinguishable by table membership alone — both populate
`Execution`, `Feature_Name="Image_Classification"`, and `Image_Class`.
Ground-truth rows have `Confidence IS NULL`; prediction rows have
`Confidence` populated (`record_test_predictions` in
`src/models/cifar10_cnn.py` is the canonical writer).

Catalog 27 already has **two ground-truth-class executions**, not one:
[execution 854](https://localhost/id/27/854@355-RWN7-R3D8) (500 rows,
the first failed loader attempt at `--num-images 500`) and
[execution HSR](https://localhost/id/27/HSR@355-RWN7-R3D8) (1100 rows,
the successful retry at `--num-images 1100`). They agree on class for
every image in their intersection — see
`findings/curator/01-duplicate-image-classification-feature-rows.md`
for the full audit. This is "retry leftover," not contradictory truth.

Implications for collaborators: when reading this feature as ground
truth, **filter by execution** (`Execution == "HSR"` covers all 1100
images and is the right pick) or by `Confidence IS NULL` (covers both
GT executions before any training prediction has been written). After
the Modeler runs even one training execution, a bare
`ml.feature_values("Image", "Image_Classification")` returns GT +
predictions interleaved; the `newest` selector is *not* a safe
substitute because "newest" is whichever execution last wrote, not
"ground truth." The HSR-filter is the durable answer.

---

<a id="tk-002"></a>
### tk-002 — Training-derived `split_dataset` outputs leak across train/test in this catalog ([dataset TCC v0.1.0.post1.dev1](https://localhost/id/27/TCC@355-RWN7-R3D8))
**When:** 2026-05-28T10:35:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (the double-tagging that powers the leak)

The `cifar10_labeled_split` family
([TCC](https://localhost/id/27/TCC@355-RWN7-R3D8) → TCM + TCY) and
the `cifar10_small_labeled_split` family
([VAP](https://localhost/id/27/VAP@355-RWN7-R3D8) → VAY + VB8) are
**not disjoint splits in this catalog**:

| Split family | Train RID | Test RID | Image overlap |
|---|---|---|---|
| TCC | [TCM](https://localhost/id/27/TCM@355-RWN7-R3D8) (361 images) | [TCY](https://localhost/id/27/TCY@355-RWN7-R3D8) (105 images) | **33 images in both** |
| VAP | [VAY](https://localhost/id/27/VAY@355-RWN7-R3D8) (339 images) | [VB8](https://localhost/id/27/VB8@355-RWN7-R3D8) (95 images) | **24 images in both** |

`_cifar10_datasets.py` calls `split_dataset(...,
row_per="Execution_Image_Image_Classification", ...)` which partitions
feature *rows*, not image RIDs. Combined with the loader-retry
double-tagging recorded in [tk-001](#tk-001) (250 of M16's 550 images
carry two feature rows), an image with two feature rows can land on
both sides of the split — 100% of the observed overlapping images are
exactly the doubly-tagged ones. The advertised split sizes (440/110
and 400/100) also don't match actual member counts in this catalog
because the splitter is sampling from a larger row pool than the
image pool.

Implications for collaborators: held-out evaluation on TCC or VAP
*will* score 30%+ of test images against a model that saw them at
train time. The clean alternative in this catalog is the **Toronto
family** ([M16](https://localhost/id/27/M16@355-RWN7-R3D8) training
× [M1G](https://localhost/id/27/M1G@355-RWN7-R3D8) testing): zero
overlap by construction (different Toronto source batches), 55/class
on each side, ground truth on both halves. The default_dataset config
still points at VAP (small + labeled) — fine for "smoke-test the
pipeline runs" purposes; **not fine for accuracy claims**. See
`findings/curator/02-train-test-leakage-in-labeled-split-datasets.md`
for the audit. Filing the finding rather than rebuilding a clean
split in the worktree was the e2e-fitness-run call; a fix-pass on
`_cifar10_datasets.py` (use `row_per="Image"` and dedupe upstream) is
the durable answer.

---

<a id="tk-003"></a>
### tk-003 — Convention — class balance is preserved per-dataset, including on the split-by-row partitions
**When:** 2026-05-28T10:40:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-002](#tk-002) (counts the leakage; this entry records the orthogonal class-balance property)

Class balance is uniform across every dataset partition in this catalog,
regardless of which split family produced it. Toronto families
(M16/M1G, M28/M2J) are exactly N/class by construction; the
training-derived split families (TCM/TCY, VAY/VB8) are *approximately*
balanced (32–38/class on TCM at 361 total; 9–11/class on TCY at 105
total) because `split_dataset` is stratifying. **Class imbalance is
not a confound in this catalog** for any of the 13 datasets — the
problem space is row-partition mechanics ([tk-002](#tk-002)), not
class-distribution drift.

Implications for collaborators: a model that underperforms here can
be safely diagnosed as "the model isn't learning that class," not
"the training set didn't show it that class." Per-class precision /
recall / confusion-matrix work doesn't need stratification reweighting
adjustments — the priors are uniform.

---

<a id="tk-004"></a>
### tk-004 — Modeler arc chose the Toronto pair (cifar10_toronto_pair) over the leaky labeled-split families ([dataset M16 v0.1.0.post1.dev2](https://localhost/id/27/M16@355-RYPE-KKW8))
**When:** 2026-05-28T14:10:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-002](#tk-002) (named the Toronto pair as the leakage-free pick), [tk-003](#tk-003) (class balance is uniform so no stratified reweighting needed)

Bundled M16 (Toronto training, 550 images) and M1G (Toronto testing,
550 images) into a new `cifar10_toronto_pair` dataset group in
`src/configs/datasets.py` and built three new experiments on top
(`cifar10_toronto_quick`, `cifar10_toronto_default`,
`cifar10_toronto_large` in `src/configs/experiments.py`). The
existing template experiments (`cifar10_quick`, `cifar10_default`,
`cifar10_extended`) all default to `cifar10_small_labeled_split`
(VAP) or `cifar10_labeled_split` (TCC), which [tk-002](#tk-002)
established leak 30%+ of test images back into train. Rather than
mutate the template's defaults (which the Curator pragmatically left
in place for a fresh-clone smoke run), the Modeler added net-new
experiment groups whose dataset choice is explicit in the config name.

Why bundle as one `datasets` group, not two separate ones: the model
harness in `src/models/cifar10_cnn.py` (`build_loaders`) walks
`execution.datasets` and dispatches by `Dataset_Type`. Passing
`[M16, M1G]` in a single group means the loader sees one Training bag
and one Testing bag from one execution, the per-epoch `test_acc`
column in `training_log.txt` is meaningful (not the
"test_acc on partition the model also trained on" anti-pattern), and
the final-epoch prediction CSV the harness emits covers M1G — the
clean held-out set the Analyst will join against ground truth.

Implications for collaborators: when reading per-epoch `test_acc` from
any execution that ran one of these three new experiments, the number
is on a clean held-out partition. When reading per-epoch `test_acc`
from an execution that ran one of the legacy `cifar10_quick` /
`cifar10_default` / `cifar10_extended` experiments in this catalog,
~30% of the test samples were also in train; the number is inflated.

---

<a id="tk-005"></a>
### tk-005 — Hyperparameter spread chosen so the three runs visibly differentiate, not so any one wins ([execution YHP](https://localhost/id/27/YHP@355-RYPE-KKW8))
**When:** 2026-05-28T14:25:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-004](#tk-004) (the Toronto pair these runs evaluate against)

Ran three executions on `cifar10_toronto_pair` to give the Analyst
something to rank — not to win a benchmark. The three configs were
picked so they exercise *different* parts of the pipeline:

- [execution W76](https://localhost/id/27/W76@355-RYPE-KKW8) —
  `cifar10_toronto_quick`: 3 epochs, 32->64 channels, batch 128, lr=1e-3,
  weight decay=0. Underfit by design (the model has barely started
  learning); final test_acc 24.00%. Smoke-test that the pipeline runs.
- [execution XCE](https://localhost/id/27/XCE@355-RYPE-KKW8) —
  `cifar10_toronto_default`: 10 epochs, 32->64 channels, batch 64, lr=1e-3,
  weight decay=0. The "what an out-of-the-box config gets you" number;
  final test_acc 37.82%. Train_acc closed to 67% — overfitting started
  but isn't dominant.
- [execution YHP](https://localhost/id/27/YHP@355-RYPE-KKW8) —
  `cifar10_toronto_large`: 20 epochs, 64->128 channels, hidden=256,
  batch 64, lr=1e-3, weight decay=0. Hits memorisation: train_acc
  reaches 100% by epoch 17 while test_acc plateaus around 40%. Best
  test_acc (42.36%) actually lands at epoch 13, not at the final
  epoch (41.09%). The prediction CSV the harness commits is from the
  *final* epoch (memorised), not from the best epoch — the
  early-stopping policy isn't wired in (see the
  `# Validation lane wiring` comment block in
  `src/models/cifar10_cnn.py` for the deliberate gap).

What I did *not* try: lower learning rates (`cifar10_slow_lr` is in
the model-config registry but would need >30 epochs to differentiate
on 550 images and the goal is pipeline differentiation, not
convergence); dropout/weight-decay sweeps (`cifar10_regularized`
exists but the overfitting signal in YHP is already the
recommendation, not a problem to solve in this arc); seed sweeps
(seed=42 is the default; reproducibility across reruns wasn't the
question this arc was asking). Multirun configs in
`src/configs/multiruns.py` also weren't exercised — three named
single-experiment runs are easier for the Analyst to point at by
execution RID than four anonymous children of one multirun parent.

Implications for collaborators: the three executions are *intended*
to span underfit → reasonable → overfit, with rising final test_acc
(24% → 38% → 41%). Any ranking the Analyst does will see W76 < XCE <
YHP on raw accuracy, but YHP is also where the per-class confusion
patterns get interesting because the model has memorised the training
distribution. The right "best" run depends on which question the
Analyst is asking — accuracy, calibration, or generalisation.

---

<a id="tk-006"></a>
### tk-006 — Convention — the prediction CSV committed by training is final-epoch, not best-epoch
**When:** 2026-05-28T14:35:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-005](#tk-005) (where this convention started to matter)

The training entry point `cifar10_cnn()` in
`src/models/cifar10_cnn.py` runs `epochs` epochs and then calls
`save_training_artifacts(...)` + `record_predictions(...)` with the
*last* model state. There is no save-best, no early-stopping
checkpoint, no model-selection-by-val-acc logic. The validation lane
(`val_loader`) is wired through `evaluate()` and surfaces a `val_acc`
column in the per-epoch training log when a Validation bag is
present, but the validation metric does not drive checkpointing — see
the `# Validation lane wiring` comment in `cifar10_cnn.py`. The
`source_label` on every recorded prediction row is `epoch_N` where N
== the configured `epochs`, never `epoch_best`.

Implications for collaborators: when reading
`prediction_probabilities.csv` from any execution, the rows reflect
the final-epoch model state, which may be the worst test_acc the run
ever saw (see [tk-005](#tk-005) for an example: YHP's best test_acc
was 42.36% at epoch 13; the committed CSV is from epoch 20 at
41.09%). If you need best-epoch predictions, you have two options:
(a) run with the `cifar10_test_only` experiment using the saved
weights and a different `epochs` value (the weights checkpoint is
the *final* state, so this still won't recover an earlier checkpoint
— it only re-evaluates), or (b) extend the model to save and re-emit
the best-epoch state. Option (a) is the e2e-run-compatible workaround;
option (b) is fix-pass scope.

---

<a id="tk-007"></a>
### tk-007 — Accuracy ranking and calibration ranking disagree for the Toronto-pair runs; calibration is the more useful ranking for a domain user ([execution 1012](https://localhost/id/27/1012@355-RZWY-9B9R))
**When:** 2026-05-28T17:45:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-005](#tk-005) (hyperparameter spread the Modeler intentionally constructed), [tk-006](#tk-006) (final-epoch CSV convention that makes YHP's overconfidence visible)

Built the joined wide table (Image_RID × ground-truth × per-model
prediction + per-class probabilities) over the full 550-image M1G
test partition and ranked the three Toronto runs two different ways:

| Run | Top-1 acc (n=550) | "Confidently-wrong" (conf ≥ 0.8 ∧ wrong) | Mean conf when wrong |
|---|---|---|---|
| W76 (3 ep) | 24.0 % | 1 / 418 errors (0.2 %) | 0.245 |
| XCE (10 ep) | 37.8 % | 23 / 342 errors (6.7 %) | 0.472 |
| YHP (20 ep) | 41.1 % | **171 / 324 errors (52.8 %)** | 0.774 |

By accuracy: YHP > XCE > W76. By calibration: XCE > YHP > W76. The
two rankings disagree on the top two slots and the disagreement is
load-bearing: 31 % of all 550 M1G test images get a confidence ≥ 0.8
prediction from YHP that is actually wrong, vs 4 % for XCE and 0.2 %
for W76. YHP has learned to be confident as a side-effect of
memorising the 550-image training set (train_acc hit 100 % by epoch
17 per Modeler tk-005); the final-epoch CSV convention (tk-006) is
what makes that overconfidence the *committed* model state rather
than a transient mid-training artifact.

For an analyst handing the model off to a domain reviewer who wants
"the model is uncertain → look at this image yourself" to be a
useful gate, XCE is the better pick despite being 3 points less
accurate: only 23 of its predictions clear the 0.8-confidence bar
incorrectly, and the 5 of 550 predictions it makes at conf < 0.2 are
a real "I don't know" signal. YHP makes zero predictions at conf <
0.2 — the model never *says* it doesn't know, even when wrong.
W76's calibration is technically the best (mean confidence is barely
higher when right than when wrong), but it achieves that by failing
to learn — 165 of 550 predictions are at conf < 0.2, basically
random.

Weighed alternatives: ranked by macro-AUC instead (the
roc_analysis.ipynb default). YHP (0.822) > XCE (0.813) > W76 (0.735).
That ranking *agrees* with accuracy and doesn't capture the
calibration problem — AUC is threshold-free, so it doesn't punish a
model for being confidently-wrong as long as the wrong-confident
predictions are still ranked below the right-confident ones in some
threshold ordering. AUC is the right metric for "does the model rank
images correctly"; the confidently-wrong count is the right metric
for "is the model's confidence trustworthy as a triage signal." The
report leads with both rankings rather than picking one, and names
the use case each is right for.

Implications for collaborators: the wide table
(`docs/reports/joined-wide-table.csv`, also attached to this
execution as an Execution_Asset) is the artifact downstream analyses
should anchor on, not the headline accuracy number. Anyone who needs
"the best run for triage" should reach for XCE; anyone who needs
"the best raw classifier" should reach for YHP and live with the
overconfidence. A fix-pass on save-best-by-val-acc checkpointing
(the validation lane is already wired through `cifar10_cnn.py` per
tk-006; only the save-best policy is missing) would close most of
the YHP overconfidence gap without retraining.

---

<a id="tk-008"></a>
### tk-008 — YHP's biggest pairwise confusion is bird ↔ deer (21 mix-ups), not a domain-intuitive pair; signals a feature-learning gap ([execution 1012](https://localhost/id/27/1012@355-RZWY-9B9R))
**When:** 2026-05-28T17:50:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-007](#tk-007) (the joined wide table this confusion analysis was computed on)

Inspected the symmetric off-diagonal mass in the per-model confusion
matrices on the 550-image M1G test set. The expected CIFAR-10
confusions (cat ↔ dog, automobile ↔ truck, airplane ↔ ship) all
show up in both XCE and YHP at counts of 15–23 — they're the visual
pairs a human looking at 32×32 thumbnails would also confuse. The
*unexpected* result is that YHP's largest single pair is **bird ↔
deer at 21 mix-ups (14 bird→deer, 7 deer→bird)**, which is not a
visual confusion a domain reader would predict.

Reading the per-class probabilities in the wide table: when YHP
mispredicts a bird as a deer, its top-3 is typically deer / horse /
dog — the "small-or-medium subject in a natural-background scene"
cluster. When it mispredicts a deer as a bird, top-3 is bird / cat /
frog — the same "natural-background small-subject" cluster, from
the other side. The model has learned a *scene texture* feature
that dominates whatever silhouette feature would distinguish a
flying bird shape from a four-legged deer shape at 32×32. A human
reader uses silhouette as the primary cue at this resolution; the
model isn't.

Bird ↔ deer also outranks cat ↔ dog (18 mix-ups) and automobile ↔
truck (18) for YHP, which is the headline domain-intuition-fails
result of this analysis. Cat ↔ dog and automobile ↔ truck are the
confusions a non-ML reader expects; bird ↔ deer is the one that
tells them something about *how* this model is failing that
inspection of the confusion matrix alone would surface but a single
"top-1 accuracy" number would hide.

For XCE the same bird ↔ deer signal is weaker — only 13 mix-ups, vs
its 23-count automobile ↔ truck top spot — suggesting the
scene-texture confusion gets *worse* as the model gains capacity on
this dataset size, not better. That's consistent with the broader
overfitting story in tk-007 / tk-005: the extra parameters in YHP's
64→128 channel block found a shortcut feature (scene texture) on
the 550-image training set that doesn't generalise.

Weighed alternatives: ranked the confusions by per-class recall gap
instead of by pairwise count. That ranking puts cat (0.273 recall —
the bottleneck for all three models) at the top and surfaces "the
model can't find cat" as the result, which is also true but is the
*expected* CIFAR-10 result (cat is well-known as the hardest
class). The pairwise-confusion ranking is the one that surfaced
something a non-expert reader wouldn't predict, so the report leads
with pairwise rather than per-class. Both are in the
`per-class-recall.csv` and `per-class-confusion-long.csv` for any
reader who wants to ask the other question.

Implications for collaborators: if a future iteration of this
project pushes for higher accuracy (>50 % on M1G), the silhouette
vs scene-texture failure mode is a more useful direction than
"train longer with the same data." Either more cats (the bottleneck
class — see tk-007 implications) or augmentations that break scene
texture cues (random crop, color jitter) would target the actual
failure mode bird ↔ deer surfaces. Just adding more epochs on the
current 550-image training set would deepen the shortcut, not fix
it.
