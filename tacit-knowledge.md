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
