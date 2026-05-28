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
