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

## 2026-06-01 — Curator characterization of catalog 2 (e2e-test-20260601)

**Context.** Bootstrapped CIFAR-10 catalog handed off for the multi-persona
e2e run. Catalog 2 on localhost, domain schema `e2e-test-20260601`. The
Curator's job was to characterize the substrate before the Modeler builds on
it. Full data checks done with read-only set-algebra over `Dataset_Image`
membership and the `Image_Classification` ground-truth feature.

**What the data is (verified, all good).**
- 1100 images, ground truth written by a **single** execution
  (`Image_Classification` feature; exec `CWC`), `Confidence` null on all (correct
  — it's ground truth, the column exists for model predictions to reuse).
- **Perfectly class-balanced** everywhere: 110/class over the full 1100, and
  proportionally uniform in every one of the 13 datasets. No image carries a
  conflicting class. No smell-test failures.
- `F38` (complete, flat 1100) `= F3T (train 550) ⊎ F44 (test 550)` exactly,
  disjoint. The canonical full split `F3J → {F3T, F44}` is clean and leakage-free.
- All four split families are internally disjoint (train ∩ test = 0 each).

**The one gotcha that matters downstream — split source pools.**
There are four split families and they do NOT all draw from the same pools:
- **Small split** `F4M → {F4W train(500), F56 test(500)}`: train 100% from
  F3T, test 100% from F44. A *proper* scaled-down mirror of the canonical split.
  Safe to treat F56 as a real held-out test set relative to F3T/F4W.
- **Labeled split** `NF0 → {NF8 train(440), NFJ test(110)}` and **small
  labeled split** `PJM → {PJW train(400), PK6 test(100)}`: **both children are
  carved 100% from F3T (the TRAINING pool)** — stratified 80/20 re-splits of the
  labeled training set (exec `NE0`: "Create Labeled_Split and Small_Labeled_Split
  from the training set"). NFJ and PK6 have **zero overlap with the canonical
  test partition F44**.

**Why this is a trap, not a bug.** NFJ/PK6 are valid hold-outs *relative to
their own sibling train sets* (NF8/PJW). But they sit entirely inside F3T. So:
- Train on NF8, eval on NFJ → fine (disjoint within NF0).
- Train on **F3T or F38**, eval on **NFJ/PK6** → **silent leakage**: the eval
  images were in training. The catalog can't warn you — NFJ and F44 are both
  typed `Testing`+`Labeled`, and NFJ has no catalog parent link back to F3T
  (the split parents NF0/PJM are siblings, not F3T). The only machine-checkable
  signal is set-intersection on membership.

**Decision / guidance for the Modeler & Analyst.**
- For an honest **train + held-out-eval against the canonical test partition**,
  use the matched pair from one family and don't cross families:
  - Full: train `F3T` (550) / eval `F44` (550).
  - Small (fast iteration): train `F4W` (500) / eval `F56` (500).
- For a **self-contained labeled train/eval where both partitions come from the
  same labeled training distribution** (e.g. quick ROC where you want both sides
  labeled and identically distributed): use a single family end-to-end —
  `NF8`/`NFJ` together, or `PJW`/`PK6` together. **Never** pair `NF8`/`NFJ` or
  `PJW`/`PK6` against `F3T`/`F38`/`F44`.
- The README's steer ("use `*_labeled_split` for evaluation/ROC") is correct
  *only* if you also train within that same family. It is the right choice for
  ground-truth-on-both-partitions evaluation; it is the wrong choice as a
  held-out test set for a model trained on the full training partition.

**Catalog left unchanged.** Data is sound and correctly typed; no curation
mutation was warranted. The gotcha is a naming/expressiveness gap in
`Dataset_Type`, recorded as a finding
(`findings/curator/labeled-test-splits-drawn-from-training-pool.md`), not a data
defect. Did not add a vocab term or re-nest datasets — that would be a schema
decision for the platform owners, and destructive/structural changes are out of
scope for this arc.

**Tooling note.** `ReadMcpResourceTool` is unavailable in this harness, so the
`deriva://...` orientation + read resources could not be read; all reads went
through `deriva_ml_*` tools + read-only Python. See
`findings/curator/mcp-resource-read-tool-unavailable.md`.

---

## 2026-06-01 — Modeler: three pipeline-validation runs in the small labeled family

**Context.** Modeler arc of the e2e run. Goal was to stress-test the modelling
pipeline against the Curator's substrate — confirm training launches cleanly,
produces a learning signal, and lands outputs with provenance the Analyst can
use — **not** to chase accuracy. Three CIFAR-10 CNN executions, all on
[dataset PJM](https://localhost/chaise/record/#2@356-DC66-N6X8/deriva-ml:Dataset/RID=PJM)
(`cifar10_small_labeled_split`, the `Split` parent that flattens to train PJW
400 / eval PK6 100, both stratified from F3T seed=42):

- Smoke: [execution QK8](https://localhost/chaise/record/#2@356-DC66-N6X8/deriva-ml:Execution/RID=QK8)
  — `cifar10_quick` (3 epochs, lr 1e-3, batch 128).
- Run A: [execution QWA](https://localhost/chaise/record/#2@356-DC66-N6X8/deriva-ml:Execution/RID=QWA)
  — `cifar10_regularized` (20 epochs, dropout 0.25 — softens reliance on any
  single feature to fight overfit — weight decay 1e-4, lr 1e-3).
- Run B: [execution R5C](https://localhost/chaise/record/#2@356-DC66-N6X8/deriva-ml:Execution/RID=R5C)
  — `cifar10_fast_lr` (15 epochs, lr 1e-2, no dropout).

**Why this dataset, and the no-cross-families rule.** Per the Curator's
leakage finding (above; `findings/curator/labeled-test-splits-drawn-from-training-pool.md`),
the labeled "testing" splits NFJ/PK6 are carved from the *training* pool F3T,
so training on F3T/F38 and evaluating on PK6 would be silent leakage the catalog
can't warn about. The leakage-safe rule is: **train and eval within ONE family,
never crossed.** Using the `Split` parent PJM accomplishes this structurally —
the model's `build_loaders` harness flattens PJM to its own children (PJW train,
PK6 eval), so train and eval are guaranteed same-family. Verified via
`deriva_ml_get_lineage`: PK6 predictions trace back through PJM → F3T cleanly,
and no F3T/F38/F44 dataset was ever an input. The small family was chosen over
the full F3T/F44 family for fast iteration — 400+100 images train in seconds.

**Did variation produce variation? Yes.** The three configs produced clearly
distinct training dynamics, confirming the pipeline reflects hyperparameter
changes rather than collapsing to one result:
- Smoke (3 ep): train_acc 7% → 28.75%, test_acc ~20% — a learning signal, no
  more (too few epochs to converge).
- Run A (regularized, 20 ep): train_acc → 87.25%, test_acc → 32%. The large
  train-vs-test gap is textbook overfitting on a 400-image train set — expected
  at this scale, and a meaningfully different curve from the smoke run.
- Run B (fast_lr 1e-2, 15 ep): epoch-1 train_loss spiked to 3.30 (the high
  learning rate overshoots early), then converged slowly and noisily to
  train_acc 45.75% while test_loss *diverged* upward (2.30 → 3.07) — an
  unstable-optimization signature distinct from both other runs.

**Implications for the Analyst.** These are pipeline-validation runs, not
performance baselines — do not cite any of these accuracy numbers as a model
capability claim. The eval set for all three is **PK6** (100 labeled images, in
PJM). Each run recorded exactly 100 `Image_Classification` feature rows on PK6
with a populated `Confidence` (softmax max-probability) plus a wider
`prediction_probabilities.csv` asset carrying per-class probabilities — that CSV
is the surface ROC analysis consumes. Because PK6 carries ground truth, accuracy
and ROC are computable. Output asset RIDs are wired into `src/configs/assets.py`
(`preds_*`, `weights_*`, and the combined `roc_modeler_e2e_three_way`).

**Reproducibility note.** All three configs pin `seed=42` (drives weight init,
shuffle order, numpy/random). Runs were launched from a clean git tree so the
workflow commit hash is honest; the dirty-tree override was used only for the
read-only config smoke tests, never for a recorded training run.

---

## 2026-06-01 — Modeler: behavior — Hydra rejects free-text `description=` overrides with parens/commas

**Context.** Tried to give Run A a descriptive execution name by passing
`description="Modeler e2e Run A: regularized (20ep, dropout 0.25, wd 1e-4) ..."`
as a `deriva-ml-run` override. The run failed before any catalog write with
`mismatched input ' (' expecting <EOF>` from Hydra's override grammar.

**Behavior (durable).** `deriva-ml-run` passes positional `key=value` args
through to Hydra's override parser, which treats `(`, `)`, and `,` as grammar
metacharacters even inside a value. A free-text `description=` containing those
characters is a parse error, not a runtime error — the process never starts.
This is a Hydra-grammar constraint, not a deriva-ml bug.

**Workaround applied.** Sanitized the description to drop parens/commas
(`'description=Modeler e2e Run A regularized 20ep dropout0.25 wd1e-4 on ...'`,
single-quoted for the shell). The auto-composed description for
`+experiment=<name>` presets sidesteps this entirely — when an experiment preset
is used, deriva-ml composes the description from the preset's text plus the
resolved overrides, so no manual `description=` is needed. For ad-hoc
`model_config=` / `datasets=` runs where you want a meaningful name, either keep
the description free of grammar metacharacters or define a one-line experiment
preset. Filed as `findings/modeler/hydra-description-override-grammar.md`.

---

## 2026-06-01 — Analyst: ranked the three runs by AUC, not top-1 accuracy

**When:** 2026-06-01T11:55:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

**Context.** Analyst arc. Evaluated the three Modeler runs — smoke
[QK8](https://localhost/id/2/QK8@356-DC66-N6X8) / regularized
[QWA](https://localhost/id/2/QWA@356-DC66-N6X8) / fast_lr
[R5C](https://localhost/id/2/R5C@356-DC66-N6X8) — all on the same 100-image
eval set [PK6](https://localhost/id/2/PK6@356-DC66-N6X8). Recorded the ROC
notebook as analysis execution
[REJ](https://localhost/id/2/REJ@356-DD8X-BD5W). Full write-up in
`docs/reports/2026-06-01-analysis.md`.

**Decision and why.** Ranked by **macro-AUC (one-vs-rest), not top-1
accuracy** — and the choice was load-bearing, not stylistic. The two
weaker runs *tie at 20% top-1 accuracy* but are not equivalent: macro-AUC
separates them (smoke 0.739 vs fast_lr 0.638). The regularized run wins on
every metric (acc 32%, macro-AUC 0.751, micro-AUC 0.757) so the ranking
isn't sensitive to the metric *at the top* — but the metric choice is what
distinguishes the two tied-on-accuracy runs, which is the analytically
interesting part. **macro-AUC** (mean over per-class one-vs-rest AUC, equal
weight per class) asks "do the model's probability scores *rank* the right
class above the wrong ones," a different question from "is the single
argmax guess correct." A model can be useless at committing to a final
answer while still ordering classes sensibly — that's the smoke run.

**The durable interpretive finding (the reason this entry exists).** The
two 20%-accuracy runs fail in qualitatively different ways, and the failure
mode is diagnosable from the *prediction distribution* (how many of the 10
classes the model ever predicts) plus the calibration gap:
- **Smoke (QK8) is underfit-but-informative.** Predicts only 7/10 classes,
  collapses 34/100 guesses onto "horse," near-zero confidence (mean 0.18,
  barely above the 0.10 chance floor) — yet its macro-AUC (0.739) is the
  second-best of the three because the softmax *ordering* is already
  sound. Three epochs taught it to rank, not to decide.
- **Fast_lr (R5C) is unstable-and-uninformative.** Uses all 10 classes but
  collapses 28/100 onto "deer," near-chance per-class AUC on deer (0.497),
  horse (0.496), cat (0.568). The high LR (1e-2) thrashed instead of
  converging. Genuinely the worst at discrimination despite the same 20%
  top-1 as the smoke run.
- **Regularized (QWA) is the only run whose mistakes are interpretable** —
  frog→cat, cat→dog, automobile→truck, ship→airplane, bird→deer: the
  classic CIFAR-10 confusable pairs, exactly what a domain expert expects
  to be hard at 32×32. The collapse-onto-one-class signature of the other
  two is the fingerprint of a model that *hasn't learned*, NOT of semantic
  confusion. This is the tell to teach a domain reader: "everything → one
  class" is non-learning; "confuses the visually-similar pairs" is learning.

**Calibration consequence for downstream domain readers.** Only QWA
produces confidence scores usable for triage (mean 0.67 correct / 0.59
wrong, a real +0.08 gap centered well above chance). The smoke run is
unsure of everything; fast_lr's gap exists but its scores are noisy. If
anyone later wants to threshold on confidence to auto-accept / queue for
review, QWA is the only candidate — and only after retraining at real data
volume. These remain pipeline-validation runs (400 train images); the
20–32% accuracies are NOT capability claims.

**Reproducibility note.** Computed every metric two independent ways — a
read-only script (`scripts/analyst_explore.py`) and the provenance-tracked
ROC notebook (execution REJ) — and they agreed to the digit. The notebook
correctly consumed QN6/QY8/R7A as inputs (now tagged `Input_File`), so the
ROC/confusion figures on REJ trace by lineage back to the three training
runs.

---

## 2026-06-01 — Convention — `Image_Classification` is dual-purpose (ground truth + predictions)

**When:** 2026-06-01T11:56:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

The `Image_Classification` feature on `Image` is written by two distinct
kinds of execution and the rows are **not** distinguishable by table
membership alone. The loader execution
([CWC](https://localhost/id/2/CWC@356-DC66-N6X8)) writes the ground-truth
layer with `Confidence IS NULL`; each training execution writes its
predictions with `Confidence` populated (softmax max-probability). At
Analyst time the feature held 1,400 rows = 1,100 GT (CWC) + 100 each from
QK8 / QWA / R5C — and it grows by 100 more on every future run that
predicts on a labeled set. So the raw count is a snapshot that rots; the
*convention* is what's durable.

**Implications for collaborators.** To read this feature as **ground
truth**, filter by the loader execution RID OR by `Confidence IS NULL`. To
isolate **one run's predictions**, filter by that run's execution RID. An
unfiltered `ml.feature_values("Image", "Image_Classification")` returns GT
+ every recorded prediction interleaved — rarely what an analysis wants. A
`newest`-style selector is also NOT a safe substitute for "ground truth":
newest is whichever execution last wrote (a prediction run), not the
loader. The ROC notebook handles this correctly — it auto-detects the GT
execution as the one whose rows are all `Confidence`-NULL, picking the
largest such execution when more than one exists — but a hand-written read
of this feature must apply the filter explicitly or it silently mixes
truth and predictions. (This is the consumer-facing face of the Curator's
`Dataset_Type` expressiveness finding: the catalog stores the
GT-vs-prediction distinction in a nullable value column, not in the type
system.)

---

## E2E run 2026-06-01 — wrap-up disposition (orchestrator)

- **Catalog KEPT, not deleted.** `e2e-test-20260601` (hostname `localhost`,
  **catalog_id 2**) is preserved for archeology by explicit user decision.
  The `[E2E-DROP]` commits on this branch (`80a6f31` deriva.py→id 2,
  `a78687b` datasets.py RIDs) point the worktree's `default_deriva` /
  `datasets.py` at it, so this worktree runs against catalog 2 as-is. Those
  commits are intentionally NOT cherry-picked back to `main`.
- **Branch archived** to `origin/archive/e2e-test-2026-06-01` (@ `da868fc`).
  No template-source fixes were cherry-picked back: the only genuine template
  improvement this run produced (the group-agnostic ROC description) landed on
  `main` independently as PR #58. All other non-`[E2E-DROP]` commits are run
  artifacts (reports, findings, this file, the analyst helper script).
- **Worktree left in place** at `../deriva-ml-model-template-e2e` per CLAUDE.md
  (read the reports/findings/notebooks without checking out the archive).
- **Findings dispositioned:** PR #58 (ROC description, merged); deriva-ml
  #272 = the `Dataset_Type`/leakage finding of record; deriva-ml #273 = the
  agreed remediation (a member-overlap guard — server-side dataset
  intersection, design captured, implementation deferred). The Hydra
  `description=` grammar nit is deferred; the `ReadMcpResourceTool`-unavailable
  finding is dismissed as a spawned-agent harness limitation, not a platform
  defect.
