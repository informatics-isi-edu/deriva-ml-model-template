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
