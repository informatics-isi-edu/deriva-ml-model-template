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
