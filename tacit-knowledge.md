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
### tk-001 — Curator substrate audit: all splits perfectly class-balanced, no within-pair leakage ([dataset F28](https://localhost/id/168/F28))
**When:** 2026-05-30T00:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

Audited the freshly-bootstrapped CIFAR-10 substrate (catalog 168) before
the Modeler/Analyst arcs. Question: do the 13 datasets actually represent
what their names and types imply, and is the ground truth sound? Joined
every split's Image members to the `Image_Classification` ground-truth
feature ([feature on Image](https://localhost/id/168/F28)) and tabulated
per-class counts.

Findings (all confirmed, read-only):
- **Ground truth is complete and unique.** 1100 GT rows, 1100 distinct
  images, exactly one label per image, no missing labels.
- **Every split is *exactly* class-balanced** — min == max per class in
  all nine member-bearing datasets: complete 110/class, canonical
  train/test ([F2T](https://localhost/id/168/F2T) /
  [F34](https://localhost/id/168/F34)) 55/class, small split 50/class,
  labeled split ([NE8](https://localhost/id/168/NE8) /
  [NEJ](https://localhost/id/168/NEJ)) 44 & 11/class, small-labeled
  ([PHT](https://localhost/id/168/PHT) /
  [PJ4](https://localhost/id/168/PJ4)) 40 & 10/class. The canonical
  F2T/F34 split is balanced even though its description doesn't claim
  stratification.
- **No leakage *within* any train/test pair**: F2T∩F34, F3W∩F46,
  NE8∩NEJ, PHT∩PJ4 are all 0 shared images.

Implications for collaborators (Modeler/Analyst): the substrate needs no
balance correction. Any of the pairs is fair to train+evaluate on *as a
pair*. The cross-pair traps are recorded separately in
[tk-002](#tk-002) — read that before mixing datasets from different
families.

**Weighed alternatives:** *(none — this was characterization, not a
choice. Considered creating a balanced curated subset and concluded none
was needed: the substrate is already uniform.)*

<a id="tk-002"></a>
### tk-002 — Trap: the "labeled_testing" splits are carved from training, not held out ([dataset NEJ](https://localhost/id/168/NEJ))
**When:** 2026-05-30T00:05:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (the balance/leakage audit this trap was found during)

The labeled-split family is 100% drawn from the canonical *training*
partition [F2T](https://localhost/id/168/F2T), not from the held-out test
partition [F34](https://localhost/id/168/F34). Verified by set
intersection: every member of NE8, [NEJ](https://localhost/id/168/NEJ),
PHT, [PJ4](https://localhost/id/168/PJ4) is in F2T and *none* is in F34.
NE8∪NEJ exactly reconstructs F2T's 550 images; PHT∪PJ4 is a 500-image
subset of F2T.

The trap is a signal mismatch. The catalog `Dataset_Type` on NEJ/PJ4 is
`Testing`+`Labeled`, and their descriptions read "Testing subset" — both
of which a consumer dispatching on `Dataset_Type` (the authoritative
"what is this for" signal) reads as *held-out evaluation set*. But these
are **internal validation splits of the training pool** — useful only
when paired with their sibling training set (`cifar10_labeled_training`
→ NE8, `cifar10_small_labeled_training` → PHT). The `src/configs/datasets.py`
header comments (lines ~70–78) already describe this family correctly as
"cross-validation workflows ... where the test_batch must stay unseen for
final evaluation," so the *config author's* intent matches reality — it's
the catalog-side `Dataset_Type`/description that overstate these as test
sets.

Concrete leakage facts (verified):
- Train F2T → eval NEJ: **110/110 NEJ images are in F2T** — total leakage.
- Train F2T → eval PJ4: **100/100 PJ4 images are in F2T** — total leakage.
- Train NE8 → eval F34: 0 shared — SAFE.
- Train PHT → eval F34: 0 shared — SAFE.
- Train F2T → eval F34: 0 shared — SAFE.

Implications for collaborators:
- For a **held-out** evaluation number, evaluate against
  [F34](https://localhost/id/168/F34) (`cifar10_testing`), regardless of
  which training set was used. Do NOT report NEJ/PJ4 accuracy as a
  held-out metric for an F2T-trained model.
- NEJ/PJ4 are the right choice only as the *validation* half of their own
  sibling pair (hyperparameter selection, early stopping), never as the
  final test set for a model trained on the full F2T.
- The small split ([F3W](https://localhost/id/168/F3W) /
  [F46](https://localhost/id/168/F46)) is an **independent** 1000-image
  draw from the full 1100 pool — F3W overlaps F2T (500 shared) and F46
  overlaps F34 (500 shared). So F3W/F46 and F2T/F34 are not
  interchangeable: training on F3W and evaluating on F34 would leak.
  Keep each family internally consistent.

**Weighed alternatives:** considered "fixing" the catalog by changing
NEJ/PJ4's `Dataset_Type` from `Testing` to `Validation` (the
`Validation` term exists in the `Dataset_Type` vocab). Did **not** do so
in this arc — a `Dataset_Type` change is a catalog mutation that flips the
dataset to a new dev version and would muddy the e2e provenance baseline;
it's also arguably a defensible naming if one reads "testing" as "the
held-out half of *this* split." Recorded the trap here instead so the
decision stays with the user. See finding
`findings/curator/01-labeled-split-is-training-derived.md`.

<a id="tk-003"></a>
### tk-003 — Split hierarchy: labeled splits are NOT registered as catalog children of F2T ([dataset F2T](https://localhost/id/168/F2T))
**When:** 2026-05-30T00:08:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-002](#tk-002) (established the F2T-derivation that the hierarchy fails to record)

The dataset nesting hierarchy records only two split roots:
[F2J](https://localhost/id/168/F2J) → {F2T, F34} and
[F3M](https://localhost/id/168/F3M) → {F3W, F46}. The two labeled-split
roots, [NE0](https://localhost/id/168/NE0) → {NE8, NEJ} and
[PHJ](https://localhost/id/168/PHJ) → {PHT, PJ4}, are **standalone** —
they have no parent and are not children of
[F2T](https://localhost/id/168/F2T), even though their members are 100%
derived from F2T (see [tk-002](#tk-002)). The only catalog-side record of
that derivation is the free-text dataset description ("...subset of F2T,
stratified by Image_Class.Name, seed=42"), which is advisory prose, not a
walkable lineage edge.

Implications for collaborators: a `deriva_ml_list_dataset_relations` or
`deriva_ml_get_lineage` walk from F2T will **not** surface NE8/NEJ/PHT/PJ4
as descendants — the provenance link lives in prose only. If you need to
know "what was derived from the training partition," you can't get it from
the hierarchy; you have to read descriptions or re-derive by set
intersection (as this audit did). This is a gap in the bootstrap loader's
split-registration, recorded in
`findings/curator/02-labeled-split-not-registered-as-child.md`.
