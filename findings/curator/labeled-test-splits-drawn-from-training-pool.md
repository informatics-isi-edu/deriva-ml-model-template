# Finding: "labeled testing" datasets are carved from the training pool, indistinguishable by Dataset_Type

- **Persona:** Curator
- **Date:** 2026-06-01
- **Catalog:** localhost / catalog 2 / schema `e2e-test-20260601`
- **Severity:** Medium (no data corruption; a real provenance/leakage trap for downstream personas)
- **Category:** Platform / vocabulary expressiveness + downstream-leakage risk

## What I expected

The 13 datasets divide into four split families. By name and by `Dataset_Type`,
I expected each `*_testing` dataset to be a held-out **test partition** —
disjoint from whatever a model would be trained on.

In particular I expected `cifar10_labeled_testing` (NFJ) and
`cifar10_small_labeled_testing` (PK6), both typed `Testing` + `Labeled`, to be
held-out the same way `cifar10_testing` (F44, also typed `Testing` + `Labeled`)
is held out from `cifar10_training` (F3T).

## What actually happened

Verified against the catalog (image-RID set algebra over `Dataset_Image`
membership + the `Image_Classification` ground-truth feature):

| Dataset | Type tags | n | Source pool | ∩ canonical test F44 |
|---|---|---|---|---|
| F44 `cifar10_testing` | Testing, Labeled | 550 | 100% F44 | 550 (is F44) |
| F3T `cifar10_training` | Training, Labeled | 550 | 100% F3T | 0 |
| NFJ `cifar10_labeled_testing` | Testing, Labeled | 110 | **100% F3T (training pool)** | **0** |
| PK6 `cifar10_small_labeled_testing` | Testing, Labeled | 100 | **100% F3T (training pool)** | **0** |
| F56 `cifar10_small_testing` | Testing, Labeled | 500 | 100% F44 (test pool) | 500 |

- Every dataset is **internally clean**: perfect 10-way class balance, and
  within every split family `train ∩ test = 0` (verified for all four families).
  `F38 = F3T ⊎ F44` exactly (disjoint union, 1100 images).
- The labeled split family is **not a bug**: execution `NE0` describes itself
  "Create Labeled_Split and Small_Labeled_Split from the training set," and the
  filenames in NFJ/PK6 all carry the `train_` prefix. The loader did exactly
  what it intended — a self-contained stratified train/eval split carved
  entirely from the labeled *training* pool (NF8⊎NFJ ⊆ F3T; PJW⊎PK6 ⊆ F3T).

The trap is at the **project level, not the dataset level**. NFJ and PK6 are
proper hold-outs *relative to their own sibling train sets* (NF8, PJW). But
they are **100% inside F3T**. So a Modeler who trains on the full training
partition F3T (or F38) and then evaluates on NFJ/PK6 "because it says testing"
is evaluating on images that were in the training set — silent leakage that the
catalog's own type system does not warn about.

## Repro

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id="2")
def imgs(rid):
    return {m["RID"] for m in ml.lookup_dataset(rid).list_dataset_members().get("Image", [])}
F3T, F44, NFJ = imgs("F3T"), imgs("F44"), imgs("NFJ")
print(len(NFJ & F3T), len(NFJ & F44))   # -> 110 0   (all of NFJ is in the TRAIN pool)
```

## Impact

- A downstream persona dispatching on `Dataset_Type` — exactly what the
  `dataset-lifecycle` skill says is the **authoritative** signal of intent,
  over the "advisory, non-authoritative" description — **cannot distinguish**
  NFJ (in-training-pool hold-out) from F44 (canonical held-out test). Both read
  `Testing` + `Labeled`. The distinction that actually matters for leakage
  lives only in the free-text description, which the same skill tells consumers
  not to route on.
- This is a concrete gap between the platform's stated typing discipline and
  the expressiveness of the built-in `Dataset_Type` vocabulary: there is no
  term that captures "held-out from the project's canonical training pool" vs
  "a labeled subset re-split out of the training pool."

## Suggested direction (NOT done — out of scope for this arc)

Either (a) a `Dataset_Type` qualifier term distinguishing canonical-holdout
test sets from re-split-from-train eval sets, or (b) explicit parent/child
nesting so a consumer can walk `list_dataset_parents()` and see that NFJ
descends from F3T. Today NFJ/PK6 have **no** catalog parent link to F3T (the
`NF0`/`PJM` split parents are siblings, not F3T); the only machine-checkable
signal of the shared pool is set-intersection on membership, which no consumer
will do by default. Recorded as guidance in `tacit-knowledge.md` and in the
Modeler handoff so the leakage trap is surfaced regardless.
