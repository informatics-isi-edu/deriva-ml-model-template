# `Dataset_Type=Testing` on labeled-split partitions overstates them as held-out test sets

**Persona:** Curator
**Phase:** Substrate characterization (class-balance + leakage audit of the 13 bootstrap datasets)

## What happened

While auditing catalog 168 (`e2e-test-20260530`, localhost) before the
Modeler/Analyst arcs, I checked whether each split dataset represents what
its name and `Dataset_Type` imply.

**Attempted:** verify that the four "testing" partitions are usable as
held-out evaluation sets.

**Expected:** a dataset tagged `Dataset_Type = Testing` (the authoritative
"what is this for" signal per the `dataset-lifecycle` skill) is disjoint
from the data a model would be trained on.

**Actual:** the labeled-split testing partitions are carved *from the
training pool*, so they leak completely against a model trained on the
canonical training partition:

- `NEJ` (`cifar10_labeled_testing`, types `Testing`+`Labeled`, 110 images):
  **110/110 members are in `F2T`** (the canonical training partition);
  0/110 are in `F34` (the canonical held-out test partition).
- `PJ4` (`cifar10_small_labeled_testing`, types `Testing`+`Labeled`, 100
  images): **100/100 members are in `F2T`**; 0/100 are in `F34`.
- `NE8 ∪ NEJ` exactly reconstructs `F2T`'s 550 images (80/20 partition of
  the training pool). `PHT ∪ PJ4` is a 500-image subset of `F2T` (90/10).

So `NEJ`/`PJ4` are **internal validation splits of the training pool**,
not held-out test sets. They are sound *only* when paired with their
sibling training set (`NE8` / `PHT`). The mismatch is between the
catalog-side signal (`Dataset_Type = Testing`, description "Testing
subset") and the actual role (validation half of a training-pool split).

Notably, the `src/configs/datasets.py` header comments (lines ~70–78)
*already* describe this family correctly — "cross-validation workflows ...
where the test_batch must stay unseen for final evaluation." The config
author's intent matches reality; it is the catalog `Dataset_Type` /
description that overstate the role.

Safe vs. leaky train→eval pairings (all verified by set intersection):

| Train | Eval | Shared images | Verdict |
|-------|------|---------------|---------|
| `F2T` (550) | `NEJ` (110) | 110 | LEAK (total) |
| `F2T` (550) | `PJ4` (100) | 100 | LEAK (total) |
| `F2T` (550) | `F34` (550) | 0 | SAFE |
| `NE8` (440) | `F34` (550) | 0 | SAFE |
| `PHT` (400) | `F34` (550) | 0 | SAFE |

## Reproduction

Read-only; no catalog mutation. Against localhost catalog 168:

1. Pull ground truth: `query_attribute` on
   `e2e-test-20260530:Execution_Image_Image_Classification`, attributes
   `["Image", "Image_Class"]` (1100 rows, one label per image).
2. Pull membership: `query_attribute` on
   `e2e-test-20260530:Dataset_Image/Dataset=<RID>`, attribute `["Image"]`,
   for each of `F2T`, `F34`, `NEJ`, `PJ4`, `NE8`, `PHT`.
3. Intersect member sets:
   - `set(NEJ) <= set(F2T)` → True (110/110)
   - `set(PJ4) <= set(F2T)` → True (100/100)
   - `set(NEJ) & set(F34)` → empty
   - `set(NE8) & set(F34)` → empty

(Equivalent script: a small deriva-py `ErmrestCatalog` reader using
`get_credential("localhost")` for auth — the MCP `query_attribute` path
works without writing a script.)

## Notes

- This is the canonical "splits don't represent what their names imply"
  case the Curator role exists to catch. The data is *internally* sound
  (perfectly balanced, no within-pair leakage — see finding inputs); the
  hazard is purely in how a downstream consumer interprets the
  `Dataset_Type` signal.
- I did **not** mutate the catalog to "fix" the `Dataset_Type` (e.g.
  retag `NEJ`/`PJ4` as `Validation`, a term that exists in the
  `Dataset_Type` vocab). That is a judgment call left to the user: it is a
  catalog mutation that would flip the dataset to a new dev version and
  perturb the e2e provenance baseline, and one could defend the current
  naming if "testing" is read as "the held-out half of *this* split."
  Recorded the trade-off in `tacit-knowledge.md` `tk-002` instead.
- Handoff guidance for the Modeler/Analyst is in `tacit-knowledge.md`
  `tk-001`/`tk-002`: report held-out accuracy against `F34`
  (`cifar10_testing`) regardless of training set; use `NEJ`/`PJ4` only as
  the validation half of their own sibling pair.
