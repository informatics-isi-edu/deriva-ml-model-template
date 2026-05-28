# Train/test leakage in training-derived split datasets (TCC, VAP)

**Persona:** Curator
**Phase:** Catalog characterisation — split-integrity smell-check

## What happened

The training-derived holdout splits do not satisfy the basic property
that a train/test split should be **disjoint**:

| Split family | Train RID | Test RID | Train n | Test n | Overlap |
|---|---|---|---|---|---|
| `cifar10_labeled_split` (TCC) | TCM (440 advertised) | TCY (110 advertised) | **361 actual** | **105 actual** | **33 images in both** |
| `cifar10_small_labeled_split` (VAP) | VAY (400 advertised) | VB8 (100 advertised) | **339 actual** | **95 actual** | **24 images in both** |

Two problems are visible at once:

1. **Counts don't match advertised sizes.** TCM is advertised as 440
   members and TCY as 110, but the actual unique-image counts are 361
   and 105. Same for VAY (400 → 339) and VB8 (100 → 95). The
   `Description` column on each dataset still carries the advertised
   numbers, as does `src/configs/datasets.py`, `README.md`, and
   `CLAUDE.md`. A user reading those will expect 440 training images
   and get 361.

2. **Train and test halves overlap.** 33 images appear in both TCM and
   TCY. Same image RID is a member of both datasets — they will appear
   in the bag for either side. A model trained on TCM and evaluated on
   TCY will see 33 images at test time it saw at train time. Same
   issue at smaller scale on VAY/VB8 (24 overlap).

The Toronto family (M16/M1G, M28/M2J) is **not** affected. M16 ∩ M1G =
∅ because those come from different Toronto source pools (training
batches vs test_batch); the data leakage is confined to splits
produced by `split_dataset()` on M16.

## Reproduction

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id=27)
ds = ml.catalog.getPathBuilder().schemas["e2e-test-20260528"]
di = ds.Dataset_Image
for a, b in [("TCM","TCY"), ("VAY","VB8")]:
    s1 = {r["Image"] for r in di.filter(di.Dataset == a).entities().fetch()}
    s2 = {r["Image"] for r in di.filter(di.Dataset == b).entities().fetch()}
    print(f"{a}∩{b}: {len(s1 & s2)}  (|{a}|={len(s1)}, |{b}|={len(s2)})")
# TCM∩TCY: 33  (|TCM|=361, |TCY|=105)
# VAY∩VB8: 24  (|VAY|=339, |VB8|=95)
```

## Notes

**Root cause (high confidence).** `_cifar10_datasets.py` calls
`split_dataset(..., row_per="Execution_Image_Image_Classification",
include_tables=["Image","Execution_Image_Image_Classification"], ...)`.
With `row_per` set to the *feature* table, the splitter partitions
feature **rows**, not image RIDs. Every doubly-tagged image (finding
01: 250 of M16's 550 carry two feature rows from executions 854 and
HSR) has *two independent draws* in the partition — one row can land
on the training side while the other lands on the testing side, and
the underlying image is now in both halves of the dataset. All 33
overlapping images in TCM∩TCY are doubly-tagged (100% match); same
for the 24 overlapping in VAY∩VB8.

The discrepancy between advertised and actual sizes likely comes from
the same mechanism — the splitter is asked for "440 from a pool of 800
feature rows" (550 images × 1.45 avg feature rows), so it draws 440
*rows* corresponding to a slightly different unique-image count.

**Implications for downstream personas (real, not theoretical):**

- The Modeler should treat TCM/TCY and VAY/VB8 as **leaky** for
  evaluation purposes. A clean alternative for held-out evaluation
  in this catalog is M16 trained / M1G evaluated (Toronto family,
  zero overlap by construction). Both M16 and M1G carry ground-truth
  labels.
- The Analyst should be aware that any per-image accuracy / confusion
  matrix computed from a TCM-trained model evaluated on TCY will
  include ~30% double-counted images and may overstate test
  performance.
- The discrepancy between advertised and actual counts also affects
  any consumer that has hard-coded "TCM = 440 images" in a downstream
  config (e.g., expected dataloader length, sample budget).

**Workaround the Curator considered.** Building a new clean
training-derived split by image RID (not by feature row) would solve
this in one new dataset. Decision deferred — see tacit-knowledge.md
entry. The short version is: the leakage is small (≤ 30% of test
images), the Toronto family is a leakage-free alternative already
wired into `datasets.py`, and creating a "cifar10_clean_labeled_split"
would diverge the demo catalog further from what `load-cifar10`
produces by default. Filing the finding and leaving the leak in place
is the right call for an e2e fitness run; a fix-pass on
`_cifar10_datasets.py` would be the durable answer.

Detected during the 2026-05-28 e2e run with sibling versions:
deriva-ml v1.40.2, deriva-ml-mcp v0.5.9, deriva-mcp-core latest main,
deriva-skills v1.2.4, deriva-ml-skills v1.4.11.
