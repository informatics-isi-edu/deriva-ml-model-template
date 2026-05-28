# `split_dataset(row_per=feature_table)` is the wrong default for image-row datasets in `_cifar10_datasets.py`

**Persona:** Evaluator
**Severity:** High
**Category:** Bug
**Phase:** Cross-arc synthesis (consumes Curator-02)

## What happened

`scripts/_cifar10_datasets.py` calls
`split_dataset(..., row_per="Execution_Image_Image_Classification",
include_tables=["Image","Execution_Image_Image_Classification"], ...)`
to build the `cifar10_labeled_split` (TCC) and
`cifar10_small_labeled_split` (VAP) families.

`row_per` here is a *feature table*. That means the splitter
stratifies and partitions feature **rows**, not image RIDs.
Whenever the feature table has more rows than the underlying image
set — which it always will on any catalog where the loader has
been retried (see `findings/evaluator/01`) — an image with two
feature rows can have one row land in the training partition and
the other in the testing partition, putting the same image RID on
both sides of the split.

On this catalog (verified directly): TCM∩TCY = 33 image RIDs
(9% of TCM training images appear in TCY test), and VAY∩VB8 = 24
(7% of VAY training images appear in VB8 test). 100% of the
overlapping images in both cases are exactly the doubly-tagged
images from the loader-retry orphan-row issue. The Curator
characterised this as `findings/curator/02`.

This is **separable from evaluator/01** because even if the loader
were fixed tomorrow, the choice of `row_per` here is brittle: any
future feature-table consumer that ends up with `image_rid` not
being one-to-one with `feature_row` (multi-execution annotations,
multi-label features, time-series features, anything where one
image legitimately carries multiple rows) will silently leak. The
defensive default for partitioning *images* is `row_per="Image"`
with an upstream dedupe step.

## Reproduction

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id=27)
ds = ml.catalog.getPathBuilder().schemas["e2e-test-20260528"]
di = ds.Dataset_Image
for a, b in [("TCM","TCY"), ("VAY","VB8")]:
    s1 = {r["Image"] for r in di.filter(di.Dataset == a).entities().fetch()}
    s2 = {r["Image"] for r in di.filter(di.Dataset == b).entities().fetch()}
    print(f"{a}∩{b} = {len(s1 & s2)}")
# TCM∩TCY = 33
# VAY∩VB8 = 24
```

Compare with the Toronto pair (M16/M1G), which was built from two
*distinct* Toronto source batches rather than via `split_dataset`:
overlap is 0 by construction.

## Why upgraded to High / Bug

- The template ships `cifar10_labeled_split` and
  `cifar10_small_labeled_split` as named "labeled split" datasets
  whose advertised semantics are "use this for held-out evaluation
  on labeled data". The current implementation does not satisfy
  that semantic on any retried-load catalog (a real failure mode).
- `default_dataset` in `src/configs/datasets.py` points at VAP, so
  any agent or user invoking `deriva-ml-run` with no
  `datasets=` override gets the leaky family. The leak is silent —
  per-epoch `test_acc` numbers will look fine; they're just
  inflated by 30%+ of test images having been seen at training time.
- The fix (`row_per="Image"`, dedupe upstream) is a one-call site
  change in `_cifar10_datasets.py`. Persistence of the bug is a
  matter of nobody having filed it as such, not difficulty.

## Suggested fix

In `scripts/_cifar10_datasets.py`:

1. Change `row_per` from the feature table to `"Image"`.
2. Add an `include_tables` adjustment so the produced bag still
   carries the `Image_Classification` feature rows for downstream
   consumers (the harness needs them to label the loaders).
3. Add a `split_dataset` invariant: assert
   `len(train_image_rids) + len(test_image_rids) == len(parent_image_rids)`
   and `train_image_rids ∩ test_image_rids == set()` before
   returning. Catching this at split time would have caught the
   present bug immediately.

Detected during the 2026-05-28 e2e run with sibling versions:
deriva-ml v1.40.2, deriva-ml-mcp v0.5.9, deriva-mcp-core latest main,
deriva-skills v1.2.4, deriva-ml-skills v1.4.11.
