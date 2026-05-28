# `src/configs/datasets.py` descriptions for TCC/VAP families overstate sizes vs catalog actuals

**Persona:** Evaluator
**Severity:** Low
**Category:** Doc gap
**Phase:** Cross-arc synthesis (Curator-02 noticed; not separately filed)

## What happened

The `with_description(...)` strings on the `cifar10_labeled_*` and
`cifar10_small_labeled_*` configs in `src/configs/datasets.py`
advertise sample counts that don't match what `load-cifar10`
actually produces on this catalog (or, per Curator finding 02, on
any catalog where the feature-row-vs-image-row mechanics apply):

| Config | Advertised in `with_description` | Actual on catalog 27 |
|---|---|---|
| `cifar10_labeled_split` (TCC) | "stratified 80/20 from training (440/110, seed=42)" | 361 / 105 |
| `cifar10_labeled_training` (TCM) | "Training subset (440)" | 361 |
| `cifar10_labeled_testing` (TCY) | "Testing subset (110)" | 105 |
| `cifar10_small_labeled_split` (VAP) | "stratified 400/100 from training (seed=42)" | 339 / 95 |
| `cifar10_small_labeled_training` (VAY) | "Training subset (400)" | 339 |
| `cifar10_small_labeled_testing` (VB8) | "Testing subset (100)" | 95 |

A `deriva-ml-run --info` reader expects 440 training images and
gets 361; an evaluator config that hard-codes "expected DataLoader
length = 440" silently runs with a smaller pool. The Curator
called out the discrepancy as part of `findings/curator/02` (the
leakage finding) but the **config descriptions themselves are
template artifacts** that travel with `main`, not catalog-specific
state. A fresh-clone user who hits this gets the wrong picture
even before they hit the leakage.

## Reproduction

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e
uv run deriva-ml-run --info 2>&1 | grep -A 1 "labeled_split\|labeled_training\|labeled_testing"
```

versus

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id=27)
ds = ml.catalog.getPathBuilder().schemas["e2e-test-20260528"]
di = ds.Dataset_Image
for rid in ["TCM","TCY","VAY","VB8"]:
    n = len({r["Image"] for r in di.filter(di.Dataset==rid).entities().fetch()})
    print(f"{rid}: {n}")
```

## Suggested fix

Two reasonable options, paired with `findings/evaluator/02`:

1. **If the loader / `split_dataset` fix in evaluator/02 lands** —
   restoring 440/110 and 400/100 as actual counts — leave the
   descriptions alone; they'll match after the fix.
2. **If the fix is deferred** — update the descriptions to match
   actual counts and add a note "(actual size depends on per-image
   feature-row count; see evaluator/02)".

Don't both fix the upstream cause and rewrite the descriptions;
the upstream fix is the durable answer and the descriptions are
correct under that fix.

Detected during the 2026-05-28 e2e run with sibling versions:
deriva-ml v1.40.2, deriva-ml-mcp v0.5.9, deriva-mcp-core latest main,
deriva-skills v1.2.4, deriva-ml-skills v1.4.11.
