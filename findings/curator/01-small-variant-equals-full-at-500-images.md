# Small Toronto-variant datasets are byte-identical to the full Toronto datasets at `--num-images 500`

**Persona:** Curator
**Phase:** Audit of bootstrapped catalog 18 (e2e-test-20260526), 2026-05-26
**Severity:** Medium
**Component:** `deriva-ml-model-template/src/scripts/_cifar10_datasets.py` (load-cifar10 datasets phase)

## What happened

While auditing the 13 datasets `load-cifar10` produced on catalog 18, the
Curator confirmed via direct set comparison that:

```text
970 (cifar10_training)        == 982 (cifar10_small_training)        ✓
97A (cifar10_testing)         == 98C (cifar10_small_testing)         ✓
```

i.e. the `Small_Training` / `Small_Testing` datasets contain the *exact
same Image RIDs* as their full-size siblings — not a subset, not a
sample, not a stratified pick. They're separate dataset rows pointing
at the same 250 + 250 images.

Reproduction (against catalog 18):

```python
from deriva_ml import DerivaML
ml = DerivaML('localhost', '18')
t970 = {m['RID'] for m in ml.lookup_dataset('970').list_dataset_members()['Image']}
s982 = {m['RID'] for m in ml.lookup_dataset('982').list_dataset_members()['Image']}
assert t970 == s982   # holds at --num-images 500
```

The cause is structural, not data-dependent: the loader's
`_cifar10_datasets.py` always tries to draw `SMALL_TRAIN_SIZE = 500` and
`SMALL_TEST_SIZE = 500` random samples regardless of how many train/test
images actually exist. When the source pool is smaller than the target
sample size (the bootstrap loaded 250 + 250), the `if len(train_rids)
>= SMALL_TRAIN_SIZE` branch in `_cifar10_datasets.py:294` falls through
to `sample = train_rids` (use everything). Same for test.

## Reproduction

1. Bootstrap a fresh catalog with `--num-images 500` (or anything
   `< 1000`):
   ```
   uv run load-cifar10 --hostname localhost --create-catalog test-N --phase schema
   uv run load-cifar10 --hostname localhost --catalog-id N --num-images 500 --phase images
   uv run load-cifar10 --hostname localhost --catalog-id N --num-images 500 --phase datasets
   ```
2. After datasets phase, check member equality between cifar10_training
   (Toronto train) and cifar10_small_training — they will be set-equal.

## Impact on the persona's work

Moderate. The Curator's audit had to flag this so the Developer doesn't
assume the `*_small_*` datasets are an actual smaller variant.

Two downstream consequences:

1. **Misleading config names.** `cifar10_small_training` resolves to the
   same data as `cifar10_training` — running experiments on "small" gives
   the same numbers as "full," which silently invalidates any
   `small_vs_full` comparison the Developer might attempt.
2. **Dataset-row proliferation.** Two separate dataset RIDs (982, 98C)
   carry identical content. Downstream bag materialization, lineage
   queries, and version bumps all double up.

Routed around by: creating the `cifar10_balanced_demo` (DB0) curated set
as a *genuinely* small (50-image) hand-picked sample, so the persona
arcs have a real "smaller than full" option for smoke runs.

## Suggested classification

Bug (loader behavior) **OR** Polish (description should reflect that
"small" is degenerate at this `--num-images`). The cleanest fix is
probably to skip creating `Small_Training` / `Small_Testing` entirely
when the source pool is smaller than `SMALL_TRAIN_SIZE` — emit a log
line ("skipping small variants; source has only N images") instead of
silent duplication.

This is related to but distinct from pending task **C03** (dataset
descriptions reference hard-coded "50,000 / 10,000 / 1,000" image
counts regardless of `--num-images`). C03 is the description-text
problem; this finding is the *structural* duplication.

## Notes for the fix-pass

- Code site: `src/scripts/_cifar10_datasets.py:284-308`, the
  `_batched_add(datasets["small_training"], ...)` block.
- A one-line guard (`if len(train_rids) < SMALL_TRAIN_SIZE * 0.9: skip`)
  would solve the small-but-not-tiny case. A more honest fix is to
  parameterize the small-variant fraction (e.g. always 1/5 of source)
  so it's well-defined at any pool size.
- Touches the same code as C03 — fix together.
