# Bootstrap doc prescribes --num-images 500 but loader requires >= 1002

**Persona:** Phase 0 (Bootstrap)
**Phase:** Step 7 — load datasets

## What happened

Bootstrap doc `docs/test-plans/e2e-bootstrap.md` step 7 prescribes:

```
uv run load-cifar10 --hostname localhost \
    --catalog-id <new_id> --num-images 500 --phase images
uv run load-cifar10 --hostname localhost \
    --catalog-id <new_id> --num-images 500 --phase datasets
```

The images phase succeeded at `--num-images 500` (250 train / 250 test).
The datasets phase failed:

```
scripts._cifar10_datasets.SmallVariantDegenerateError: At this catalog size
(train_pool=250, test_pool=250) the 'small' Toronto split family would be
byte-identical to the full Toronto split. SMALL_TRAIN_SIZE=500 and
SMALL_TEST_SIZE=500 require strictly larger source pools to yield a distinct
sample. Re-run load-cifar10 with --num-images >= 1002 so each partition
exceeds the small-variant sample size, or skip the small Toronto split and
use the labeled-split family instead — split_dataset() partitions the
training images directly and stays distinct at any catalog size.
```

The loader is doing the right thing — it correctly refuses to produce a
byte-identical "small" variant. The bug is in the documented invocation:
500 images can never produce the small Toronto split family because the
small-variant constants (`SMALL_TRAIN_SIZE=500`, `SMALL_TEST_SIZE=500`) need
strictly *larger* source pools to draw distinct samples from.

## Reproduction

```
cd .../deriva-ml-model-template-e2e
uv run load-cifar10 --hostname localhost --create-catalog <name> --phase schema
uv run load-cifar10 --hostname localhost --catalog-id <id> --num-images 500 --phase images
uv run load-cifar10 --hostname localhost --catalog-id <id> --num-images 500 --phase datasets
# -> SmallVariantDegenerateError
```

## Notes

Worked around by bumping `--num-images` to 1100 (a comfortable margin above
the 1002 minimum). The bootstrap doc should either:

- update the prescribed invocation to `--num-images 1100` (or another
  comfortably-above-1002 value), or
- explicitly call out the floor and the trade-off (more images → longer
  load, but the small-Toronto-split family becomes meaningful).

A second-order question: are the `SMALL_*_SIZE=500` constants still the
right floor? They were presumably calibrated against a different historical
target catalog size. If the e2e run is the primary fitness vehicle, the
constants and the bootstrap invocation should agree.

Detected during the 2026-05-28 e2e run with sibling versions:
deriva-ml v1.40.2, deriva-ml-mcp v0.5.9, deriva-mcp-core latest main,
deriva-skills v1.2.4, deriva-ml-skills v1.4.11.
