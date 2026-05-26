# Bootstrap dataset descriptions hard-code "50,000 / 10,000 / 1,000" image counts that don't match `--num-images`

**Persona:** Curator
**Phase:** Audit, 2026-05-26 (catalog 18, `--num-images 500`)
**Severity:** Low (Polish)
**Component:** `deriva-ml-model-template/src/scripts/_cifar10_datasets.py`

## What happened

Audit revealed that `load-cifar10`'s built-in dataset descriptions are
hard-coded to reference the full CIFAR-10 sizes regardless of what
`--num-images` actually used:

| RID | Description | Actual member count (`--num-images 500`) |
|-----|-------------|-----------|
| 970 | "CIFAR-10 training set with 50,000 labeled images" | 250 |
| 97A | "CIFAR-10 testing set with 10,000 labeled images" | 250 |
| 97T | "Small CIFAR-10 dataset split with 1,000 randomly selected images for testing" | (parent of 982 + 98C, each 250) |
| 982 | "Small CIFAR-10 training set with 500 labeled images for quick testing" | 250 |
| 98C | "Small CIFAR-10 testing set with 500 labeled images for quick testing" | 250 |

A user browsing the catalog via Chaise or `deriva_ml_list_datasets` and
reading these descriptions would expect ~50,000 training images and
~10,000 test images. Actual: 250 each.

## Reproduction

Same bootstrap as in finding 01; then `deriva_ml_list_datasets` against
the catalog and read the `description` column.

## Impact on the persona's work

Polish. The Curator's `tacit-knowledge.md` Curator entries already
record the right counts (cross-channel-verified), so downstream
personas in this run aren't misled. But:

- For an external reader landing on the catalog without `tacit-knowledge.md`
  context, the descriptions are misleading.
- The Curator's `datasets.py` entries had to *intentionally* replace the
  loader's descriptions with truthful ones (see the
  `cifar10_validation_from_test` and `cifar10_balanced_demo` blocks).

## Suggested classification

Polish (cosmetic in dev catalogs, sloppy in shared ones). Already
tracked as pending task **C03** ("dataset descriptions adapt to
`--num-images`"). This finding is the catalog-18-specific instance.

## Notes for the fix-pass

- Code site: `src/scripts/_cifar10_datasets.py` — the `description=`
  arguments to each `exe.create_dataset(...)` call carry the
  hard-coded strings.
- Fix template: substitute the actual partition size: e.g.
  `f"CIFAR-10 training set with {len(train_rids)} labeled images"`.
- The `Small_*` descriptions need an extra branch — when the source
  pool is smaller than the target small size (finding 01), the "500
  labeled images for quick testing" claim is doubly wrong.
- Fix together with finding 01 — both are in
  `_cifar10_datasets.py:158-280`.
