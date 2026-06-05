# `cifar_canonical_partition` tests still use lowercase `Image.filename` after the case fix — 3 tests red

**Persona:** Curator
**Phase:** Sanity-running the template test suite during catalog characterization

## What happened

Running the template's own test suite on the e2e branch fails:

```
DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/ -q
...
FAILED tests/test_cifar10_datasets.py::test_cifar_canonical_partition_splits_by_filename_prefix
FAILED tests/test_cifar10_datasets.py::test_cifar_canonical_partition_handles_all_train
FAILED tests/test_cifar10_datasets.py::test_cifar_canonical_partition_handles_all_test
3 failed, 53 passed, 2 skipped
```

All three fail with:

```
KeyError: 'Image.Filename'
```

## Root cause

The Phase-0 filename-case fix (`findings/phase0/01`, commit `65ae86b`
"fix(cifar): cifar_canonical_partition reads Image.Filename (catalog
case)") corrected the **source** selector to read the catalog-cased
column `Image.Filename` (capital F), and updated the source docstring
example. But it **did not update the tests**, which still construct their
fixture DataFrames with the old lowercase column name:

- `tests/test_cifar10_datasets.py:43` — `"Image.filename": [...]`
- `tests/test_cifar10_datasets.py:64` — `pd.DataFrame({"Image.filename": [...]})`
- `tests/test_cifar10_datasets.py:74` — `pd.DataFrame({"Image.filename": [...]})`
- `tests/test_cifar10_datasets.py:34` — stale docstring: "The selector
  reads ``Image.filename`` from the denormalized dataframe"

So the fixed source does `df["Image.Filename"]` while the test feeds a
DataFrame whose only column is `Image.filename` → `KeyError`.

There is also a now-stale lowercase reference in the **source** that the
fix missed:

- `src/scripts/_cifar10_datasets.py:426` — comment still says
  "Image.filename for the predicate to inspect."

## Why this matters

The case fix was correct for the *runtime* path — the live bootstrap
(`load-cifar10 --phase datasets`) succeeded against catalog 69 with the
fixed source, and my independent set-arithmetic verification confirms
the canonical KE0/KEA partition is correct. But the **fix was only
half-applied**: source updated, tests not. The unit tests that exist
specifically to guard this selector are now testing nothing (they error
before reaching their assertions), and `uv run python -m pytest tests/`
is red on the branch. A user who fixed the runtime bug and trusted a
green test suite would have shipped a red one.

This is the classic "fix the code, forget the test" gap — and it is
exactly the kind of thing that surfaces when you actually run the suite
rather than assuming the runtime success implies test health.

## Reproduction

1. On the `e2e-test/2026-06-05` branch (which carries commit `65ae86b`).
2. `DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/test_cifar10_datasets.py -q`
3. Observe 3 failures, all `KeyError: 'Image.Filename'`.

## Notes

- Not fixed mid-arc (Curator no-fix-during-run rule). Routed around it:
  my own verification (`scripts/curator_verify_splits.py`) does not
  depend on these tests and passes all 19 checks.
- The fix is mechanical and belongs with the original case fix as a
  cherry-pick to `main` (genuine template improvement, not `[E2E-DROP]`):
  change the three test fixtures and the test docstring from
  `Image.filename` → `Image.Filename`, and update the stale source
  comment at `_cifar10_datasets.py:426`.
- The other 53 tests pass (2 skipped), including the config smoke tests
  and the `_require_small_variant_distinct` regression coverage — so the
  redness is isolated to this one selector's three tests.
