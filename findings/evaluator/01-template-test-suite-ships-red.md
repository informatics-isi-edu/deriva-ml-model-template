# Template test suite ships red on the e2e branch — `cifar_canonical_partition` tests never updated for the filename-case fix

**Severity:** High
**Category:** Bug (template)
**Component:** `deriva-ml-model-template` — `tests/test_cifar10_datasets.py` + `src/scripts/_cifar10_datasets.py`
**Persona findings this consolidates:** `findings/phase0/01`, `findings/curator/03`

## What the evaluator found

`uv run python -m pytest tests/` on `e2e-test/2026-06-05` is **red**:

```
FAILED tests/test_cifar10_datasets.py::test_cifar_canonical_partition_splits_by_filename_prefix
FAILED tests/test_cifar10_datasets.py::test_cifar_canonical_partition_handles_all_train
FAILED tests/test_cifar10_datasets.py::test_cifar_canonical_partition_handles_all_test
3 failed, 65 passed, 2 skipped
```

All three fail with `KeyError: 'Image.Filename'`. Verified by the
evaluator by re-running the suite directly.

## Why this is the run's most actionable finding

The Phase-0 bootstrap hit a genuine template bug: `cifar_canonical_partition`
read the denormalized column `Image.filename` (lowercase) but deriva-ml's
denormalization (reworked in deriva-ml #283 / v1.45.0) produces
`Image.Filename` (catalog column case). The runtime fix (commit `65ae86b`)
corrected the **selector source** but left the **three unit tests that
guard that selector** still feeding lowercase-keyed fixture DataFrames.
The result is the classic "fix the code, forget the test" half-fix:

- The runtime path is correct (the live bootstrap produced a verifiably
  correct KE0/KEA partition — the evaluator confirmed disjointness and
  exact 550/550 coverage directly).
- But the suite is red, and the three tests that exist *specifically* to
  protect this selector now error before reaching their assertions —
  they are guarding nothing. A user who fixed the runtime bug and trusted
  a green suite would ship a red one.

This is a verifiable, deterministic defect that the next run (and any
user who clones the template at this commit) will hit on the first
`pytest`. It is the single most concrete thing to fix.

## Severity rationale

High, not Medium: a primary template invariant (a green test suite) is
broken on the shipped branch, and the broken tests are exactly the
regression guard for a bug that already bit this run once. It did not
block the personas (they routed around it), so it is not a Blocker —
but it compromises the template's "clone and `pytest`" deliverable.

## Fix (mechanical, belongs with the original case fix as a `main` cherry-pick — NOT `[E2E-DROP]`)

- `tests/test_cifar10_datasets.py` lines ~43/64/74: `"Image.filename"`
  → `"Image.Filename"` in the three fixture DataFrames.
- `tests/test_cifar10_datasets.py:34`: stale docstring "reads
  ``Image.filename``" → `Image.Filename`.
- `src/scripts/_cifar10_datasets.py:426`: stale comment "Image.filename
  for the predicate to inspect" → `Image.Filename`.

## Optional defensive hardening (separate, lower priority)

`split_dataset`'s `selection_fn` path does not validate the columns the
selector reads (the stratified path already emits a friendly "column X
not in denormalized df; available: [...]" error at split.py:337-340).
Extending that guard to the `selection_fn` path would have turned this
`KeyError` into a self-explaining message. This is a deriva-ml change,
not a template change.

## Reproduction

```
git checkout e2e-test/2026-06-05
DERIVA_ML_ALLOW_DIRTY=true uv run python -m pytest tests/test_cifar10_datasets.py -q
# -> 3 failed, KeyError: 'Image.Filename'
```
