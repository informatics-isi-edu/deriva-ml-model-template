# `cifar10_cnn` silently skips Validation-typed bags and uploads a degenerate execution

**Persona:** Developer
**Phase:** Training arc — trying to honor Curator's recommendation to use `DAP` (Validation) for held-out evaluation, 2026-05-26
**Severity:** Medium (already tracked as pending task **D01**; this finding is the catalog-18 instance with the full reproduction trace)
**Component:** `deriva-ml-model-template/src/models/cifar10_cnn.py` (`_bag_role` + `_flatten_bags` + the no-train fallback path)

## What happened

The Curator's tk-003 handoff recommends `cifar10_validation_from_test` (RID
`DAP`, `Dataset_Type=[Validation, Labeled]`) as the held-out evaluator for
training runs. Following that advice via:

```
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run \
    +experiment=cifar10_quick datasets=cifar10_validation_from_test
```

succeeded as far as the Hydra wiring was concerned: the dataset bag downloaded,
the execution opened (RID `F40`), and the runner reached the
`load_cifar10_from_execution()` call. From there, the
runner reported:

```
Building DataLoaders from execution datasets...
WARNING: No training data found in execution datasets.
  Make sure your execution configuration includes CIFAR-10 datasets.
Committed 1 asset(s) to catalog:
  - deriva-ml/Execution_Asset: 1   (training_status.txt)
```

`F40` shows up in the catalog with `Status=Uploaded`, one asset
(`F5T` — `training_status.txt`, 50 bytes), and no other indication that the
run did nothing useful. To downstream tools (the Analyst), this looks exactly
like a normal completed training execution.

Root cause is in `_bag_role()` (line 50-64 of `src/models/cifar10_cnn.py`):
it recognizes only `training`, `testing`, `split`, and returns
`"unknown"` for everything else. The Curator-supplied `DAP` dataset has
type `["Validation", "Labeled"]`, so:

- `_bag_role(DAP) == "unknown"` → falls through in `_flatten_bags`.
- The `for bag in _flatten_bags(...)` loop in `load_cifar10_from_execution`
  never enters either the `training` or `testing` branch.
- Both `train_loader` and `test_loader` come back `None`.
- The training-mode branch in `cifar10_cnn()` (line 514) hits the
  `train_loader is None` guard, writes `training_status.txt`, returns.

Net effect: a 7-second execution (3 of which were bag download) that wrote a
zero-information row to the catalog and a 50-byte status file.

## Reproduction

Catalog 18 / `e2e-test-20260526`. Any dataset whose type does not contain
"Training" or "Testing" or "Split" will reproduce. The minimum repro on the
existing catalog is:

```
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run \
    +experiment=cifar10_quick datasets=cifar10_validation_from_test
```

Expected: the runner refuses (or remaps `Validation` to `testing` for
inference), and emits a non-zero exit status, OR a louder
warning that the execution is degenerate.

Actual: silent no-op, status=Uploaded, one 50-byte status file, no failure
signal anywhere except a single `WARNING:` line in stdout that's lost in the
HTTPS-noise above it.

## Impact on the persona's work

Routed around: the Developer used `cifar10_small_labeled_split` (CRR, type
`Labeled+Split`) instead of `DAP` for evaluation, accepting that test_acc is
computed against a 50-image in-pool subset (CSA) rather than the 250-image
held-out DAP that the Curator carefully prepared. **The Curator's careful
curation work is partially invisible to the Developer arc**, which is the
real cost: the Curator added DAP specifically to give the Developer/Analyst a
clean held-out evaluator, and the runner can't consume it.

A separate cost: F40 sits in the catalog as a normal-looking Uploaded
execution. If the Analyst sweeps `find_executions(status="Uploaded",
workflow_type="Training")` to compare runs, F40 will appear in the list and
make the comparison silently wrong. The Analyst should be told to skip F40
in the handoff (done here in the tk handoff).

## Suggested classification

Bug (silent degenerate execution) + Missing feature (Validation bag support).
Two separable fixes:

1. **`Validation` → `testing` role mapping** in `_bag_role()`, since the
   inference loop is identical (load images, run model, record predictions).
2. **Loud failure on no-train + no-test** in `load_cifar10_from_execution`:
   raise instead of returning `(None, None, _)` and writing a 50-byte
   status file. The current behavior is silent-fail-as-success, which is
   the worst kind for a provenance-tracking system.

Fix 1 unblocks the Curator's intent (DAP is consumable). Fix 2 makes the
runner self-policing for any future dataset-type drift.

## Notes for the fix-pass

- Code site: `src/models/cifar10_cnn.py:50-82` (`_bag_role` and
  `_flatten_bags`) and lines 514-526 (the no-train fallback).
- Test:`tests/test_configs_load.py` doesn't cover runner behavior. A new
  smoke test (`tests/test_runner_validation_bag.py`) that runs the model
  on a Validation-typed dataset in `dry_run` mode would catch this if the
  fallback path raised.
- Related pending task D01 already tracks the broader question — this
  finding is the catalog-18 reproduction with execution RIDs cited.
