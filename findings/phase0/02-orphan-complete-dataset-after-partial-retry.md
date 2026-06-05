# Orphan `Complete` dataset (F2J) left after partial datasets-phase retry

**Persona:** Phase 0 (bootstrap / `load-cifar10` harness)
**Phase:** P0.7b — create dataset hierarchy (`--phase datasets`), after retry

## What happened

The `datasets` phase ran three times in P0.7b:

1. **First run** — aborted partway with the `Image.filename` KeyError
   (see finding 01). Before aborting, it had already created a
   `Complete` dataset, RID **F2J** (1100 members, tagged
   `Complete,Labeled`).
2. **Second run** — aborted immediately on the dirty-tree provenance
   guard (the fix for finding 01 was uncommitted). No new datasets.
3. **Third run** — succeeded, creating a *fresh* `Complete` dataset,
   RID **H8M**, and the full hierarchy beneath it (Split KDT, Training
   KE0, Testing KEA, etc., all deriving from H8M).

Result: the catalog now holds **two** `Complete,Labeled` datasets —
H8M (live, has the whole hierarchy as descendants) and F2J (orphan,
0 children, unreferenced).

## Root cause

The `datasets` phase is documented as "idempotent against partial
state," but its idempotency did not reuse the F2J `Complete` dataset
created by the partial first run. The likely reason: the first run
failed inside `split_dataset` *after* creating `Complete` but the
reuse path keys off execution-committed state, and the partial run's
execution never committed — so the third run saw no reusable Complete
and created a new one (H8M). The orphan is a side effect of
"abort mid-hierarchy, then re-run."

## Impact

Cosmetic only. The live hierarchy and `src/configs/datasets.py` both
point at H8M; F2J is unreferenced (0 children, no execution consumes
it). No persona workflow is affected. A Curator exploring the catalog
may legitimately notice "there are two Complete datasets" — which is a
realistic curation signal, not a blocker.

## Reproduction

1. Run `--phase datasets` against a catalog with images loaded.
2. Force it to abort after the `Complete` dataset is created but
   before the split completes (e.g., the finding-01 bug, or any
   selector error).
3. Fix the cause and re-run `--phase datasets`.
4. Observe two `Complete` datasets: the orphan from step 2 and the
   fresh one from step 3.

## Notes

- Left in place deliberately — deleting a dataset is a destructive op
  requiring explicit authorization, and the orphan is harmless.
- Possible hardening (separate from finding 01): make the `datasets`
  phase either (a) reuse an existing unreferenced `Complete` dataset
  on re-run, or (b) detect and warn about a pre-existing partial
  `Complete` before creating a new one.
