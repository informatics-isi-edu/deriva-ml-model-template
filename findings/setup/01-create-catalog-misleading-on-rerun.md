# load-cifar10 --create-catalog message is misleading on idempotent re-run

**Persona:** Setup (Phase 0)
**Phase:** P0 step 5 part A — create catalog
**Severity:** Low (cosmetic / orienting)
**Component:** `src/scripts/load_cifar10.py` in deriva-ml-model-template

## What happened

First invocation:
```
uv run load-cifar10 --hostname localhost \
    --create-catalog e2e-test-20260526 --phase schema
```
Output ended with:
```
============================================================
  SCHEMA PHASE COMPLETE
  Re-run with --phase images or --phase datasets.
============================================================
```
No catalog id was printed in the body of the run (only the "complete"
banner showed).

Second (intended-idempotent) invocation of the **same command** to
recover the id:
```
============================================================
  CREATED NEW CATALOG
  Catalog ID:  18
============================================================
```
The wording suggests a new catalog was created, but verification via
the alias registry shows the same id 18 was reused (the
`create_or_retarget_ml_catalog` helper from setup/06 worked correctly —
it retargeted the existing alias, not created a duplicate).

## Reproduction

```bash
uv run load-cifar10 --hostname localhost --create-catalog <name> --phase schema
# then re-run the exact same line
uv run load-cifar10 --hostname localhost --create-catalog <name> --phase schema
```

The first run does not surface the new id; the second run claims
"CREATED NEW CATALOG" when it actually retargeted.

## Impact on the persona's work

Forced an extra cross-channel verification step (querying
`/ermrest/alias/<name>` directly) to be sure no duplicate catalog
was created — wasted ~30 seconds and required reaching for the
direct channel earlier than the test plan expects.

## Suggested classification

Polish. The retargeting behavior is correct; only the human-facing
message is wrong on the second invocation. Two cheap fixes:
- First-run banner should always echo the new id (so re-runs aren't
  needed to recover it).
- Idempotent-reuse path should say "REUSING EXISTING CATALOG (alias retargeted)"
  rather than "CREATED NEW CATALOG."

## Notes for the fix-pass

Touch points likely live in `src/scripts/load_cifar10.py`'s
schema-phase output and/or `create_or_retarget_ml_catalog`'s return
value handling.
