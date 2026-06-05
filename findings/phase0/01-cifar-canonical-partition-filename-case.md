# `cifar_canonical_partition` reads `Image.filename` but denormalized column is `Image.Filename`

**Persona:** Phase 0 (bootstrap / `load-cifar10` harness)
**Phase:** P0.7b — create dataset hierarchy (`--phase datasets`)

## What happened

`uv run load-cifar10 --hostname localhost --catalog-id 69 --num-images 1100 --phase datasets`
aborted with:

```
KeyError: 'Image.filename'
  at src/scripts/_cifar10_datasets.py:303, in cifar_canonical_partition
    is_train = df["Image.filename"].str.startswith("train_")
  called from split_dataset(..., selection_fn=cifar_canonical_partition,
                            element_table="Image", include_tables=["Image"], ...)
  → deriva_ml/dataset/split.py:_compute_partitions → selector(df, ...)
```

The catalog was left in a partial state: only the `Complete` dataset
(RID `F2J`, 1100 images) was created. The `Split` / `Training` /
`Testing` / `Small_*` / `*_Labeled_*` hierarchy was never written
because `split_dataset` raised before any of those catalog writes.

## Root cause

Case mismatch between the column the selector reads and the column
deriva-ml's denormalization actually produces.

- `cifar_canonical_partition` reads `df["Image.filename"]` (lowercase
  `f`) at `src/scripts/_cifar10_datasets.py:303`, and its docstring
  example (lines 292-301) likewise uses `"Image.filename"`.
- `Dataset.get_denormalized_as_dataframe(["Image"])` produces
  `Image.Filename` (capital `F`) — verified empirically against
  catalog 69:
  ```
  ['Image.RID','Image.URL','Image.Filename','Image.Description','Image.Length','Image.MD5']
  ```
- The actual catalog column is `Filename` (capital F) — verified
  against the live Image table schema. So deriva-ml's denormalized
  column name correctly mirrors the canonical catalog column case.

**deriva-ml is behaving correctly; the harness script has the bug.**
The denormalized DataFrame column namespace mirrors *catalog column
case* (`Filename`), whereas the asset *domain-object* attribute is
lowercase (`a.filename`, used correctly elsewhere in the same file at
lines 356-357). The selector confused the two namespaces — it applied
the Python-attribute case to a DataFrame column.

## Why it surfaced now

deriva-ml's denormalization wide-table column production was reworked
in `feat(denormalize): opt-in system columns (RCB/RCT/RMT/RMB) in wide
tables` (#283, commit 5158d8f9, shipped in v1.45.0). The selector's
hard-coded lowercase `Image.filename` predates / didn't track that
change. The e2e env is pinned at deriva-ml 4d56677d (one docs-only
commit past v1.45.0).

## Reproduction

1. Fresh catalog with the cifar10 schema + 1100 images loaded
   (`--phase schema` then `--phase images`).
2. `uv run load-cifar10 --hostname localhost --catalog-id <id> --num-images 1100 --phase datasets`
3. Observe the `KeyError: 'Image.filename'` traceback.

## Fix

One-character correction in `src/scripts/_cifar10_datasets.py`:
`df["Image.filename"]` → `df["Image.Filename"]` in the selector body
(line 303), plus the matching docstring example (lines 292-301). This
is a genuine template improvement (cherry-pick to `main`, not an
`[E2E-DROP]`).

## Notes

- The `datasets` phase is documented as idempotent against partial
  state, so re-running after the fix is safe (the existing `Complete`
  dataset `F2J` is reused, not duplicated).
- Defensive hardening worth considering separately: the selector could
  resolve the column case-insensitively, or `split_dataset` could
  raise a friendlier "selector read column X not in denormalized df;
  available: [...]" error (the stratified path already does this at
  split.py:337-340 — the `selection_fn` path does not, because the
  selector's read set is opaque to `split_dataset`).
