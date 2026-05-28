# Curator Summary — 2026-05-28 e2e run

**Persona:** Curator | **Catalog:** `localhost` id 27 (`e2e-test-20260528`) |
**Worktree branch:** `e2e-test/2026-05-28`

## What I did

Inspected the 13 datasets, the `Image_Classification` feature, the
`Dataset_Type` and `Image_Class` vocabularies, and the
dataset-membership graph for all 1100 Image rows. Cross-checked counts,
advertised vs actual split sizes, class balance, and disjointness of
every train/test pair.

## What I decided

Two findings filed, three tacit-knowledge entries written. I did
**not** create a new "clean" dataset — the leakage is well-
characterised and the Toronto family (M16 × M1G) is already wired
into `datasets.py` as a clean alternative. Filing the findings rather
than rebuilding splits is the right call for an e2e fitness run;
fixing `_cifar10_datasets.py` belongs in a fix-pass.

## What surprised me

1. **Two ground-truth executions, not one.** The first failed loader
   attempt (`854`, 500 rows) and the successful retry (`HSR`, 1100
   rows) both wrote `Image_Classification` feature values. Classes
   agree where they overlap, but feature-row counts are inflated by
   ~45%. Filed as `findings/curator/01-...`.
2. **TCC and VAP leak across train/test.** 33 images appear in both
   TCM and TCY; 24 in both VAY and VB8. 100% of overlapping images
   are exactly the doubly-tagged ones from finding 01 — the loader's
   `split_dataset(row_per=feature_table)` partitions feature rows,
   not images. Advertised sizes (440/110, 400/100) also don't match
   catalog actuals (361/105, 339/95). Filed as
   `findings/curator/02-...`.
3. Class balance is otherwise uniform across every partition — the
   downstream concern is *which split*, not *which class* (tk-003).

## What I left for the next persona

- **Tacit knowledge `tk-001`/`tk-002`/`tk-003`** in
  `tacit-knowledge.md` — the dual-write shape of
  `Image_Classification`, the leakage in TCC/VAP, and the
  uniform-class-balance property.
- **Two findings** under `findings/curator/`.
- **No catalog mutations**; the 13 Phase-0 datasets are unchanged.
- **Pragmatic recommendation** (in tk-002, not a directive): for
  ground-truth evaluation work, the Toronto family
  (`cifar10_training` M16 × `cifar10_testing` M1G, 55/class, zero
  overlap) is the clean pick. `cifar10_small_labeled_split` (VAP)
  is fine for smoke-testing but not for accuracy claims.
