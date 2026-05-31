# Labeled-split datasets are derived from F2T but not registered as its catalog children

**Persona:** Curator
**Phase:** Substrate characterization (dataset-hierarchy walk on catalog 168)

## What happened

**Attempted:** map the full dataset nesting hierarchy so the
Modeler/Analyst can navigate parent/child relationships and walk
provenance with `deriva_ml_get_lineage` / `deriva_ml_list_dataset_relations`.

**Expected:** datasets derived from the training partition `F2T` (the
labeled splits, whose descriptions say "subset of F2T ... stratified by
Image_Class.Name, seed=42") would appear as descendants of `F2T` — or at
least be reachable from it via the hierarchy.

**Actual:** the catalog records only two split roots:

- `F2J` (`cifar10_split`) → children `F2T`, `F34`
- `F3M` (`cifar10_small_split`) → children `F3W`, `F46`

The two labeled-split roots are **standalone** — no parent, and not
children of `F2T`:

- `NE0` (`cifar10_labeled_split`) → children `NE8`, `NEJ`
- `PHJ` (`cifar10_small_labeled_split`) → children `PHT`, `PJ4`

Yet a set-membership audit (see finding 01) shows `NE8`/`NEJ`/`PHT`/`PJ4`
are 100% derived from `F2T`. The only catalog-side record of that
derivation is the free-text dataset *description*, which is advisory prose
— not a walkable lineage edge.

Concretely: `deriva_ml_list_dataset_relations(dataset_rid="F2T",
direction="both", recurse=true)` returns parent `F2J` and **no children**.
The training-pool-derived labeled splits are invisible to that walk.

## Reproduction

Read-only. Against localhost catalog 168:

1. `deriva_ml_list_dataset_relations(hostname="localhost",
   catalog_id="168", dataset_rid="F2T", direction="both", recurse=true)`
   → `{"parents": [F2J], "children": []}`.
2. `deriva_ml_list_dataset_relations(... dataset_rid="NE0" ...)`
   → `{"parents": [], "children": [NE8, NEJ]}` (NE0 is a root, not a child
   of F2T).
3. Confirm the derivation the hierarchy omits: every member of `NE8`,
   `NEJ`, `PHT`, `PJ4` is in `F2T` (set-intersection via the membership
   query in finding 01).

## Notes

- The practical impact: "what was derived from the training partition?"
  is **not answerable from the hierarchy or from `get_lineage`** — it
  lives in description prose only. A consumer has to read descriptions or
  re-derive by set intersection (as this audit did).
- This looks like a gap in the bootstrap loader's split-registration:
  when `split_dataset()` (or the loader's equivalent) created the labeled
  splits *from* `F2T`, it did not register the resulting split root as a
  child of `F2T`. The canonical splits (`F2J`→F2T/F34, `F3M`→F3W/F46) *do*
  register parent→child, so the registration mechanism works; the
  labeled-split path appears not to use it.
- I did not attempt to add the missing parent→child edges. That would be
  a catalog mutation (adding `Dataset` members to a parent dataset, which
  flips its version) and is out of scope for a read-oriented curation arc;
  it is also a loader-behavior question better answered by whoever owns the
  bootstrap path. Routed around it by recording the true derivation in
  `tacit-knowledge.md` `tk-002`/`tk-003` so downstream readers have the
  lineage even though the catalog doesn't expose it as an edge.
