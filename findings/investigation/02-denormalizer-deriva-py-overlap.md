# deriva-py overlap and the "should the denormalizer move?" question

**Investigator:** denormalizer audit (research-only)
**Date:** 2026-05-28
**Sources read:**
- `/Users/carl/GitHub/deriva-py/deriva/core/datapath.py`
- `/Users/carl/GitHub/deriva-py/deriva/core/ermrest_catalog.py`
- `/Users/carl/GitHub/deriva-py/deriva/bag/path_walker.py`
- `/Users/carl/GitHub/deriva-py/deriva/bag/catalog_builder.py`
- `/Users/carl/GitHub/deriva-py/deriva/bag/catalog_loader.py`

## What deriva-py already provides

The shared FK-graph primitive **already lives in deriva-py.**
`deriva.bag.path_walker.SchemaPathWalker` is "the **shared
primitive** for 'walk the foreign-key graph from a root table,
recording paths, honoring scope and depth limits, and treating
vocabularies as leaves'" (path_walker.py docstring lines 1-44).
It's the extracted core that:

- `deriva.bag.catalog_builder.CatalogBagBuilder` uses for bag
  export (BFS from one or more anchors, multi-path routes, capped
  by `max_paths`).
- `deriva_ml.model.denormalize_planner.DenormalizePlanner._schema_to_paths`
  uses for denormalize planning (exhaustive DFS from a single
  root, emitting every prefix).

Both consumers want "all valid FK chains from the root through
the schema graph"; they differ in BFS-shortest vs. every-prefix.

The walker explicitly **excludes** domain-specific transparency
rules (path_walker.py lines 25-37):

> What this module deliberately does **not** carry:
> * No anchor-RID filtering...
> * No domain-specific transparency rules (e.g. DerivaML's
>   "feature-association tables are transparent bridges"). Those
>   belong in the consumer because they reference domain concepts
>   (Execution) that have no place in a generic walker.
> * No vocab-export choice, asset mode...

So the layering is already correct: the *graph traversal* is
generic in deriva-py; the *domain semantics* is in deriva-ml.

## What deriva-py does NOT provide

- **No denormalize / wide-table abstraction.** `DataPathBuilder`
  (`deriva/core/datapath.py:245-688`) is a row-fetching DSL.
  `DataPath.link(other_table, on=...)` extends a path; `entities()`
  fetches whole rows from the current path context. There's no
  shape API that says "give me a wide table with columns from
  these N tables on each row, where 'row' is defined by table T."
  The DataPath model is "I am building a specific JOIN; let me
  specify each link by hand."
- **No `row_per` / sink-finding / Rule 5 / Rule 6 / orphan-row
  emission.** Path-walker tells you what tables are reachable;
  whether a particular join chain produces a sensible wide-table
  shape is up to the caller.
- **No feature-association awareness.** Path-walker has no
  concept of "feature" — that's a deriva-ml abstraction layered
  on top of generic FK tables.
- **No feature-name shorthand resolver.**
  `Denormalizer._resolve_table_names` translates feature names to
  feature-association table names; there's no analogue in
  deriva-py.
- **No anchor / dataset-membership filter.** `DataPath.filter(...)`
  takes ad-hoc predicates; the dataset-membership-as-anchor
  pattern (Rule 7, orphan rows from upstream anchors) is
  deriva-ml-specific.

`CatalogBagBuilder` (`deriva/bag/catalog_builder.py:74-...`)
walks FK paths to decide which **tables** to include in a bag and
emits CSV per table. It's table-shaped output; it's never
asked the wide-table question.

## What the denormalizer adds on top of `DataPathBuilder`

If you tried to express `Denormalizer.as_dataframe(include_tables,
row_per=...)` as a `DataPathBuilder` query, what would the
gap be?

1. **Sink-finding / `row_per` auto-inference** (Rule 2). The
   caller of `DataPathBuilder` has to know which table to
   `link()` first. There's no API that says "given these N
   tables, which is the deepest sink?"
2. **Diamond / multi-path detection** (Rule 6). `DataPath.link()`
   raises if you ask for an ambiguous FK direction
   (`datapath.py:444`: `'%s is not an inbound or outbound foreign
   key for the path\'s context'`), but it doesn't *enumerate*
   the alternative paths and ask the user to pick.
   `Denormalizer` does — via `_find_path_ambiguities` and the
   `DerivaMLDenormalizeAmbiguousPath` exception with a
   `suggested_intermediates` list.
3. **Downstream-leaf rejection** (Rule 5). `DataPathBuilder` will
   happily build `Image.link(Execution_Image_Image_Classification)`
   and produce one row per feature observation; it doesn't
   refuse the request because there's no claim being made about
   "wide table with one row per Image." The Rule 5 refusal is a
   wide-table-shape rule, not a join-correctness rule.
4. **Transparent intermediates / feature-assoc transparency.**
   `DataPathBuilder` makes every join explicit; the caller has to
   include the association table. `Denormalizer` makes 2-FK
   topological associations and 3-FK feature-association tables
   *transparent* — the caller asks for the two endpoints, the
   planner inserts the bridge join automatically, the bridge's
   columns are dropped from the output (unless the caller named
   the bridge in `include_tables`). This is the
   `_is_transparent_intermediate` family of predicates plus the
   bidirectional bridge hop in `_outbound_reachable`.
5. **Anchor classification (Rule 7).** Dataset members of mixed
   types each contribute differently: members at `row_per` →
   one row each; members upstream → either filter-only or orphan
   row depending on reachability. `DataPathBuilder` has no
   anchor model — anchors come in as `filter()` predicates.
6. **LEFT-JOIN orphan emission.** The orphan-row machinery in
   `Denormalizer._emit_orphan_rows` produces NULL-`row_per` rows
   for upstream anchors that don't reach a `row_per` row. To
   emulate this with `DataPathBuilder` you'd need to issue two
   queries and union them yourself.
7. **Local-SQLite cache + bag/catalog dual-source.** This is the
   §3 state-ownership model in the spec — the local engine
   accumulates rows from prior fetches and serves both bag and
   catalog paths. `DataPathBuilder` always talks to the server.
8. **Wide-table column naming.** `Table.column` /
   `schema.Table.column` dotted column names with multi-schema
   detection. `DataPathBuilder` returns raw `(table, column)`
   pairs from `entities()`.

## What a port to deriva-py would look like

A natural API in deriva-py, given the existing path-builder
shape:

```python
# deriva-py: catalog-level method (no DerivaML required)
pb = catalog.getPathBuilder()
wide = pb.denormalize(
    include_tables=[("e2e-test-20260528", "Image"),
                    ("e2e-test-20260528", "Image_Class")],
    row_per=("e2e-test-20260528", "Image"),
    via=[...],
    anchors=[("Image", "1-ABCD"), ...],   # or DataPath filter
    transparent=lambda tbl: my_predicate(tbl),  # extension hook
)
df = wide.as_dataframe()
```

Two things this version would have to be honest about that
deriva-ml's version currently hides:

1. **Schema-qualified table names.** deriva-ml has a single
   domain schema convention so it can drop the schema prefix.
   deriva-py is cross-schema by design and would need
   `(schema, table)` tuples.
2. **Generic transparency rule.** Without DerivaML's
   `Execution` table, the 3-FK feature-association predicate
   doesn't make sense. The deriva-py version would have a
   pluggable `transparent: Callable[[Table], bool]` hook (just
   like `SchemaPathWalker` has `edge_filter`), and deriva-ml
   would wire its `_is_transparent_intermediate` predicate in
   via that hook.

The deriva-ml-specific stays in deriva-ml:

- The feature-name shorthand resolver (`Image_Classification` →
  `Execution_Image_Image_Classification`) is a feature/Execution
  concept — stays in deriva-ml.
- Dataset-membership-as-anchor is a Dataset concept — stays in
  deriva-ml.
- Nested-dataset recursion is a Dataset concept — stays in
  deriva-ml.

The deriva-py-side core would be the **JOIN planner +
materializer:** sink-finding, ambiguity detection, transparent-
bridge hopping (generalized), join-tree construction, SQL
emission. About 1800 lines of the current 2500 in
`local_db/denormalize.py` + `denormalize_planner.py`.

## Does the planner duplicate path-builder logic?

**Mostly no.** The planner *uses* `SchemaPathWalker` (the shared
deriva-py primitive) for the bidirectional FK walk. The
deriva-ml side layers on top:

- the transparency hook (feature-assoc tables, 2-FK
  associations);
- the nested-dataset loopback edge-filter (block
  `Subject → Dataset_Subject → Dataset` reverse);
- the sink-finding / Rule 5 / Rule 6 semantic enforcement;
- the JOIN-tree construction (`_build_join_tree`);
- the SQLAlchemy SELECT/UNION emission (in `denormalize.py`).

The closest thing to duplicated logic is `_table_relationship`
(`denormalize_planner.py:898-942`), which inspects FK
column-pairs between two tables. `DataPath.link()` has its own
"find the FK between context and right" logic
(`datapath.py:431-454`) but the two solve different problems:
`_table_relationship` returns the column pairs for SQL emission;
`DataPath.link` builds an ERMrest URL filter expression. They
agree on the *concept* (one FK constraint between two tables) and
on the *error message* shape ("ambiguous between two FKs") but
not on the output shape. Worth noting but not worth deduplicating.

## Recommendation on migration

**Yes — there's a clean port story.** The denormalize *planner*
and the SQL emission core would move down to deriva-py as a new
module `deriva.bag.denormalize` (or `deriva.core.denormalize`).
Public class `Denormalizer(catalog, ...)` with:

- pluggable transparency predicate (default: 2-FK associations
  only);
- pluggable anchor source (default: explicit RID list);
- pluggable column-naming (default: bare column, optional
  `schema.table.column`);
- no Dataset / feature / Execution concepts.

deriva-ml then re-exports a thinner `Denormalizer` (or
`DatasetDenormalizer`) that:

- wires in the DerivaML transparency predicate
  (feature-association tables ↔ Execution);
- wires in the feature-name shorthand resolver;
- consumes Dataset-membership-as-anchor and handles the
  nested-dataset recursion;
- adds the `version=` snapshot kwarg using the existing
  `_version_snapshot_catalog` resolver.

This split mirrors the existing `SchemaPathWalker` layering and
makes the contract precise: deriva-py owns the wide-table-shape
rules and SQL emission; deriva-ml owns the DerivaML-specific
domain semantics. Spec §6.2 Rules 2/5/6 move to deriva-py;
Rules 7/9 stay in deriva-ml (they reference Dataset and nested
datasets).

**Open question for that migration:** what to do with
`_resolve_table_names` (feature-name shorthand). It's
deriva-ml-specific, but moving the planner without it means the
public deriva-py API loses the convenience. Probably it stays in
deriva-ml as a pre-processing step the deriva-ml `Denormalizer`
runs before delegating to the deriva-py `Denormalizer`.

## Open question — should the C.2 PR consider this?

The C.2 `partition_by` parameter
(`informatics-isi-edu/deriva-ml#254`) lives on `split_dataset`
which is a deriva-ml-side concern. The partition-by-element-vs-
row distinction is a *Dataset* / *split* concept, not a
denormalize concept — so the migration doesn't affect C.2's
design. C.2 should land as-is.

What C.2 **assumes about the denormalizer** that this audit
exposed: it assumes the call shape
`include_tables=["Image", "Execution_Image_Image_Classification"],
row_per="Image"` will continue to raise
`DerivaMLDenormalizeDownstreamLeaf` (so `partition_by="element"`
needs the deduplication path, not a planner-side reduction). That
assumption is correct under the strict-downstream Rule 5 in HEAD,
and a port to deriva-py wouldn't change it. No design impact.

## TL;DR

deriva-py provides the FK graph walker (`SchemaPathWalker`) but
no denormalize / wide-table abstraction. The denormalizer fills a
real gap in deriva-py and *is* a higher-level abstraction, not a
duplicate. A clean port story exists: move planner + SQL emission
to deriva-py; keep Dataset / feature / Execution concerns in
deriva-ml. The two `local_db/denormalize*.py` files form a clean
SQL-executor / public-class split that maps to that boundary.
