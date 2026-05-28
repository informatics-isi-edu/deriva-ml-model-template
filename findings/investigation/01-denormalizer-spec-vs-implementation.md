# Denormalizer: spec vs. implementation vs. shipped docstrings

**Investigator:** denormalizer audit (research-only)
**Date:** 2026-05-28
**Scope:** deriva-ml @ HEAD on e2e-test/2026-05-28 (`uv run` against
catalog 27 on dev-localhost).
**Mode:** No code changes. Every runtime claim below is reproduced
against catalog 27 with the exact same Python interpreter the e2e
run used.

## TL;DR

The denormalizer contract has three layers that all *almost* agree
with each other and one critical pair that doesn't:

1. The user-guide spec (`docs/user-guide/denormalization.md`) and
   the planner code in `model/denormalize_planner.py` are
   **internally coherent**. Rule 5 ("explicit `row_per` with a
   downstream `include_table` raises `DerivaMLDenormalizeDownstreamLeaf`")
   is in the spec verbatim and in the code verbatim. The strict-
   downstream primitive `_outbound_reachable_strict` does what the
   spec says it does.
2. The `Denormalizer` public surface (`local_db/denormalizer.py`) and
   the two thin wrappers on `Dataset` / `DatasetBag`
   (`dataset/dataset.py` 1651-1855, `dataset/dataset_bag.py`
   899-1053) all delegate to the same `_denormalize_impl` and the
   same planner. There is no live/bag divergence in the four
   denormalize methods — both implementations are 4-line shims.
3. **The `split_dataset` docstring stratification example
   (`dataset/split.py` 1023-1046) is broken when the bridge between
   the element table and the value table is a 3-FK
   feature-association.** Filed as the headline finding.

The C.1 agent's `DerivaMLDenormalizeDownstreamLeaf` runtime error
is **not** a planner bug. The planner is doing exactly what the
contract says. The error surfaces a real contract — the
docstring oversold it.

Two `local_db/denormalize*.py` files exist (`denormalize.py`,
`denormalizer.py`) and the layering is intentional: the
underscore-y free function `_denormalize_impl` is the SQL
executor; the `Denormalizer` class wraps it with the public
semantic-rule API. Neither is legacy.

## A. Spec faithfulness

**Question:** Does `docs/user-guide/denormalization.md` match the
implementation? Specifically — does it document the
"downstream leaf" rule that fires
`DerivaMLDenormalizeDownstreamLeaf`?

**Answer:** Yes, both the user-facing summary and the contract
section call this out explicitly. Quoting the spec:

> §3 Rule 5 — Downstream-to-`row_per` is rejected. If you set
> `row_per` explicitly and another requested table is downstream
> of it (i.e., `row_per` points to it via FK), you get an error.
> This would require aggregation (collapsing N downstream rows
> per `row_per` row), which is a future feature.

and the contract restatement at §6.2 Rule 5:

> Rule 5: Explicit `row_per` with downstream table → error. If
> `row_per` is explicitly specified and any table in
> `include_tables` is **downstream** of `row_per` (i.e., `row_per`
> has an outbound FK path to it), raise
> `DerivaMLDenormalizeDownstreamLeaf`

These match the planner's actual raise site in
`denormalize_planner.py` lines 1062-1075 (in
`_determine_row_per`). The error message body matches the
exception class docstring at `core/exceptions.py:722-742`. No
spec/implementation drift on Rule 5 itself.

The spec also notes (correctly) that Rule 5 uses
**strict-downstream** semantics — no bidirectional bridge hop.
The planner has a dedicated `_outbound_reachable_strict`
primitive for exactly this (lines 705-787) and Rule 5's check
calls it at line 1069. Reproduced live: with
`include_tables=["Image","Image_Class"]` and `row_per="Image"`,
`_outbound_reachable_strict("Image", {"Image","Image_Class"})`
returns `set()` and Rule 5 passes; with
`include_tables=["Image","Execution_Image_Image_Classification"]`
and `row_per="Image"`, the strict primitive returns
`{"Execution_Image_Image_Classification"}` (because the feature-
association table has an FK *pointing at* Image) and Rule 5
fires.

So: **spec, exception class, planner code, and runtime behaviour
are all in agreement.** The C.1 refusal is the contract working
as designed.

## B. Docstring faithfulness — the broken canonical example

**Question:** Does the `split_dataset` docstring's stratification
example at `split.py:1023-1046` actually work?

**Answer:** **No.** Reproduced live.

The exact docstring example is:

```python
result = split_dataset(
    ml, "28D0",
    test_size=0.2,
    stratify_by_column="Image_Classification.Image_Class",
    include_tables=["Image", "Image_Classification"],
    element_table="Image",
)
```

Inside `split.py:639`:

```python
effective_row_per = row_per if row_per is not None else element_table
```

so `effective_row_per = "Image"`. The denormalizer call shape is
therefore:

```python
denormalizer.as_dataframe(
    include_tables=["Image", "Image_Classification"],
    row_per="Image",
)
```

But `Denormalizer._resolve_table_names`
(`local_db/denormalizer.py:1091-1190`) translates the **feature
name** `"Image_Classification"` to the underlying
**feature-association table** `"Execution_Image_Image_Classification"`
before the planner ever sees the list. This translation is
documented in the user guide (§ "Feature values on images") and
in §8.4 of the contract:

> `describe`, `as_dataframe`, `as_dict`, and `columns` all share
> the same `include_tables` / `via` / `row_per` validation via
> the private `_resolve_table_names` helper.

After resolution, the planner sees
`include_tables=["Image", "Execution_Image_Image_Classification"]`
with `row_per="Image"`. The feature-assoc table has a domain FK
*pointing at* Image (that's part of what makes it a feature
table), which puts it in
`_outbound_reachable_strict("Image", {...})`. Rule 5 fires.

Live verification on catalog 27:

```
=== d.as_dataframe(['Image','Image_Classification'], row_per='Image') ===
RAISED: DerivaMLDenormalizeDownstreamLeaf
  Table(s) ['Execution_Image_Image_Classification'] are downstream of
  row_per='Image'. One row per Image would require aggregating
  multiple rows of ['Execution_Image_Image_Classification'] —
  aggregation is not yet supported.
```

For comparison, the same call with the vocabulary table name
spelled out succeeds:

```
=== d.as_dataframe(['Image','Image_Class'], row_per='Image') ===
OK plan; row_per resolves to 'Image'.
```

The bug here is **the docstring example, not the planner.** The
docstring suggests `["Image", "Image_Classification"]` produces a
one-row-per-image projection of the vocab term, but the resolver
silently rewrites that to a feature-assoc-inclusive request that
Rule 5 must refuse. There is **no spelling of
`include_tables`** that gives the user "one row per image, with
the vocab term projected" through `split_dataset`'s current
plumbing — they have to switch to the vocab table name
(`Image_Class`) directly, and then the strict-downstream check
passes.

**Classification:** the docstring is wrong, the contract is
right.

## C. The `DerivaMLDenormalizeDownstreamLeaf` rule

**Question:** What is the *actual* condition that triggers this
exception? Why does the rule exist?

**Raise site** (`denormalize_planner.py:1062-1075`):

```python
if row_per is not None:
    if row_per not in include_tables:
        raise ValueError(...)
    downstream = self._outbound_reachable_strict(row_per, all_tables)
    downstream_in_inc = [
        t for t in include_tables
        if t in downstream and t != row_per
    ]
    if downstream_in_inc:
        raise DerivaMLDenormalizeDownstreamLeaf(
            row_per=row_per,
            downstream_tables=sorted(downstream_in_inc),
        )
```

**Strict-downstream semantics**
(`_outbound_reachable_strict` lines 705-787): walks `referenced_by`
edges only (tables with FK pointing AT the current table), does
NOT bidirectionally hop transparent bridges. Returns the set of
names reachable from `row_per` along outbound (downstream) FK
chains, restricted to tables in `all_tables = include ∪ via`.

**Why it exists** (per spec §3 Rule 5 and code comments at
1065-1068): the planner doesn't aggregate. One row per `row_per`
with another `include_table` strictly downstream would require
collapsing N downstream rows into 1 row per `row_per` — that's
aggregation, an unimplemented feature.

**Verified live:** the C.1 attempt
`include_tables=["Image","Execution_Image_Image_Classification"]`
with auto-default `row_per="Image"` hits this. The feature-assoc
table has `Execution_Image_Image_Classification.Image` FK to
`Image.RID`, so under strict-downstream Image points outbound to
the feature-assoc table. The error message is accurate: there
genuinely are multiple feature-assoc rows per image (multi-
execution annotation, multi-label features), and "one row per
Image" cannot represent them without collapsing.

**Minimal synthetic reproducer.** The schema:

- `Image` (RID, Filename)
- `Image_Class` vocabulary (RID, Name)
- `Execution` (RID, ...)
- `Execution_Image_Image_Classification` (FK→Image, FK→Image_Class,
  FK→Execution)

Call:
```python
d = Denormalizer(any_dataset_with_image_members)
d.as_dataframe(
    include_tables=["Image", "Execution_Image_Image_Classification"],
    row_per="Image",
)
# → DerivaMLDenormalizeDownstreamLeaf
```

Drop `row_per` and the planner auto-infers
`Execution_Image_Image_Classification` (one row per annotation,
not per image — which is what the curator found leaks across the
train/test partitions of the e2e run's labeled splits).

## D. Live-catalog vs bag divergence

**Question:** Are the four denormalize methods on `Dataset` and
`DatasetBag` equivalent?

**Answer:** Yes. Both are thin 4-line wrappers that construct
`Denormalizer(self [, version=...])` and forward to the same
underlying public method. Verified by reading:

- `dataset/dataset.py:1651-1855` — Dataset versions all instantiate
  `Denormalizer(self, version=version)`.
- `dataset/dataset_bag.py:899-1080` — DatasetBag versions
  instantiate `Denormalizer(self)`.

Both end up in `Denormalizer.__init__`
(`local_db/denormalizer.py:97-216`) which derives `source` from
the dataset shape: live `Dataset` → `source="catalog"` with a
real `PagedClient`; bag/fixture → `source="local"` with a
pre-populated engine. From there, both call `_denormalize_impl`
with the same planner output and the same SQL emission code
(spec §2: "The SQL emission code (Steps 4 & 5) is identical across
the three modes; only the row-population side differs.").

Same exception classes raise from both paths (`as_dataframe`
raises `DerivaMLDenormalizeMultiLeaf` etc. — see method docstring
§8.2). Same `_resolve_table_names`. Same Rule 7 / Rule 8 anchor
classification. The only documented difference is the `version=`
kwarg: meaningful on `Dataset`, silently ignored on `DatasetBag`
because bags are already version-pinned (spec §6.1).

**No divergence finding.** This is the place the audit was
worried about (two parallel implementations diverging silently);
the actual implementation is one path with two front doors.

## E. The two `local_db/denormalize*.py` files

**Question:** Why two files? What is the layering?

**Answer:** Intentional layering, no legacy:

- **`local_db/denormalize.py`** (789 lines): the SQL executor.
  Hosts `_denormalize_impl` (private), `DenormalizeResult`,
  `_populate_from_catalog`, `_foreign_keys_off`. No public API —
  the module docstring explicitly says "The free function below
  is private."

- **`local_db/denormalizer.py`** (1702 lines): the public class
  `Denormalizer`. Owns `__init__` / `from_rids` /
  `as_dataframe` / `as_dict` / `columns` / `describe` /
  `list_paths`. Calls into `_denormalize_impl` from the
  materialization paths. Hosts `_resolve_table_names`,
  `_classify_anchors`, `_anchors_as_dict`, the orphan-row
  emitter, and the dry-run envelope for `describe`.

The split mirrors the contract document's split between §2-§6
(architecture, fetcher contract, INSERT contract, denormalize
pipeline — covered by `denormalize.py`) and §8 (public class
surface — covered by `denormalizer.py`). Naming overlap is
unfortunate but the docstrings call it out explicitly:

> denormalize.py: "Unified denormalization engine for the
> local_db layer. This module hosts the low-level
> `_denormalize_impl` primitive."
> denormalizer.py: "Denormalizer — public API for producing wide
> tables from Deriva data. Wraps the lower-level
> `_denormalize_impl` primitive."

**Conclusion:** the split is by design; the only finding here is a
naming hazard (two files differing by one trailing `r`). Worth
noting but not a refactor target — every active code path uses
both files and the docstrings disambiguate at the top.

## F. deriva-py overlap

See `02-denormalizer-deriva-py-overlap.md` for the long form.
Short answer: the denormalizer's *FK-graph walker*
(`SchemaPathWalker`) **already lives in deriva-py**
(`/Users/carl/GitHub/deriva-py/deriva/bag/path_walker.py`),
shared between deriva-ml's planner and deriva-py's
`CatalogBagBuilder`. What's deriva-ml-specific is everything on
top: transparency rules (feature-assoc tables),
row_per/sink-finding, the four semantic Rules 5/6/7/8, anchor
classification, the SQL-emission step, and the
`_resolve_table_names` feature-name shorthand. None of those have
analogues in deriva-py today.

## G. C.1 — bug, contract gap, or user error?

**Combining all of the above:** the C.1 agent's
`DerivaMLDenormalizeDownstreamLeaf` is *not* a planner bug. It's
the planner correctly enforcing a contract the docstring
oversold.

Sequence of facts:

1. `_cifar10_datasets.py` calls
   `split_dataset(..., include_tables=["Image",
   "Execution_Image_Image_Classification"],
   row_per="Execution_Image_Image_Classification",
   element_table="Image")`. The explicit
   `row_per=<feature-assoc>` is what produced the train/test leakage
   (33+24 image-RID overlap) — feature rows partition, but two
   feature rows of the same image can split. (Curator-02,
   Evaluator-02.)
2. The C.1 fix drops `row_per`. Now `effective_row_per` =
   `element_table` = `"Image"`, with `include_tables` still
   carrying `Execution_Image_Image_Classification`. **Rule 5
   fires correctly:** the feature-assoc table genuinely is
   strict-downstream of Image, and aggregation is unimplemented.
3. The fix path the C.1 commit message implies (drop `row_per`,
   rely on docstring §1040-1046's "auto-defaults row_per=Image")
   does not work for **any** feature-bridged stratification on
   the current planner, because of the feature-name shorthand
   silently inserting the feature-assoc table.

**Two separable problems here:**

- The **docstring at lines 1040-1057** is wrong about the
  `["Image", "Image_Classification"]` case (the resolver makes it
  `["Image", "Execution_Image_Image_Classification"]` and Rule 5
  fires). Fix: rewrite the example to use the vocab table name
  (`Image_Class`) directly, or to use the
  `row_per=feature-assoc` shape and document the partition-by-
  feature-row semantics it commits to (the partition-leakage
  problem C.2's `partition_by="element"` parameter is designed
  to handle).
- The **C.1 call site** can't reach a working denormalize through
  `Execution_Image_Image_Classification` while keeping
  `row_per="Image"`. The right shape is:
  - `include_tables=["Image", "Image_Class"]` (drop the
    feature-assoc; let the resolver hop through it as a
    transparent bridge), and either
  - omit `row_per` and let auto-infer pick — but on this catalog
    `["Image", "Image_Class"]` is a multi-leaf (both tables are
    sink candidates because the feature-assoc bridge is
    transparent and neither side is strict-downstream of the
    other), so the planner raises
    `DerivaMLDenormalizeMultiLeaf` and the caller must pick.
  - or pass `row_per="Image"` explicitly (works on the
    feature-bridge case under strict-downstream).

**Concrete recommendation for C.1's revision:** see
`03-recommendations.md` §1. The cleanest shape is to keep the
C.2 `partition_by` parameter (which already exists on issue #254
and addresses the leakage at the partition-tracking level) and
to migrate the C.1 call site to use the vocab table name with
`partition_by="element"` once C.2 lands. The denormalizer side
needs no further change for this specific call site.

## H. The routes-around-the-denormalizer pattern

The Analyst's `scripts/build_joined_wide_table.py` and the
Curator's discovery of `row_per=<feature-assoc>` as the original
template's workaround are the same pattern from two ends:

- The Analyst hit `describe()` returning the wrong row-count
  estimate (the F6 / A02 case — fixed) and decided not to trust
  the denormalizer for a per-image GT+predictions wide table.
  They built a PathBuilder-direct query that picks one feature
  row per image manually.
- The original `_cifar10_datasets.py` had to put
  `row_per="Execution_Image_Image_Classification"` to get the
  planner to accept the call at all — without it, Rule 5 fires
  (auto-inferred row_per becomes Image, feature-assoc is
  downstream). The workaround **causes** the leakage by
  partitioning feature rows instead of images.

The pattern under both: **the contract that the docstring
suggests (one row per Image, projecting a feature column)
doesn't have a clean expression through the current API.** The
caller wants "per-Image with the feature value reduced", and
the planner's Rule 5 + lack of aggregation means there's no
shape that produces that result. Three workarounds exist:

1. Hand-roll PathBuilder + Python aggregation (Analyst's
   choice — `scripts/build_joined_wide_table.py`).
2. Use `feature_values(..., selector=...)` (spec § "When to reach
   for `feature_values` vs `Denormalizer`") and join in Python.
3. Denormalize at one-row-per-feature-row and post-process with
   pandas groupby to collapse to one row per image. This
   is what the C.2 `partition_by="element"` parameter does
   inside `split_dataset` for stratification-specific shapes.

There's no in-`Denormalizer` answer to "give me one row per
Image with the feature value reduced." That's the **contract
gap**, and it's real. It's not a bug — Rule 5's "no aggregation"
is an explicit and consistent rule — but the docstring and the
user-guide both undersell that consequence.

## Open question — should `Denormalizer` grow an aggregation knob?

The contract section (§7 row F6, "honest unknown" pattern)
suggests this would be welcome. A `reduce_by=` or
`select=Newest` knob that takes a selector and collapses
multi-feature-row windows to one row would let the docstring's
"one row per Image with the feature value" example actually
work.

But that's out of scope for this audit. Filed as a contract gap
worth surfacing.

## Limitations of this audit

1. I only reproduced against the catalog 27 schema (CIFAR-10:
   Image + Image_Class + 3-FK feature-assoc). Other shapes
   (4-FK associations, multi-vocab features, asset-typed
   features) were not exercised.
2. I did not run the `tests/local_db/` or `tests/dataset/` test
   suites end-to-end against the current HEAD — relied on
   reading the test names and one-shot runtime probes against
   catalog 27.
3. The C.2 `partition_by` PR is open at
   `informatics-isi-edu/deriva-ml#254` but has not been merged.
   My recommendations assume it will land roughly as proposed.
4. Did not look at the in-flight slice path
   (`source="slice"`); the spec mentions it but the e2e signals
   are all on catalog/local.
5. Did not reproduce the C.5x "freshness regression" xfail
   case mentioned in spec §6.5 — those are flagged but unwritten
   in deriva-ml already.
