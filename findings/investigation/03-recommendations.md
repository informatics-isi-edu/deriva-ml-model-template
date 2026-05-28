# Recommendations

**Investigator:** denormalizer audit (research-only)
**Date:** 2026-05-28
**Reads from:** `01-denormalizer-spec-vs-implementation.md`,
`02-denormalizer-deriva-py-overlap.md`.

These are concrete recommendations. Each is independently
applicable; severity-tagging is left for the eventual fix pass.

## 1. C.1 — revise the call site, not the planner

C.1's current shape
(`include_tables=["Image","Execution_Image_Image_Classification"]`,
`element_table="Image"`, no `row_per`) is **incompatible with
Rule 5** — and Rule 5 is enforcing exactly the contract the
caller wants when partitioning images, namely "one row per Image
without aggregation."

The fix to land at the C.1 call site (regardless of whether C.2
ships at the same time):

```python
labeled = split_dataset(
    ml,
    datasets["training"],
    test_size=test_count,
    train_size=train_count,
    stratify_by_column="Image_Class.Name",   # vocab column on the value table
    seed=split_seed,
    training_types=["Labeled"],
    testing_types=["Labeled"],
    element_table="Image",
    include_tables=["Image", "Image_Class"], # vocab table, not feature-assoc
    partition_by="element",                  # C.2 — when it lands
    split_description=_labeled_split_description(len(train_rids)),
)
```

The key change: pass the **vocab/value table** `"Image_Class"` in
`include_tables`, not the **feature-association table**
`"Execution_Image_Image_Classification"`. The feature-assoc
table is then a transparent bridge, the planner accepts
`row_per="Image"` (auto-default from `element_table`), the
stratification column projects to one column per image, and
C.2's `partition_by="element"` dedupes the underlying multi-
execution rows during selector evaluation.

**If C.2 has not landed yet:** the C.1 fix as written
(dropping `row_per`) is dead. Use the original
`row_per=<feature-assoc>` shape with an explicit post-split
disjointness assertion in the regression test
(`tests/test_load_cifar10_split_no_leakage.py`) and accept that
the train/test leakage is suppressed by the dedupe-and-retry
loop, not by the planner. This is the worst of the options
because the failure mode (leakage) returns silently if the
post-split assertion is removed; recommended only as a
short-term holdover.

## 2. Planner Rule 5 — keep it as-is

`DerivaMLDenormalizeDownstreamLeaf` is **correct.** It says "I
can't aggregate; pick `row_per` such that no `include_table` is
downstream, or drop the downstream table from `include_tables`."
That's an honest contract and the only safe one given the SQL
emission step doesn't aggregate.

**Do not relax it.** The audit's hypothesis ("the planner is
over-restrictive") was wrong — the planner is enforcing exactly
what it claims.

**Do, however:** add a one-line note to the Rule 5 user-facing
spec section about the feature-name resolver. Something like:

> Note: passing a feature *name* (e.g., `"Image_Classification"`)
> in `include_tables` is shorthand for the underlying
> **feature-association table**
> (`"Execution_Image_Image_Classification"`), which is genuinely
> downstream of the feature target. Setting `row_per=<target>`
> with a feature shorthand in `include_tables` therefore raises
> Rule 5. To project a feature value as a per-target column,
> pass the *value table* (vocab) instead.

This is the spec-side counterpart to recommendation §4.

## 3. `split_dataset` docstring — fix the broken canonical example

`src/deriva_ml/dataset/split.py:1023-1057` has two examples that
both fail at runtime today:

**Example A** (lines 1023-1030): uses
`include_tables=["Image", "Image_Classification"]` with the
implicit `row_per="Image"` auto-default. The resolver rewrites
this to `["Image", "Execution_Image_Image_Classification"]` and
Rule 5 fires. **Reproduced live** against catalog 27.

**Example B** (lines 1032-1046): same shape, with the comment
"`split_dataset` auto-defaults `row_per=element_table` when
stratifying, so the join produces one row per Image with the
classification label projected as a column." This is the
contract the docstring asserts but the resolver+planner do not
deliver.

**Suggested rewrite** for both examples:

```python
# Example A: stratified split via the vocab table column
result = split_dataset(
    ml, "28D0",
    test_size=0.2,
    stratify_by_column="Image_Class.Name",       # column on the vocab table
    include_tables=["Image", "Image_Class"],     # vocab table, not feature-assoc
    element_table="Image",
    partition_by="element",                      # C.2 — required if landed
)

# Example B: stratifying on the feature value with one row per feature
# observation (per-annotation statistics; element RIDs may overlap
# across partitions).
result = split_dataset(
    ml, "28D0",
    test_size=0.2,
    stratify_by_column="Image_Class.Name",
    include_tables=["Image", "Image_Classification"],  # feature shorthand → feature-assoc
    row_per="Execution_Image_Image_Classification",   # one row per feature observation
    partition_by="row",                                # C.2 — required if landed
)
```

Wording change at line 1037-1039 to drop the claim that the
auto-default produces "one row per Image with the classification
label projected as a column" — the strict-downstream Rule 5
prevents that combination today, and saying so in the docstring
is misleading. Either delete the claim or rewrite it to point
the user at the vocab-table shape above.

## 4. User-guide spec — close the feature-name-shorthand trap

The user guide (`docs/user-guide/denormalization.md` § "Feature
values on images", lines 264-278) shows:

```python
ds.get_denormalized_as_dataframe(["Image", "Image_Classification"])
# row_per = Execution_Image_Image_Classification (auto — feature
# association table; points to Image).
# One row per feature observation; Image columns repeated for
# multi-execution images.
```

This is correct as written — it explicitly says
`row_per = Execution_Image_Image_Classification` will auto-
infer. But the **interaction with explicit `row_per=<target>`
is not documented anywhere in the user guide.** The reader of
the user-facing guide never finds out that
`["Image", "Image_Classification"]` + `row_per="Image"`
deterministically raises Rule 5.

**Suggested addition** at the end of that section:

> Setting `row_per=<target_table>` with a feature shorthand in
> `include_tables` is rejected: the feature-association table is
> downstream of the target (Rule 5). To get "one row per target
> with the feature column projected," pass the value table
> directly (e.g., `["Image", "Image_Class"]`) rather than the
> feature shorthand. To get "one row per feature observation,"
> let auto-inference pick the feature-assoc as `row_per`.

This is the missing user-guide concept that the C.1 agent (and
the original template authors) needed and didn't have.

## 5. C.2 (`informatics-isi-edu/deriva-ml#254`) — no design changes

C.2 introduces `partition_by: Literal["element","row"] | None`
on `split_dataset`. The audit found no assumption in its design
that this investigation invalidates:

- C.2 assumes the call shape with
  `row_per=<feature-assoc>` continues to work end-to-end (it
  does — Variant 4 in §C of finding 01).
- C.2 assumes the call shape with `row_per=<element>` and
  `include_tables` containing a feature-assoc raises Rule 5
  (it does, reliably — that's the contract).
- C.2 deduplicates at the **selector layer**, not the **planner
  layer**, which is the correct seam — it doesn't require any
  planner relaxation.

Land C.2 as-is. After it lands, the C.1 revision in §1 above
becomes straightforward and the regression test
`tests/test_load_cifar10_split_no_leakage.py` becomes a clean
end-to-end check.

## 6. Should the denormalizer move to deriva-py?

**Yes, but it's a phase-3 concern.** See
`02-denormalizer-deriva-py-overlap.md` for the long version.
Short:

- `SchemaPathWalker` already lives in deriva-py — the FK-graph
  walk is shared.
- The denormalizer's *planner* (sink-finding, Rule 5, Rule 6,
  JOIN-tree construction) is the natural next thing to extract.
- The denormalizer's *SQL emission* (`_denormalize_impl` and
  `_populate_from_catalog`) is naturally a deriva-py concern
  because it builds SQLAlchemy queries against the local SQLite
  cache.
- What stays in deriva-ml: feature-name shorthand resolver
  (`_resolve_table_names`), dataset-membership-as-anchor,
  nested-dataset recursion, the feature-association transparency
  predicate.

**Suggested staging:**

1. **Phase 1** — leave the denormalizer where it is. Land the
   docstring fix (§3), the spec note (§4), the C.1 revision (§1),
   and C.2 (§5). Resolve the e2e signals.
2. **Phase 2** — introduce `Denormalizer` in deriva-py as a
   `_Denormalizer`-style internal class with a pluggable
   transparency hook. Have deriva-ml's `Denormalizer` re-export
   over it. No public-API breakage.
3. **Phase 3** — promote the deriva-py class to public. Move the
   spec doc to deriva-py. deriva-ml's class becomes a thin
   feature-aware wrapper.

This staging keeps the e2e regression test stable, lets the C.2
work land cleanly, and gives the migration a clean test
boundary at each phase.

## 7. One contract gap worth surfacing — aggregation knob

Out of scope for the e2e fix pass, but worth filing as a
deriva-ml issue:

The current contract has **no aggregation operator.** The
canonical use case "one row per Image with the feature value
reduced to a single value via selector (newest, by workflow,
by annotator)" has three workarounds (`feature_values(...,
selector=...)`, hand-rolled PathBuilder query, post-process
with pandas groupby) but no in-`Denormalizer` answer.

A `reduce_by={"Execution_Image_Image_Classification":
FeatureRecord.select_newest}` knob, or equivalently a
`row_per`+`select=` pair, would close this gap and unify the
`feature_values` and `Denormalizer` answer space.

Filing this as a known contract gap, not a blocker for the
current e2e fix pass.

## Summary of recommended changes

| # | Change | Where | Blocking? |
|---|---|---|---|
| 1 | Revise C.1 call to use `Image_Class`, `partition_by="element"` | `_cifar10_datasets.py` | Yes — for the e2e |
| 2 | Keep Rule 5 in the planner | (no change) | n/a |
| 3 | Fix split_dataset docstring stratification example | `dataset/split.py:1023-1057` | Yes — docstring is wrong |
| 4 | Add feature-shorthand+row_per note to user guide | `docs/user-guide/denormalization.md` | Yes — gap that bit |
| 5 | C.2 `partition_by` PR — no changes | `deriva-ml#254` | Already in flight |
| 6 | Denormalizer → deriva-py migration | Phase-3 work | No |
| 7 | File aggregation-knob contract gap | New deriva-ml issue | No |
