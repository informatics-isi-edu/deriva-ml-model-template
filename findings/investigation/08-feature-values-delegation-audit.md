# 08 — Can `feature_values` Delegate to `Denormalizer.as_dict`?

**Audit date:** 2026-05-29
**Scope:** Stage 2 of the `feature_values` / `Denormalizer`
consolidation. Decides whether Stage 3 (delegation) is safe.
**Method:** Code reading only — dev-localhost catalog 27 was not
reachable from this environment (`curl https://dev-localhost.derivacloud.org/ermrest/catalog/27/schema` returned 000 / DNS).
**Inputs:** PR #257 diff (commit `46836177`, branch
`feat/denormalizer-selector`), current `main` for the
PathBuilder-side surfaces.

---

## 1. TL;DR — **NO-GO for a thin wrapper. CONDITIONAL-GO for a
hybrid.**

Stage 3 as proposed ("`feature_values` becomes 5-10 lines
delegating to `Denormalizer.as_dict` + FeatureRecord
materialization") **cannot ship without behavior changes**. There
are three structural mismatches:

1. **`DerivaML.feature_values` is NOT dataset-scoped; `Denormalizer`
   is intrinsically dataset-scoped.** The denormalize SQL has a
   hard-coded `WHERE Dataset.RID IN (dataset_rid, ...children)`
   clause (`_denormalize_impl` Step 4) and traverses dataset
   membership tables. `DerivaML.feature_values("Image", "Foo")`
   today returns every Foo row in the catalog regardless of any
   dataset. `Denormalizer` cannot reproduce that "catalog-wide
   read with no Dataset anchor" shape.
2. **`materialize_limit` and `execution_rids` have no equivalent
   on `Denormalizer`.** Both are server-side query knobs on the
   PathBuilder query; the denormalize path runs the full plan and
   reduces post-materialization. Adding them to `Denormalizer`
   means widening its API for a single-feature use case, which
   inverts the consolidation goal.
3. **Bag mode reads from two different stores.** `DatasetBag.feature_values`
   reads from a per-feature `_feature_cache_*` SQLite table whose
   columns are all stored as TEXT (`BagFeatureCache._ensure_cache_populated`
   uses `Column(name, Text)`). `Denormalizer` (bag mode) reads
   from the SQL-typed source ORM tables. Subtle datetime / array
   coercion divergences are likely.

The **conditional-go shape**: `Dataset.feature_values` and
`DatasetBag.feature_values` can delegate to `Denormalizer.as_dict`
(both are already dataset-scoped, both already eagerly materialize),
**but `DerivaML.feature_values` must keep its PathBuilder path**.
The consolidation eliminates two of three implementations, not
three of three.

Material findings ranked: §F (dataset-scope) > §D
(`materialize_limit`) > §G (bag-mode TEXT vs typed read).

---

## 2. Row-shape equivalence (Question A)

### `DerivaML.feature_values("Image", "Image_Classification")`

Path: `pathBuilder().schemas[...].tables[<feat_assoc>].entities().fetch()`
→ each raw dict → `record_class(**{k: v for k, v in raw.items() if k in field_names})`
yields one `FeatureRecord` per feature-assoc row.

Fields populated (`FeatureRecord.feature_record_class` in
`feature.py:632`):

- `RID` — **NOT a declared field on FeatureRecord** (no `RID` in
  `FeatureRecord.__fields__` — only `Execution`, `Feature_Name`,
  `RCT`, plus the target column and any value/term/asset columns).
  So `RID` from the raw row is silently dropped at the
  `{k: v ... if k in field_names}` filter step.
- `RCT` — declared on `FeatureRecord` (`Optional[str]`). The
  PathBuilder fetch returns it as an ISO-8601 string already.
- `Execution` — declared on `FeatureRecord`. Bare RID string.
- `Feature_Name` — declared, defaults to feature name. (Pulled from
  the row but the default applies if missing.)
- `<TargetTable>` (e.g. `Image`) — declared as required `str`. Bare
  RID.
- Each value/term/asset column — declared via `feature_columns`
  (e.g. `Image_Class`, `Confidence`).
- `RMT`, `RCB`, `RMB` — **NOT on FeatureRecord**. Dropped by the
  field-name filter.

### `Denormalizer.as_dict(include_tables=["Image", "Execution_Image_Image_Classification"], row_per="Execution_Image_Image_Classification")`

Path: planner builds wide SQL, materializes rows, each row is a
`dict[str, Any]` keyed by **dotted column labels**.

Column-name shape (`denormalize_column_name`):
- `Image.RID`, `Image.<col>` for the Image target.
- `Execution_Image_Image_Classification.RID`,
  `Execution_Image_Image_Classification.Image`,
  `Execution_Image_Image_Classification.Image_Class`,
  `Execution_Image_Image_Classification.Confidence`,
  `Execution_Image_Image_Classification.Execution`,
  `Execution_Image_Image_Classification.Feature_Name`.

System columns (`RCT`, `RMT`, `RCB`, `RMB`) are **skipped by the
planner** for every contributing table (per PR #257 commit
message; the `_prepare_wide_table` skip list). So a no-selector
`as_dict` call **does not return RCT at all**. The selector path
recovers RCT via the supplementary SELECT keyed by feature-assoc
RID, but that recovery is only on the rows surviving selector
reduction — the supplementary RCT is stuffed into the synthesized
`FeatureRecord`, but does **not** appear in the output dict.
This is fine for a selector-using caller (`select_newest` works)
but a plain `feature_values()` (no selector) call would lose RCT
in the dict shape.

### Divergence summary

| Property | `feature_values` | `Denormalizer.as_dict` |
|---|---|---|
| Output type | `FeatureRecord` (pydantic) | `dict[str, Any]` |
| Column name | bare (e.g. `Image`) | dotted (`Image.RID`, etc.) |
| RID column | not on record | included as `<feat>.RID` |
| RCT column | included | **skipped** unless selector path |
| RMT/RCB/RMB | dropped (not on record) | skipped (planner) |
| Target FK | `Image` (bare RID str) | `<feat>.Image` (bare RID str) |
| Image's own cols | absent | included (`Image.URL`, etc.) — extra |

**Conclusion:** column sets are different. The dict from
`as_dict` carries Image's wide-table columns the FeatureRecord
shape doesn't expose. Any delegation must project down to the
feature-assoc columns and re-key bare names before constructing
FeatureRecord. That's not 5-10 lines — it's a non-trivial
adapter.

---

## 3. Selector semantics (Question B)

The two paths share the **target-RID grouping** and **selector
callable shape**. PR #257's `_apply_selector` is structurally
parallel to `reduce_with_selector` in `feature.py`:

- Both group by the feature's target-table FK column.
- Both apply `selector(list[FeatureRecord]) -> FeatureRecord | None`.
- Both drop `None` returns.

Key non-obvious detail PR #257 handles: the planner skips RCT, so
the selector path's `FeatureRecord` shadows would have
`RCT=None`, and `select_newest`'s `max(records, key=lambda r: r.RCT or "")`
would tie on the empty string and pick arbitrarily. The
supplementary SELECT (`_apply_selector` lines reading
`feat_orm.__table__.c.RID.in_(feature_rids)`) backfills RCT (and
any other missing FeatureRecord fields) keyed by feature-assoc
RID, with a datetime → ISO-8601 string coercion to match the
PathBuilder shape.

**Equivalence verdict:** YES, for the rows that survive
grouping. Both paths feed identical `FeatureRecord` lists to the
selector, modulo the FeatureRecord shadow's missing
non-PathBuilder fields (Image's own columns, e.g.) — but those
are not declared on `FeatureRecord`, so `extra="forbid"` would
reject them anyway. The selector receives the same data shape.

**One subtle concern:** the structural fallback (offline-fixture
case where `find_features()` returns empty) synthesises a
`FeatureRecord` subclass where **every value column is
`Optional[str]`** (PR #257 line ~`_apply_selector`: "Field types
collapse to `Optional[str]`"). Customer selectors that read a
typed `Confidence: float` would see a string. This is fine for
the offline-fixture path the PR explicitly scopes it to, but a
Stage 3 delegation needs to verify `find_features()` always
returns the live catalog's feature before falling through — and
that path needs a test.

---

## 4. `FeatureRecord` typing layer (Question C)

Where does FeatureRecord materialization happen today?

- `DerivaML.feature_values` (feature.py:514): builds
  `feat.feature_record_class()` once per call, then iterates raw
  PathBuilder rows constructing instances.
- `Dataset.feature_values` (dataset.py:657-687): delegates to
  `DerivaML.feature_values` (records already typed), then
  filters by `getattr(rec, target_col, None) in members`.
- `DatasetBag.feature_values` (dataset_bag.py:537-650): delegates
  to `BagFeatureCache.fetch_feature_records`, which builds the
  record class from `bag.model.lookup_feature(...)` and yields
  `record_class(**filtered)` for each cache row.

Can FeatureRecord be constructed from a generic dict?

- **Yes**, modulo two things:
  - The dict needs **bare column names** (not dotted), and only
    keys in `record_class.model_fields`. The denormalize dict has
    dotted labels and extra columns — a delegation adapter must
    strip the prefix and project down.
  - `extra="forbid"` (from `FeatureRecord.Config`) means stray
    keys raise. The adapter must be strict.

What state is needed? Only the `Feature` object (to get the record
class). `Denormalizer` doesn't carry one — but `DerivaML.lookup_feature(table, feature_name)`
is cheap (uses `model.find_features` which walks the schema once).

**Verdict:** the materialization is reproducible from
`Denormalizer.as_dict` output **plus a model handle**, with a
strip-prefix-and-project adapter. The adapter is small but
non-trivial. The Stage 3 wrapper would look something like:

```python
def feature_values(self, table, feature_name, ...):
    feat = self.lookup_feature(table, feature_name)
    record_cls = feat.feature_record_class()
    feat_table = feat.feature_table.name
    target_col = feat.target_table.name
    field_names = set(record_cls.model_fields.keys())
    prefix = f"{feat_table}."  # multi_schema-aware in real code
    for row in denormalizer.as_dict(
        include_tables=[target_col, feat_table],
        row_per=feat_table,
        selector=selector,
    ):
        kwargs = {
            k[len(prefix):]: v
            for k, v in row.items()
            if k.startswith(prefix) and k[len(prefix):] in field_names
        }
        yield record_cls(**kwargs)
```

(Plus RCT recovery — this version inherits the
no-RCT-in-no-selector-mode regression unless you always go
through the selector branch or extend `as_dict` to optionally
include system columns.)

---

## 5. `materialize_limit` gap (Question D)

`Denormalizer` has **no row cap**. There is no `LIMIT` parameter,
no row-counter, no `DerivaMLMaterializeLimitExceeded` raise
point in `_denormalize_impl` or `Denormalizer`.

The denormalize pipeline materializes the full SQL result into
`rows = [dict(row._mapping) for row in result]` (denormalize.py:426)
before either `_apply_selector` or the orphan-row combine runs.
A 1M-row feature would balloon memory before any user code sees
a row.

Three options for Stage 3:

a. **Pre-count via `describe()`'s `estimated_row_count`.** Doesn't
   work cleanly: `describe()` returns `None` for downstream-anchor
   row counts (the common feature-table case), and the docstring
   explicitly says the row count is unknown without a catalog
   query (this is the same A02 honest-unknown finding from
   2026-05-21).
b. **Add `materialize_limit` to `Denormalizer.as_dict`.** Pushes
   the cap into the planner. Defensible — it's not feature-specific,
   it's a general "bound the wide table" knob. But widens the
   denormalize API for what's effectively a single-call-site need.
c. **Post-filter in the `feature_values` wrapper.** Run
   `Denormalizer.as_dict`, count as you yield, raise if the
   count exceeds. Problem: by the time the wrapper sees the
   count, the rows are already materialised — the cap was
   supposed to guard against memory blow-up *before*
   materialization.

**Recommendation:** option (b), but scoped — add
`materialize_limit` to `Denormalizer.as_dict` as a generic
"raise if N exceeded" gate. The check fires after row
materialization but before the (potentially expensive) selector
pass. This is the same imperfect guard `feature_values`
provides today (the PathBuilder fetch is also a full
materialization), so it's not a regression.

---

## 6. `execution_rids` gap (Question E)

`feature_values` filters the upstream PathBuilder query
server-side: `feature_path.filter(reduce(or_, predicates))` where
predicates are `feature_path.Execution == rid`. The benefit is
the round-trip pulls only the matching subset.

`Denormalizer` has no analogue. The closest mechanism would be:

- **`via=["Execution"]`** — forces Execution into the join chain
  without column projection. But it does not filter; it only
  routes.
- **Anchor classification (Rule 7).** If you classified
  Execution as an upstream anchor (via `from_rids`), the SQL
  join would filter feature rows by those Execution RIDs. But
  this requires `from_rids` construction, not a Dataset, and
  is only valid for the `DerivaML.feature_values` case (no
  Dataset scope).

For Stage 3 of the consolidation, the realistic shape is
**post-filter in the wrapper**:

```python
if execution_rids is not None:
    if not execution_rids:
        return
    exec_set = set(execution_rids)
    rows = (r for r in rows if r.get(f"{feat_table}.Execution") in exec_set)
```

This loses the round-trip savings. For a dataset with 10 trained
models and one filter at evaluation time, the cost is N (rows
materialized) vs k (rows after filter). On a small dataset
that's fine; on a 100k-image feature table with a
single-execution filter, it's 100x wasted work.

**The alternative — extending `Denormalizer.as_dict` with a
`filter_by_column` parameter** — is also possible but is a real
API widening. The Denormalizer's design philosophy explicitly
keeps the column set declarative (via `include_tables`) rather
than predicate-based. Adding row-predicate filtering changes the
design.

**Verdict:** post-filter is a regression but tolerable for
dataset-scoped reads. Skip the optimization for Stage 3.

---

## 7. Dataset-scope filter and Rule 7 (Question F) — THE BLOCKER

This is the deepest mismatch.

### What `DerivaML.feature_values` does

`DerivaML.feature_values("Image", "Foo")` returns **every row of
the Foo feature-assoc table in the catalog**, regardless of any
dataset. It is a catalog-wide read.

### What `Dataset.feature_values` does

It calls `self._ml_instance.feature_values(...)` (the catalog-wide
read), then filters Python-side: `if getattr(rec, target_col, None) in members`
where `members = set(self.list_members(table))`.

### What `Denormalizer` does

`Denormalizer` is **always** dataset-scoped. The SQL emitted by
`_denormalize_impl` has `WHERE Dataset.RID IN (dataset_rid, ...children)`
(denormalize.py:400) and the join chain traverses dataset
membership tables (`Dataset → Dataset_Image → Image → feat_assoc`).
There is no "no-dataset" mode short of `from_rids`, which still
requires a `dataset_rid` against a live catalog (rejected with
`SC-02` ValueError otherwise).

### The implication

`DerivaML.feature_values` cannot delegate to `Denormalizer` without
losing the catalog-wide-read semantics. Two reads of the same
catalog with the same feature:

| Today | After hypothetical delegation |
|---|---|
| `ml.feature_values("Image", "Foo")` → N rows (whole catalog) | requires synthesizing a Dataset → impossible without one |

You could try to make this work by constructing a synthetic
`Denormalizer.from_rids(...)` where the anchor set is "every Image
RID in the catalog". But that's strictly worse than the existing
PathBuilder path: it requires a pre-flight catalog scan to list
all Image RIDs, then a second query to fetch the feature rows.

### `Dataset.feature_values` is the consolidation candidate

For `Dataset.feature_values`, delegating to `Denormalizer` is
**semantically aligned**: both are dataset-scoped.

The current Dataset wrapper does Python-side filtering after a
catalog-wide fetch. The Denormalizer SQL would filter via
JOIN — which is structurally better (less data over the wire)
**but** only if the catalog-wide feature table is large relative
to the dataset-scoped subset. For a 100-image dataset against a
100k-row feature table, the SQL filter wins by 1000x. For a
100k-image dataset against the same table, the two paths
materialize comparable row counts.

### Verdict

- `DerivaML.feature_values`: **cannot delegate** (catalog-wide
  read has no Denormalizer analogue).
- `Dataset.feature_values`: **can delegate** (semantically
  equivalent; SQL filter is a behavior improvement).
- `DatasetBag.feature_values`: **can delegate** (bag is already
  dataset-scoped).

This matches the hybrid migration shape (§11).

---

## 8. Bag mode equivalence (Question G)

`DatasetBag.feature_values` reads from `BagFeatureCache`, which
stores every column as **TEXT** (`bag_feature_cache.py:163`:
`cache_columns.append(Column(name, Text))`). The comment
explicitly says: "Store everything as TEXT; Pydantic reifies
types at FeatureRecord construction time."

`Denormalizer` in bag mode reads from the ORM source tables —
which carry **their declared types**, including the
TypeDecorators that the bag schema uses:

- `ArrayAsJson` (post-PR #266) decodes JSON-text array columns
  back to Python lists at SQLAlchemy read time.
- `StringToDate` decodes ISO date strings.
- Datetime columns may come back as Python `datetime` objects
  (the PR #257 supplementary fetch coerces these via `.isoformat()`
  — but only on the supplementary path, not the main wide-table
  read).

**Divergence points:**

1. **Array columns.** `feature_values` (bag) reads pre-stringified
   TEXT and lets Pydantic coerce — for an `Optional[str]` field
   declared on the FeatureRecord, the array column would arrive
   as a JSON string. For a `list[str]`-typed field (rare on
   features today, but possible), Pydantic would refuse to coerce
   a JSON string into a list and raise. Denormalizer would see
   the ArrayAsJson-decoded list directly.
2. **Datetime columns.** `feature_values` (bag) cache stores
   strings → FeatureRecord gets strings (matches the live
   catalog ISO-8601 shape). Denormalizer wide-table read may
   return Python datetimes for the main columns (only the
   selector path's supplementary fetch coerces).
3. **NULL columns on the target.** `feature_values` (bag)
   includes a fallback filter: only records whose `target_col` RID
   is in `bag.model.get_table_contents(target_col)` are yielded
   (dataset_bag.py:610-618). This catches "dangling rows" where
   the bag walker over-reached an association table. Denormalizer
   has Rule 7 anchor classification that approximates this but
   is not identical — Rule 7 cares about whether the target FK
   path is reachable; the bag's `target_rids` check is a
   set-membership check after the wide-table join.

**Verdict:** divergence is real and not trivially fixable. A
Stage 3 delegation in bag mode needs:
- Live test against a downloaded bag with at least one array
  column, one datetime column, and one feature whose target
  table has dangling FKs (the #126 case).
- Either confirm the divergence is zero in practice, or document
  the new shape as a breaking change.

---

## 9. Performance (Question H)

Single-feature read against a live catalog:

| Path | HTTP requests | SQL | Memory peak |
|---|---|---|---|
| `feature_values` (DerivaML, PathBuilder) | 1 (ERMrest entities() page) | 0 | N (rows) |
| `Denormalizer.as_dict` (catalog mode) | 1+ (planner walks join chain, fetches every contributing table page-by-page) | 1 (SQL JOIN against local SQLite) | N + size of join intermediates |

The denormalize path is materially heavier for a single-feature
read. The planner walks the FK graph, the catalog fetcher pulls
every contributing table's rows (Dataset, Dataset_Image, Image,
feature_assoc all populated into local SQLite), then the SQL join
runs. Compared to a single PathBuilder query that returns the
feature-assoc rows directly, this is multiple round-trips and an
intermediate SQLite materialisation.

For Stage 3:
- **`Dataset.feature_values` already pays for this** today — its
  underlying `_ml_instance.feature_values` is a single PathBuilder
  query, but the wrapper does an in-Python `list_members` lookup
  (which is its own catalog fetch) and a Python loop filter.
  Denormalizer's join-side filter is structurally cleaner and
  may be faster on large catalogs.
- **`DerivaML.feature_values`** would be slower under
  Denormalizer — but as §7 establishes, `DerivaML.feature_values`
  shouldn't delegate anyway.
- **`DatasetBag.feature_values`** would be marginally slower (the
  cache table is a single SELECT today vs. a join against source
  ORM tables under Denormalizer), but bag-local SQLite is fast
  enough that this likely doesn't matter in practice.

No material concern, given the hybrid shape.

---

## 10. Edge cases (Question I)

| Case | `feature_values` | `Denormalizer.as_dict` |
|---|---|---|
| Feature with zero rows | empty iterator | empty (no rows materialise) |
| Null FK on target | (target=None → drops in `reduce_with_selector` per `feature.py:97`) | LEFT-JOIN style: row appears with `target_col=None`, target's hoisted cols=None |
| Feature on table with zero dataset members | non-empty if catalog has rows; Dataset wrapper filters to empty | dataset-scoped → empty |
| selector returns None for every group | empty | empty (verified in PR #257 tests) |
| `execution_rids=[]` | short-circuits to empty (feature.py:492) | no short-circuit — runs planner and SQL |

**Notable behavior change candidates:**
- The "null FK on target" case differs. PathBuilder feature
  rows with NULL target FK are silently dropped (the
  `reduce_with_selector` skip on `target_rid is not None`).
  Denormalizer would emit an orphan-style row with all target
  cols NULL. A no-selector pass through the wrapper would yield
  a record with `target_col=None`, which currently can't happen.
- `execution_rids=[]` short-circuit — Stage 3 wrapper needs to
  preserve this explicitly.

---

## 11. Recommended migration shape (Question J) — Hybrid

Of the three candidate shapes:

| Shape | Verdict |
|---|---|
| 1. `feature_values` body → 5-10 lines wrapping `Denormalizer.as_dict` | **No.** Loses catalog-wide-read semantics, `materialize_limit`, `execution_rids`, RCT in no-selector mode. |
| 2. Extract one shared helper, all three call it | **No.** Same problem — the helper has to handle three different scoping models. |
| 3. **Hybrid: `DerivaML.feature_values` keeps PathBuilder. `Dataset.feature_values` and `DatasetBag.feature_values` delegate to `Denormalizer`.** | **Conditional yes.** |

The hybrid:

- Eliminates the dataset-scope-related work duplication
  (`Dataset.feature_values` no longer does Python-side member
  filtering; it leans on the SQL join).
- Eliminates the bag-cache layer for `DatasetBag.feature_values`
  (read straight from ORM tables via `Denormalizer`). The
  `BagFeatureCache` becomes dead code — but its TEXT-storage
  shape is exactly the divergence point in §8, so removing it
  *might* fix the array-decode divergence rather than reveal it.
- Leaves `DerivaML.feature_values` as the canonical
  feature-read primitive for the catalog-wide use case, with no
  Dataset context required.

**Reduction:** three implementations → two (the
`DerivaML.feature_values` PathBuilder path + the `Denormalizer`
SQL path). The two cover non-overlapping needs (catalog-wide
read vs dataset-scoped read), so the architectural goal —
"minimize different ways to implement the same functionality" —
is reached: the SAME functionality is implemented exactly once
per axis.

Stage 3, as a concrete PR, is then:

1. Rewrite `Dataset.feature_values` to use `Denormalizer.as_dict`
   with the wide-table-row → FeatureRecord adapter described in
   §4. Preserve `materialize_limit` and `execution_rids` via
   post-filters in the wrapper (with the regression notes
   documented).
2. Rewrite `DatasetBag.feature_values` analogously. Delete
   `BagFeatureCache` after verifying the bag-mode array /
   datetime divergence (§8) is zero against a real downloaded
   bag.
3. Leave `DerivaML.feature_values` untouched.

Test coverage required:
- Live test of `Dataset.feature_values` against dev-localhost
  showing FeatureRecord output is identical pre/post.
- Live test of `DatasetBag.feature_values` against a downloaded
  bag with the array/datetime divergence cases from §8.
- Regression test for the `execution_rids=[]` short-circuit
  (now Python-side instead of PathBuilder-server-side).

---

## 12. Distinct-API-vs-redundant assessment (Question K)

The two APIs serve **adjacent but distinct purposes**:

- **`feature_values`** is "read one feature's values, optionally
  reduced". Single feature in, typed FeatureRecord out. Idiomatic
  for ML code: a loop over annotations, a label lookup, a
  ground-truth retrieve.
- **`Denormalizer`** is "read a wide table joining multiple
  catalog entities". Multi-table in, dotted-label dict out.
  Idiomatic for analysis: a DataFrame for a ROC plot, a join
  across N executions' predictions, a join across feature and
  metadata.

They overlap on **exactly one use case**: "read one feature's
values within a dataset scope". Stage 1's selector parameter
closed the API-shape gap for that use case. Stages 2/3 then
ask whether the overlap should be deduplicated.

**Should one go away?** No. Each is the natural shape for its
own use case:

- A user wanting "the newest Glaucoma label per Image" reaches
  for `feature_values(... selector=select_newest)` — one
  feature, typed access, single-table mental model.
- A user wanting "Image, Subject, Glaucoma label, Diagnosis
  date, all in one row" reaches for
  `Denormalizer.as_dataframe(["Image", "Subject", ...])` — multi-table, dotted-column DataFrame.

The Stage 3 consolidation makes `Dataset.feature_values` a
**thin sugar over `Denormalizer`**, but the sugar stays because
the input/output shape is more ergonomic for the
one-feature-at-a-time case.

The reverse direction the user raised — "should
`Denormalizer.as_dict([target, feat_assoc], row_per=feat_assoc, selector=...)` be sugar that hides as
`feature_values(target, feature, selector=...)`?" — is exactly
backwards. The denormalize call is the implementation; the
`feature_values` call is the sugar. Keep both surfaces.

---

## 13. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bag-mode array/datetime divergence breaks an in-flight caller | Medium | Medium | Live verify before Stage 3 lands; document any new shape as a breaking change in changelog |
| `Dataset.feature_values` callers depend on `materialize_limit` raising before any work (memory protection) | Low | Low | Post-filter is no worse than the PathBuilder fetch the wrapper does today |
| `Dataset.feature_values` callers depend on `execution_rids` round-trip savings | Low | Medium | Document the new shape; provide `Denormalizer.as_dict` with `via=["Execution"]` if needed as escape hatch |
| Null target FK now emits a row instead of being silently dropped | Low | Low | Add a `target_col is not None` filter in the wrapper to match existing semantics |
| Loss of RCT in no-selector mode | High | Medium | Either always go through the selector path (with a passthrough `select_first`) or extend `as_dict` to optionally surface system columns |
| Performance: Dataset.feature_values gets faster on large catalogs (SQL join), slower on small (planner overhead) | Medium | Low | Benchmark on dev-localhost before deciding |
| Test debt: PR #257's selector path tests use in-memory SQLite; the bag-mode delegation has never been live-verified | High | High | Live verification on a real downloaded bag is the gating step for §11 |

---

## 14. Limitations

1. **No live verification.** Dev-localhost was unreachable from
   this environment (`curl ...catalog/27/schema` returned 000).
   All findings are from code reading. The biggest unverified
   claim is the bag-mode array / datetime divergence in §8 —
   that needs a real bag with the failure-mode columns to
   confirm or refute.
2. **PR #257 not on `origin/main`.** The Stage 1 code lives on
   `feat/denormalizer-selector` (commit `46836177`), not on
   `origin/main` (which is at `e8626feb`). The audit reads the
   PR diff directly. If the PR is reshaped before merge — for
   instance, if the supplementary-fetch RCT recovery is
   replaced by extending `_prepare_wide_table` to include
   system columns — §2 and §4 conclusions shift.
3. **No assessment of multi-feature selector.** PR #257
   explicitly punts on `selectors={feature_name: selector}`
   shape. Stage 3 delegation hits the same limit: if a Dataset
   caller wants "newest per Image for both Feature A and Feature
   B in the same call", they need either two `feature_values`
   calls or the future multi-feature selector extension. No
   regression from today, but it bounds the scope.
4. **No assessment of `update_navbar` /
   `apply_catalog_annotations` side effects.** Both
   `feature_values` and `Denormalizer` are read-only on the
   catalog, so this isn't an issue, but it's worth noting that
   the Stage 3 PR should not touch the annotation surface.
5. **`@validate_call` decorator behavior.** `feature_values` is
   decorated with `validate_call(config=VALIDATION_CONFIG)`;
   `Denormalizer.as_dict` is not. A delegation must either
   keep `@validate_call` on the outer `feature_values` (so input
   validation behavior is preserved) or replicate it on
   `Denormalizer.as_dict` — but the latter widens the
   denormalize contract.

---

## Appendix: File pointers

- `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/core/mixins/feature.py:377-529` — `DerivaML.feature_values`
- `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/dataset/dataset.py:597-687` — `Dataset.feature_values`
- `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/dataset/dataset_bag.py:537-650` — `DatasetBag.feature_values`
- `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/dataset/bag_feature_cache.py` — bag cache (the TEXT-storage divergence point)
- `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/feature.py:49-103` — `reduce_with_selector` shared helper
- `/Users/carl/GitHub/DerivaML/deriva-ml/src/deriva_ml/local_db/denormalize.py:195-433` (current `main`) — `_denormalize_impl` without selector
- PR #257 / commit `46836177` (`feat/denormalizer-selector`) — Stage 1 selector branch: `_apply_selector`, supplementary RCT fetch, datetime → ISO coercion
- `/Users/carl/GitHub/DerivaML/deriva-ml/docs/user-guide/denormalization.md` §6.2 Rule 7 — anchor classification semantics
- `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e/findings/investigation/01-denormalizer-spec-vs-implementation.md` §7 (aggregation gap), §H (routes-around) — original motivation for PR #257
