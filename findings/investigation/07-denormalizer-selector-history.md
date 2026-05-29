# 07 — Denormalizer Selector History Audit (Research-Only)

**Author:** audit agent
**Date:** 2026-05-29
**Branch:** `e2e-test/2026-05-28`
**Scope:** verify whether `Denormalizer` (and the
`Dataset.get_denormalized_as_*` / `DatasetBag.get_denormalized_as_*`
wrappers) ever had a `selector` parameter, why the rest of the
selector ecosystem has it, and what the right fix shape is.

---

## 1. TL;DR

- **Did `Denormalizer.as_dataframe` / `as_dict` ever have a
  `selector` parameter?** **No.** Verified via
  `git log --all -S "selector" -- src/deriva_ml/local_db/denormalize*.py`
  — zero hits, ever. Both files have 0 occurrences of the string
  `"selector"` at HEAD and have never carried it on any branch in
  history.
- **Did `Dataset.get_denormalized_as_dataframe` /
  `DatasetBag.get_denormalized_as_dataframe` ever have a
  `selector` parameter?** **No.** They have only ever existed as
  thin sugar wrappers over `Denormalizer.as_dataframe` /
  `as_dict`, introduced under the names
  `denormalize_as_dataframe` / `denormalize_as_dict` and renamed
  to `get_denormalized_as_*` by `4bcfaacf`
  (`refactor(denormalize): add Dataset/DatasetBag sugar methods;
  remove old names`, 2026-04-17). They have always passed only
  `include_tables`, `row_per`, `via`, `ignore_unrelated_anchors`,
  and (later) `version` through to the underlying `Denormalizer`.
- **Why does `feature_values` have selectors but `Denormalizer`
  doesn't?** **Organic divergence, not deliberate removal.** The
  two surfaces were built in different sprints, by different work
  streams, and **never connected**:
  - **Denormalize:** introduced 2025-10-24 (`e1588ff8`, "Initial
    cut at demormalization method in dataset"). At that point
    the selector concept did not exist in the codebase.
  - **Selector concept on features:** added 2026-03-03
    (`9626c374`, "Add fetch_table_features, selector support, and
    fix RID comparison bug") — on the *feature-reading* surface
    (`fetch_table_features` on FeatureMixin / DatasetBag,
    `FeatureRecord.select_newest`, then later `select_first`,
    `select_latest`, `select_majority_vote`, `select_by_execution`,
    `select_by_workflow`).
  - **Modern `feature_values(selector=)`:** added 2026-04-22 /
    2026-04-23 (`5c9ae303`, `ab8cd702`, `adce810c`, the S2 task
    sweep). At that point denormalize had already been renamed to
    its current shape (`4bcfaacf`, 2026-04-17) five days earlier
    and was not re-touched.
  - **`reduce_with_selector` helper extraction:** 2026-05-22
    (`3c5ce587`, PR #206, audit P1 F-8). Consolidated the
    group-by-RID + apply-selector pattern across exactly three
    `feature_values` sites (FeatureMixin, Dataset, DatasetBag).
    PR description names those three — denormalize is not in
    scope and is not mentioned.
- **Recommended fix shape:** add `selectors: dict[str, Callable]`
  (or, equivalently, `reduce_by: dict[str, FeatureSelector]`)
  to `Denormalizer.as_dataframe` / `as_dict` and propagate
  through the `Dataset` / `DatasetBag` wrappers. Map is keyed by
  feature-name (resolved by `Denormalizer._resolve_table_names`
  the same way `include_tables` is) → selector callable. See §6
  for the concrete signature.
- **Does the original finding 01 §7 proposal still stand?**
  Yes, with one refinement: the proposal called it `reduce_by=`,
  but the existing ecosystem term is `selector` / `selectors`.
  Aligning the new parameter name with `feature_values(selector=)`
  is a no-brainer for symmetry; the dict shape (per-feature, to
  cover multi-feature `include_tables`) is the only material
  decision left.

---

## 2. Selector ecosystem inventory (current state)

Every place selectors live today, with file:line:

| Surface | File | Line | Parameter |
|---|---|---|---|
| `reduce_with_selector` helper (shared) | `deriva_ml/feature.py` | 49 | `selector: Callable[[list[FeatureRecord]], FeatureRecord \| None]` |
| `FeatureRecord.select_newest` / `_first` / `_latest` / `_by_execution` / `_by_workflow` / `_majority_vote` | `deriva_ml/feature.py` | 165, 329, 355, 189, 248, 371 | classmethod/staticmethod selectors |
| `DatasetLike.feature_values` (abstract) | `deriva_ml/interfaces.py` | 247 | `selector: Callable[...] \| None = None` |
| `FeatureMixin.feature_values` (live catalog, on `DerivaML`) | `deriva_ml/core/mixins/feature.py` | — | `selector=None` |
| `Dataset.feature_values` (live catalog, scoped to a dataset) | `deriva_ml/dataset/dataset.py` | 601 | `selector=None` |
| `DatasetBag.feature_values` (bag-side) | `deriva_ml/dataset/dataset_bag.py` | 541 | `selector=None` |
| `_resolve_targets` (adapter shared helper) | `deriva_ml/dataset/target_resolution.py` | 78 | `targets: list[str] \| dict[str, FeatureSelector] \| None` |
| `DatasetBag.as_torch_dataset` | `deriva_ml/dataset/torch_adapter.py` | — | `targets: dict[str, FeatureSelector]` passes through |
| `DatasetBag.as_tf_dataset` | `deriva_ml/dataset/tf_adapter.py` | — | same |
| `DatasetBag.restructure_assets` | `deriva_ml/dataset/restructure.py` | — | `value_selector` (per-feature in dict form) |

All consumers of `selector` ultimately route through
`reduce_with_selector(records, target_col, selector)` (or, on
adapters, through `_resolve_targets` which itself calls
`feature_values(selector=...)`).

**Note:** `target_resolution.py` already defines the type alias

```python
FeatureSelector = Callable[[list["FeatureRecord"]], "FeatureRecord | None"]
```

— the same shape used everywhere else. This is the type a
hypothetical `Denormalizer.as_dataframe(selectors=)` would key
its dict values to.

---

## 3. Denormalizer surface inventory (confirm absence)

Every public method on the denormalize surface, with full
parameter list. `selector` is absent everywhere.

### `local_db/denormalizer.py:Denormalizer`

```python
def __init__(self, dataset: "DatasetLike", *, version: Any = None) -> None
def as_dataframe(self, include_tables: list[str], *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> pd.DataFrame
def as_dict(self, include_tables: list[str], *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> Generator[dict[str, Any], None, None]
def columns(self, include_tables: list[str], *,
    row_per: str | None = None,
    via: list[str] | None = None,
) -> list[tuple[str, str]]
def describe(self, include_tables: list[str], *,
    row_per: str | None = None,
    via: list[str] | None = None,
    ignore_unrelated_anchors: bool = False,
) -> DenormalizeDescription
def list_paths(...)
```

### `dataset/dataset.py:Dataset` (live-catalog sugar)

```python
def get_denormalized_as_dataframe(self, include_tables, *,
    row_per=None, via=None, ignore_unrelated_anchors=False,
    version=None,
) -> pd.DataFrame                                       # line 1651
def get_denormalized_as_dict(...)                       # line 1703
def list_denormalized_columns(...)                      # line 1749
def describe_denormalized(...)
```

### `dataset/dataset_bag.py:DatasetBag` (bag-side sugar)

```python
def get_denormalized_as_dataframe(...)                  # line 899
def get_denormalized_as_dict(...)                       # line 940
def list_denormalized_columns(...)
def describe_denormalized(...)
```

### `interfaces.py:DatasetLike` (abstract)

```python
def get_denormalized_as_dataframe(self, include_tables, *,
    row_per=None, via=None, ignore_unrelated_anchors=False,
) -> pd.DataFrame                                       # line 296
def get_denormalized_as_dict(...)                       # line 339
def list_denormalized_columns(...)                      # line 368
def describe_denormalized(...)
```

**Confirmed:** No `selector`, `selectors`, `reduce_by`, or
`reducer` parameter on any of these signatures.

---

## 4. Git archaeology

### 4.1 Direct searches on the denormalize files

```bash
$ cd deriva-ml
$ git log --all --oneline -S "selector" \
    -- src/deriva_ml/local_db/denormalize.py \
       src/deriva_ml/local_db/denormalizer.py
   # (zero commits returned)
```

```bash
$ grep -c "selector" \
    src/deriva_ml/local_db/denormalizer.py \
    src/deriva_ml/local_db/denormalize.py
src/deriva_ml/local_db/denormalizer.py:0
src/deriva_ml/local_db/denormalize.py:0
```

Definitive: the string `"selector"` has never appeared in either
`denormalize.py` or `denormalizer.py` on any branch in the
deriva-ml history. The `-G` regex pass for
`"selector|reducer|aggregate"` against `src/deriva_ml/local_db/`
returned three Phase 2 / ErmrestPagedClient commits — verified
the "aggregate" hits were ermrest URL fragments
(`/aggregate/{table}/n:=cnt(*)`), not feature-reduction logic.

### 4.2 Direct searches on the sugar-method files

```bash
$ git log --all --oneline -S "selector" -- src/deriva_ml/dataset/dataset.py
3c5ce587 refactor(feature): extract reduce_with_selector ... (#206)
ab8cd702 feat(dataset): feature_values/lookup_feature/list_workflow_executions (S2 Task 5)
```

Both commits added the `selector` parameter to `feature_values`,
not to `get_denormalized_as_*`. Manual inspection of
`get_denormalized_as_dataframe` at every revision in
`git log --oneline -- src/deriva_ml/dataset/dataset.py` confirms
no selector parameter has ever appeared on the denormalize
wrappers.

```bash
$ git log --all --oneline -S "selector" -- src/deriva_ml/dataset/dataset_bag.py
# (multiple commits, all touching feature_values / fetch_table_features /
# restructure_assets / adapters — none on the denormalize sugar methods)
```

### 4.3 Precursor / alternate-name searches

```bash
$ git log --all --oneline -S "denormalize_with_selector"   # 0 commits
$ git log --all --oneline -S "reduce_denormalized"         # 0 commits
$ git log --all --oneline -S "aggregate" -- src/deriva_ml/dataset/ src/deriva_ml/local_db/
   # only ermrest URL fragments (not feature reduction)
$ git log --all --oneline -S "reducer" -- src/deriva_ml/dataset/ src/deriva_ml/local_db/
   # 0 commits
```

No precursor name. No removed method named
`denormalize_with_selector`, `reduce_denormalized`,
`aggregate_denormalized`, etc.

### 4.4 Removed-method scan in `dataset.py`

`git log --diff-filter=D --name-only -p -- src/deriva_ml/dataset/dataset.py`
inspected; removed defs over history are:

| Removed name | Commit | Replaced by |
|---|---|---|
| `denormalize(dataset_rid: RID, columns: list[str])` | early — `e1588ff8` predecessor | absorbed into `denormalize_table` |
| `denormalize_table(table: str \| Table)` | `0b04262d` | `denormalize_table_as_dataframe` + `denormalize_table_as_dict` |
| `denormalize_as_dataframe`, `denormalize_as_dict`, `denormalize_columns`, `denormalize_info` | `4bcfaacf` (2026-04-17) | `get_denormalized_as_dataframe`, `get_denormalized_as_dict`, `list_denormalized_columns`, `describe_denormalized` |

Each rename preserved the parameter list. None of the
removed-method signatures carried a `selector`-like parameter.

### 4.5 The key sprint sequence

| Date | Commit | What it added | Where |
|---|---|---|---|
| 2025-10-24 | `e1588ff8` | initial `denormalize()` | `dataset.py` |
| 2026-03-03 | `9626c374` | `fetch_table_features(... selector=)` + `FeatureRecord.select_newest` | feature mixin, FeatureRecord |
| 2026-03-06 | `c865a507` | `select_majority_vote`, `select_latest`, `select_first` for `restructure_assets` | dataset module |
| 2026-04-17 | `4bcfaacf` | rename `denormalize_*` → `get_denormalized_*` (no parameter changes) | dataset.py, dataset_bag.py |
| 2026-04-22 | `5c9ae303` | `DerivaML.feature_values(... selector=)` — modern iterator API | feature mixin |
| 2026-04-23 | `ab8cd702` | `Dataset.feature_values(... selector=)` | dataset.py |
| 2026-04-23 | `adce810c` | `DatasetBag.feature_values(... selector=)` | dataset_bag.py |
| 2026-04-23 | `d65bbcbc` | retire `fetch_table_features` (S2 Task 10) | feature mixin |
| 2026-05-14 | `27c0502b` | retirement shims for `fetch_table_features` / `list_feature_values` | dataset_bag.py |
| 2026-05-22 | `3c5ce587` (#206) | extract `reduce_with_selector` helper across the three `feature_values` sites | feature.py + 3 sites |
| 2026-05-26 | `cf135864` (#228) | resolve feature names in denormalize `include_tables` | `_resolve_table_names` in denormalizer |
| 2026-05-28 | `2c239942` (#254) | `split_dataset(partition_by=)` with within-element uniformity check | split_dataset (not denormalize itself) |

**The denormalize surface stabilised three weeks *before* the
modern `feature_values(selector=)` shape was introduced.** PR
#206's reduce-with-selector consolidation explicitly named only
the three `feature_values` sites; the denormalize surface was
not in scope, was not mentioned, and was not touched.

### 4.6 PR #206 verification (the closest "should this have touched
denormalize?" moment)

PR #206 description (verbatim, abridged):

> Three `feature_values` implementations carried the same
> group-by-RID + apply-selector reduction inline:
> - `core/mixins/feature.py:FeatureMixin.feature_values`
> - `dataset/dataset.py:Dataset.feature_values`
> - `dataset/dataset_bag.py:DatasetBag.feature_values`
>
> All three now delegate to one helper:
> `deriva_ml.feature.reduce_with_selector(...)`.

The denormalize surface is not in the "three implementations"
list. The PR was a duplicate-code-elimination on the existing
`feature_values` callers; it was not an audit of "where else
should this selector pattern apply?" The denormalize surface is
the most natural fourth site, but no commit ever proposed
adding it there.

---

## 5. Why `feature_values` has selectors and `Denormalizer` doesn't

**Verdict: organic divergence, not deliberate removal.** The
evidence:

1. **No removed selector parameter on the denormalize path.** §4
   confirms zero commits with `-S "selector"` against the
   denormalize files, and zero removed signatures with
   selector-shaped parameters in `dataset.py`'s `--diff-filter=D`
   history.
2. **The two work streams never crossed.** Denormalize-related
   commits (e.g. `4bcfaacf` on 2026-04-17) and selector-related
   commits (e.g. `5c9ae303` on 2026-04-22) ran in adjacent but
   non-overlapping sprints. The selector ecosystem grew up
   around the *single-feature read* surface (`feature_values`)
   and the *adapter* surface (`as_torch_dataset`,
   `as_tf_dataset`, `restructure_assets`) — both of which read
   feature records one-at-a-time and need a callable to collapse
   multi-record groups. Denormalize, by contrast, expresses the
   multi-feature reduction question through table-graph
   semantics: it joins, it propagates FKs, and Rule 5 (the
   "no aggregation" rule) explicitly says it does *not*
   reduce — it refuses to combine `row_per=X` with anything
   downstream of `X`.
3. **The audit-driven consolidation (PR #206) was scoped to the
   `feature_values` trio.** The PR author saw the three nearly
   identical inline group-and-reduce blocks and pulled them out
   into one helper. They did not ask "is there a fourth site
   that should adopt this pattern?" — `Denormalizer.as_dataframe`
   does not have an inline group-and-reduce block, so it wasn't
   on the duplicate-code radar.
4. **Finding 01 §7 is the first time anyone wrote down the
   question.** The 2026-05-28 e2e audit (this branch, finding
   01) is the first artefact in either repo to explicitly say
   "the denormalize surface should grow a `reduce_by=` /
   `select=` knob to mirror `feature_values`." It was filed as
   a contract gap, not a fix.

The user's recollection ("there were reducer functions that you
could pass in for common aggregations as well as a user
specified aggregation") is consistent with the
`FeatureRecord.select_*` suite + `feature_values(selector=)`
pattern that exists today on `feature_values` and on adapters.
It is **not** consistent with that pattern ever having been on
the denormalize surface — the suite did exist (since 2026-03-03
onward), and was passable to `feature_values` and to
`restructure_assets`, but the denormalize surface was never
wired up to accept it.

---

## 6. Recommended fix shape

The natural design is to mirror the existing `feature_values`
selector shape, generalised across multiple features in a single
denormalize call. Concrete signature:

```python
# src/deriva_ml/local_db/denormalizer.py

from deriva_ml.feature import FeatureRecord  # for the type alias

FeatureSelector = Callable[[list["FeatureRecord"]], "FeatureRecord | None"]

class Denormalizer:
    def as_dataframe(
        self,
        include_tables: list[str],
        *,
        row_per: str | None = None,
        via: list[str] | None = None,
        ignore_unrelated_anchors: bool = False,
        selectors: dict[str, FeatureSelector] | None = None,
    ) -> pd.DataFrame:
        """...
        Args:
            ...
            selectors: Optional per-feature reducers. Maps a
                feature name (the same shorthand accepted by
                ``include_tables`` since #228) to a selector
                callable of the form
                ``(list[FeatureRecord]) -> FeatureRecord | None``.
                When a feature in ``include_tables`` is keyed in
                this dict, its multi-value groups (one per
                ``row_per`` element) are collapsed by calling the
                selector on each group before joining; selectors
                that return ``None`` drop the element. Features
                not in the dict pass through unchanged (preserving
                the current contract — Rule 5 applies, downstream
                feature-association tables remain forbidden as
                ``row_per`` candidates). Use
                ``FeatureRecord.select_newest`` etc. for the
                pre-built suite.
        """
```

The same parameter propagates verbatim through:

- `Denormalizer.as_dict(..., selectors=...)`
- `Dataset.get_denormalized_as_dataframe(..., selectors=...)`
- `Dataset.get_denormalized_as_dict(..., selectors=...)`
- `DatasetBag.get_denormalized_as_dataframe(..., selectors=...)`
- `DatasetBag.get_denormalized_as_dict(..., selectors=...)`
- `DatasetLike.get_denormalized_as_dataframe(..., selectors=...)`
- `DatasetLike.get_denormalized_as_dict(..., selectors=...)`

### 6.1 Why `selectors` (plural dict) over `selector` (single callable)

The denormalize surface is **multi-feature by construction** —
`include_tables` can name two or three features simultaneously
(e.g. `["Image", "Diagnosis", "Confidence"]`). A single
top-level `selector=` is ambiguous in that case: which
feature's groups does it reduce? The dict form is the
unambiguous, future-proof shape and is the one already used by
`_resolve_targets` (`targets: dict[str, FeatureSelector]`) on
the adapter surface — symmetric with what callers already learn
from `as_torch_dataset` / `as_tf_dataset`.

If the per-feature granularity is overkill for the common case,
add `selector` (singular) as **sugar** that auto-broadcasts to
every feature in `include_tables`. The dict shape is the
load-bearing one; the singular shape is a convenience.

### 6.2 Why "selectors" not "reduce_by"

The audit's finding 01 §7 wrote it as `reduce_by=`. The
ecosystem term, everywhere else, is `selector` (singular on
`feature_values`, plural / dict on adapter `targets`). Pick the
ecosystem term. The dict-of-selectors shape (one per feature
name) is the same shape `_resolve_targets` already accepts, so
callers don't need to learn a new vocabulary.

### 6.3 Relationship to PR #254's `partition_by="element"`

PR #254's `split_dataset(partition_by="element")` does
**post-denormalize dedupe** with a **within-element uniformity
check** — it requires that all rows belonging to the same
element-table RID agree on the stratification column, and
raises on disagreement. That's a *refusal-to-reduce*. A
hypothetical `Denormalizer.as_dataframe(selectors=...)` would
be a *willingness-to-reduce-by-rule* — the caller picks the
rule (newest, majority vote, by-workflow, custom), and
disagreement is resolved by the rule rather than raised.

These are **complementary, not redundant**. `partition_by` is
a partition-policy knob on `split_dataset`. `selectors` would
be a row-reduction knob on the denormalize surface. The right
composition: `split_dataset` would call
`Denormalizer.as_dataframe(selectors=...)` internally when the
caller wants a per-element reduced dataframe, and would skip
the uniformity check in that mode because the selector defines
the resolution. The current `partition_by="element"`
implementation can stay (no-selector callers still want the
uniformity check), with an explicit "or pass a selector" branch
added later.

---

## 7. Risk assessment

### 7.1 Semantic risk: Rule 5 interaction

Rule 5 forbids combining `row_per=X` with anything strictly
downstream of `X`. A `selectors` parameter changes the contract
for features only — the feature-association table becomes
*reduced* rather than *forbidden*, but only when the caller
explicitly opts in via `selectors[<feature_name>] = <callable>`.
Default behaviour (no `selectors` arg) is unchanged, so Rule 5
keeps firing for existing callers. No silent semantics change.

### 7.2 Interaction with feature-name shorthand (#228)

Since `cf135864` (#228, 2026-05-26), `include_tables` accepts
feature names like `"Image_Classification"` and resolves them
to the underlying feature-association table internally. The
`selectors` dict key should resolve the same way (feature-name
shorthand → resolved feature table), via the existing
`Denormalizer._resolve_table_names` helper. Consistent UX,
single source of truth for the name → resolved-table map.

### 7.3 Performance risk

The reducer runs in Python per element (same complexity profile
as `feature_values(selector=)`), but on the denormalize path
the records were materialised by the SQL join, not by a Python
loop. For very large datasets, the dataframe materialisation
cost dominates the per-row callable cost — same trade-off as
`feature_values` callers already accept.

### 7.4 Testability

`reduce_with_selector` is already extracted and unit-tested
(PR #206). Wiring the denormalize path through the same helper
keeps the reduction semantics in one place. The new tests are
"does `Denormalizer.as_dataframe` correctly pass each feature's
records through `reduce_with_selector` and join the result" —
straightforward to mock.

### 7.5 Backwards compatibility

The new parameter is keyword-only and defaults to `None`. All
existing callers are unchanged. The protocol (`DatasetLike`)
gains a new keyword on three methods; concrete implementations
(`Dataset`, `DatasetBag`) gain the same. Outside callers
relying on the protocol's existing shape don't break.

---

## 8. Open questions

1. **Multi-target FK columns.** For a feature on `Image` with
   a multi-target FK (e.g. `Image_RID` and `Subject_RID` both
   present), what does "the feature's target RID" mean for
   grouping? `reduce_with_selector` takes a `target_col`
   parameter to disambiguate; the denormalize helper would need
   to infer this from the feature's schema (the "anchor"
   column). The infrastructure for this resolution exists in
   the feature mixin's `feature_values` — port the same
   resolution path.
2. **Selectors that return `None`.** On `feature_values`,
   `None` drops the element. On the denormalize path, what
   happens to the joined row? Two reasonable choices:
   - Drop the row entirely (matches `feature_values` and is
     least surprising).
   - Emit the row with the feature's columns set to NaN /
     orphan-row semantics (matches `ignore_unrelated_anchors`
     style).
   The first is simpler and matches the established selector
   contract; recommend it.
3. **Sugar singular form.** Should `Denormalizer.as_dataframe`
   also accept `selector=<callable>` (singular) that
   auto-broadcasts to every feature in `include_tables`? Easy
   to add later; not load-bearing.
4. **Should `split_dataset(partition_by="element", selectors=)`
   defer the uniformity check?** Yes — when the caller passes
   a selector, the selector defines the resolution and the
   uniformity check is the wrong default. Implement after the
   denormalize-side parameter lands; not a blocker for the
   denormalize change itself.

---

## 9. Limitations

1. I did not run the deriva-ml test suite during this audit —
   it is a research-only analysis and all conclusions are
   sourced from `git log`, `git show`, and `grep` on the
   working tree.
2. I did not exhaustively read every revision of every commit
   in the §4.5 timeline — I confirmed (a) the strings
   `selector`, `reducer`, `aggregate` were absent from the
   denormalize files at every revision (via `-S`), and (b) the
   high-traffic commits' subject lines and diffs were
   consistent with the timeline I describe. I did not
   side-by-side every intermediate revision.
3. The "recommended fix shape" in §6 is a design proposal, not
   an implementation. It has been sanity-checked against the
   existing ecosystem types (`FeatureSelector`,
   `reduce_with_selector`, `_resolve_table_names`, the #228
   feature-name resolution) but not prototyped.
4. The interaction with `describe_denormalized` is not
   analysed. If `selectors` is added to `as_dataframe`,
   `describe` should probably surface the reduction in its
   metadata output (e.g. "Image_Classification: reduced by
   `select_newest`"), but the exact shape is out of scope for
   this audit.
