# `_is_feature_association` predicate audit — 3-FK assumption vs real 4-FK features

**Investigator:** feature-association predicate audit (research-only)
**Date:** 2026-05-29
**Scope:** `deriva_ml.model.denormalize_planner.DenormalizePlanner._is_feature_association`
and its blast radius (selector path, transparency/reachability, Rule 6).
**Code under test:** worktree
`/Users/carl/GitHub/deriva-ml-feature-values-stage3a` (branch
`refactor/feature-values-delegate-stage3a`, editable install
`1.40.2.post5+g25b9f4ab8`). This worktree carries the **current**
predicate + the Stage 1 selector + Option B `reduce_with_selector`,
so it is the right surface to test the selector blast radius. The
predicate body is byte-identical to `main`
(`/Users/carl/GitHub/DerivaML/deriva-ml`) — both have the `!= 3`
check at `denormalize_planner.py:455`.
**Live target:** dev-localhost catalog 27, schema `e2e-test-20260528`,
feature table `Execution_Image_Image_Classification`, dataset `M16`
(550 training images). MCP server authenticated this session.
**Mode:** No code changes shipped. Every runtime claim below was
reproduced with throwaway scripts run against catalog 27 (since
deleted). A relaxed-predicate monkeypatch was used to verify the fix
behaves; nothing was written to the catalog or committed to deriva-ml.

---

## 1. TL;DR

- **The predicate is broken.** `_is_feature_association` requires
  **exactly 3 domain FKs** (`denormalize_planner.py:455`,
  `if len(domain_fks) != 3: return False`). The real DerivaML feature
  table on catalog 27 has **4 domain FKs**:
  `['Execution', 'Image', 'Feature_Name', 'Image_Class']`. Live call
  returns **`False`**. `_is_transparent_intermediate` returns
  `False` too. `DerivaModel.find_features` (the DerivaML-native
  predicate) correctly recognizes the same table — so the planner and
  the rest of deriva-ml disagree about what a feature is.

- **The selector path (Stage 1) is DOA on real feature tables.**
  `as_dict(include_tables=["Image","Execution_Image_Image_Classification"],
  row_per=..., selector=FeatureRecord.select_newest)` **raises**
  `ValueError: "selector requires a feature-association table in
  include_tables; none found"` (gate at `denormalize.py:310-317`,
  driven by the broken predicate). It does **not** silently ignore
  the selector — it hard-fails. This is true both when passing the
  feature-association table name and (because the resolver substitutes
  the same table) when passing the feature *name* `Image_Classification`.
  Stage 1's selector is unreachable on any real catalog with the
  current predicate.

- **Transparency does NOT break — and it never depended on the
  predicate.** `as_dataframe(include_tables=["Image","Image_Class"],
  row_per="Image")` **works** (returns 800 rows, projects
  `Image_Class.Name` etc. as per-image columns). It works through a
  *different* mechanism than `_is_feature_association`: the
  materialization planner (`_prepare_wide_table` Phase 1) walks the
  raw FK graph via `_schema_to_paths()` and keeps any path that
  *ends* at a requested table and *contains* a requested table — the
  interior feature-assoc bridge is never predicate-checked there. So
  the "project the vocab term as a per-image column" story (finding 01
  §B/§H) is alive and uses `_resolve_table_names` + raw path discovery,
  **not** the transparency predicate.

- **`Feature_Name` was never "added late."** The predicate was **born
  wrong**. `Feature_Name` has been an FK in the feature-association
  table since 2025-06-04 (commit `484f3b41`); the `!= 3` predicate was
  introduced 2026-05-20 (PR #174/#176, commit `4cb7fd5c`), ~11 months
  later. The author modeled "3 FKs" (target/value/Execution) and
  forgot the `Feature_Name` FK that every real feature carries.

- **Recommended fix: key off the `Feature_Name` FK.** Define a
  feature-association table as one carrying **an FK to the
  `Feature_Name` vocabulary in the ML schema AND exactly one FK to
  `Execution` in the ML schema** (drop the FK-count equality). This
  mirrors `find_features.is_feature` exactly, has **zero false-positive
  risk** against the existing `FourWayAssoc` fixture (which has 4 FKs +
  Execution but no `Feature_Name` FK), and verified live to fix the
  selector path with no Rule 6 regression on catalog 27.

- **Rule 6 risk of the fix: low but non-zero, and only on richer
  schemas.** Making the table transparent re-enables the
  feature-bridge hop in `_is_downstream_chain`
  (`denormalize_planner.py:1306`). On a schema where a target and a
  value table are *also* connected by a second FK path, the now-hopped
  feature bridge could newly register as a competing downstream chain
  and surface a `DerivaMLDenormalizeAmbiguousPath` that doesn't fire
  today. Catalog 27 has only one path, so no regression was observed;
  the risk is latent on diamond-shaped domain schemas.

- **Test gap (why it shipped): the fixture lies.** The synthetic
  fixture `tests/local_db/conftest.py:524-577` declares `Feature_Name`
  as a **plain text column**, not an FK, and gives the table exactly
  **3 FKs**. `test_planner_rules.py:231` asserts the predicate returns
  `True` against that 3-FK fixture — green. The fixture matches the
  predicate's wrong mental model instead of the real schema, so the
  bug was invisible to CI. Same fixture-coverage-gap pattern as the
  earlier audits in this fix-pass.

---

## 2. Predicate confirmation (live)

### 2.1 Live FK count (catalog 27, `list_foreign_keys`)

`Execution_Image_Image_Classification` outbound FKs:

| FK column     | → table                  | role            |
|---------------|--------------------------|-----------------|
| `Image`       | `e2e-test….Image`        | feature target  |
| `Image_Class` | `e2e-test….Image_Class`  | value (vocab)   |
| `Feature_Name`| `deriva-ml.Feature_Name` | feature-name vocab |
| `Execution`   | `deriva-ml.Execution`    | provenance      |
| `RCB`         | `public.ERMrest_Client`  | system (excluded) |
| `RMB`         | `public.ERMrest_Client`  | system (excluded) |

**6 outbound FKs total; 4 domain FKs** after dropping the
ERMrest_Client/ERMrest_Group system edges (the same exclusion the
predicate applies at `denormalize_planner.py:385,454`).

### 2.2 The `!= 3` check (code)

`denormalize_planner.py:451-461`:

```python
fks = list(tbl.foreign_keys)
domain_fks = [fk for fk in fks if fk.pk_table.name not in ("ERMrest_Client", "ERMrest_Group")]
if len(domain_fks) != 3:
    return False
ml_schema = self.model.ml_schema
execution_fks = [
    fk for fk in domain_fks if fk.pk_table.name == "Execution" and fk.pk_table.schema.name == ml_schema
]
return len(execution_fks) == 1
```

`len(domain_fks) == 4`, so it returns `False` at the first guard. The
Execution check is never reached.

### 2.3 Live call (Question A — answered)

Constructed the real planner against catalog 27
(`ml = DerivaML("localhost", "27", domain_schemas={"e2e-test-20260528"})`,
`p = ml.model._planner`):

```
domain FKs          = 4 -> ['Execution', 'Image', 'Feature_Name', 'Image_Class']
_is_feature_association('Execution_Image_Image_Classification') = False
_is_transparent_intermediate(...)                               = False
```

The DerivaML-native discovery agrees the table IS a feature:

```
find_features('Image') count = 1
  feature_table=Execution_Image_Image_Classification target=Image name=Image_Classification
```

`find_features.is_feature` (`model/catalog.py:645-657`) checks that
`{"Feature_Name", "Execution", <target-FK>}` is a subset of the
table's columns — a **column-presence** test, not an FK-count test.
That predicate has been correct since 2025; the planner predicate is
the outlier.

**General shape, not a catalog-27 quirk:** catalog 27 has exactly one
feature, and it has 4 domain FKs. The feature-table builder
(`core/mixins/feature.py:177-184`) constructs the association via
`_define_association(associates=[execution, target_table,
feature_name_table], metadata=[…terms…asset…value])`. Any feature with
a single value column therefore has **≥4 domain FKs** by construction
(target + Feature_Name + Execution + ≥1 value/term/asset FK). The
3-FK shape the predicate expects **cannot be produced** by
`create_feature`.

---

## 3. Selector path impact (Question B — reproduced)

Gate in `local_db/denormalize.py:308-323`:

```python
feature_assoc_table: str | None = None
if selector is not None:
    feature_assoc_tables = [t for t in include_tables if model._planner._is_feature_association(t)]
    if not feature_assoc_tables:
        raise ValueError("selector requires a feature-association table in include_tables; "
                         f"none found in {include_tables!r}. ...")
```

Live, against `Denormalizer(Dataset(ml, "M16"))`:

```
as_dict(include_tables=["Image","Execution_Image_Image_Classification"],
        row_per="Execution_Image_Image_Classification",
        selector=FeatureRecord.select_newest)
→ ValueError: selector requires a feature-association table in include_tables;
  none found in ['Image', 'Execution_Image_Image_Classification']. ...
```

**Answer: the selector path RAISES (does not silently ignore).** The
list comprehension at line 310 yields `[]` because the predicate is
`False` on the only feature-assoc table in the request, so the
"none found" branch fires. Stage 1's selector is **dead on every real
DerivaML catalog**, not just on some shapes — there is no spelling of
`include_tables` (feature-assoc table name, or feature name, which the
resolver maps to the same feature-assoc table) that gets past the
gate, because the gate re-checks the resolved name with the same
broken predicate.

This is independent of `feature_values()`'s own selector path (which
does not route through this gate — see finding 08); the regression is
specific to `Denormalizer.as_dict/as_dataframe(selector=...)`, the
Stage 1 surface.

---

## 4. Transparency / reachability impact (Question C — the deep one, reproduced)

This is the surprising result: **the predicate is False, yet the
canonical transparency call works.** The two facts are reconciled by
the planner using two *different* path engines.

### 4.1 The transparency primitives DO break

```
_outbound_reachable('Image',       {'Image','Image_Class'}) = set()
_outbound_reachable('Image_Class', {'Image','Image_Class'}) = set()
_enumerate_paths('Image','Image_Class',{'Image','Image_Class'}) = []
```

`_enumerate_paths` (`denormalize_planner.py:1141-1155`) calls the raw
DFS `_schema_to_paths`, which *does* find the bridge path —

```
_schema_to_paths(root=Image, stop_at=Image_Class) raw
  ['Image', 'Execution_Image_Image_Classification', 'Image_Class']
```

— but then applies a **transparency filter** at line 1153: every
interior node must be in `tables_in_set` OR
`_is_transparent_intermediate(...)`. The interior
`Execution_Image_Image_Classification` is neither (predicate is
`False`), so the path is dropped → empty result. Likewise
`_outbound_reachable` (lines 666-703) only chains through neighbors
that pass `_is_transparent_intermediate`, so the bridge is never
hopped. **The predicate genuinely breaks these primitives.**

### 4.2 But the materialization path uses a different engine and works

`Denormalizer.as_dataframe(["Image","Image_Class"], row_per="Image")`
on dataset M16 returns a real wide table:

```
shape = (800, 12)
columns = ['Image_Class.RID','Image_Class.ID','Image_Class.URI','Image_Class.Name',
           'Image_Class.Description','Image_Class.Synonyms',
           'Image.RID','Image.URL','Image.Filename','Image.Description',
           'Image.Length','Image.MD5']
# row 0: Image_Class.Name='truck', Image.RID='47Y', ...
```

`Image_Class.Name` projects as a per-row column exactly as finding 01
§H described. Tracing `_prepare_wide_table` (the method that actually
builds the join) on the same inputs:

```
join_tables['Image'].path =
  ['Dataset','Dataset_Image','Image','Execution_Image_Image_Classification','Image_Class']
column_specs tables = ['Image','Image_Class']
```

The feature-assoc table **is** in the final join path. Why? Because
`_prepare_wide_table` Phase 1 (`denormalize_planner.py:1705-1714`)
enumerates paths with raw `_schema_to_paths()` and keeps any path
whose **endpoint** is in `include_tables` and which **contains** a
requested table:

```python
all_paths = self._schema_to_paths()
table_paths = [
    path for path in all_paths
    if path[-1].name in include_tables_set
    and include_tables_set.intersection({p.name for p in path})
]
```

There is **no `_is_feature_association` / `_is_transparent_intermediate`
gate** in this filter. The interior feature-assoc table rides along
because the path ends at `Image_Class` (requested) and contains
`Image` (requested). So the wide-table build reaches the vocab term
**regardless** of the predicate.

The 800 (not 550) row count is the fan-out: some training images have
more than one `Image_Classification` annotation row (multi-execution
/ duplicate labels — the same multiplicity the Curator/Evaluator
flagged for the train/test leakage). One row per *feature annotation*,
not per image. Collapsing to one row per image is exactly what the
**selector** is for — and the selector is the path that's dead
(§3).

### 4.3 Answer to Question C

- Is the feature table treated as a transparent bridge? **No** in
  `_outbound_reachable` / `_enumerate_paths` (predicate is False), but
  the materialization planner doesn't consult those for include-table
  path selection, so transparency-as-observed survives.
- Does `get_denormalized_as_dataframe(["Image","Image_Class"],
  row_per="Image")` work on catalog 27? **Yes** — reproduced, 800×12,
  `Image_Class` projects as columns.
- Does the 4-FK shape break it? **No.** The working path never went
  through the predicate.

**Conclusion:** the predicate being False does **not** break the
documented transparency story. The denormalize-through-feature-tables
behavior works through `_resolve_table_names` (feature-name → table)
plus raw `_schema_to_paths` endpoint filtering. The predicate is
**dead/ineffective on the materialization path** and **actively
wrong (returns False where it should return True) on the
reachability/selector/Rule-6 paths**. It is simultaneously dead code
*and* a latent bug, depending on which call site you look at.

---

## 5. Reconciliation with finding 01 (Question C cross-check)

Finding 01 §B/§H described the feature table as a "**3-FK**
feature-association" (`01-…md` lines 32, 231:
`Execution_Image_Image_Classification (FK→Image, FK→Image_Class,
FK→Execution)`). That enumeration is **incomplete** — it omits the
`Feature_Name` FK. The live catalog had 4 domain FKs at the time
finding 01 was written too (Feature_Name has been an FK since
2025-06-04; see §6). Finding 01 did not call `_is_feature_association`
directly; it reasoned about Rule 5 via `_outbound_reachable_strict`,
which keys off the FK *pointing at Image* (present regardless of
arity), so its conclusions about Rule 5 stand.

The transparency behavior finding 01 documented (§H: `["Image",
"Image_Class"]` succeeds and projects the vocab term) does **NOT**
depend on `_is_feature_association` returning True — confirmed live in
§4.2: it works today even though the predicate is False, via the raw
`_schema_to_paths` endpoint filter in `_prepare_wide_table`. So
finding 01's "this shape works" was correct, but its implied
mechanism (transparent-bridge hop) was not the one in play.
Transparency works through `_resolve_table_names` + raw path
discovery, exactly as this audit's §4.2 shows.

**Bottom line for the reconciliation:** option (b) — "the transparency
works through a different code path" — is the correct branch. The
predicate was *not* returning True at finding-01 time (the schema was
already 4-FK); the working behavior never relied on it.

---

## 6. `Feature_Name`-as-FK history (Question D)

| Event | Commit | Date |
|-------|--------|------|
| `Feature_Name` present in feature-assoc `is_feature` predicate (`find_features`) | `484f3b41` "Renamed filed." | **2025-06-04** |
| `_is_feature_association` + `len(domain_fks) != 3` introduced | `4cb7fd5c` PR #174/#176 "feature-assoc tables are transparent" | **2026-05-20** |

`Feature_Name` as an FK predates the planner predicate by ~11.5
months. The feature builder (`core/mixins/feature.py:177-184`) has
always added `feature_name_table` to the `associates` list of
`_define_association`, so the FK was structural from the start. This
is **not a regression introduced by a later schema change** — the
predicate was written against a wrong mental model of the feature
shape (3 FKs) that never matched what `create_feature` produces.

(`git log -S` was run in the stage3a worktree, which shares full
history with `main`.)

---

## 7. Correct-predicate options (Questions E + F)

The canonical real-feature FK set is
`{target, value(s), Feature_Name→Feature_Name(ML vocab),
Execution→Execution(ML)}` — minimum 4 domain FKs. The discriminators
available are: FK count, presence of an Execution FK, and presence of
a `Feature_Name` FK.

### Option E1 — relax to `>= 3` domain FKs with exactly one Execution FK

```python
if len(domain_fks) < 3: return False
return len(execution_fks) == 1
```

- **False-negative:** fixed for real features (4 FKs + Execution → True).
- **False-positive: HIGH.** The existing `FourWayAssoc` fixture
  (`conftest.py:631-668`) has FKs `{Image, Subject, UnrelatedThing,
  Execution}` — 4 domain FKs, one Execution FK, **no Feature_Name FK**.
  E1 would classify it as a feature-assoc → **breaks
  `test_four_fk_assoc_is_not_transparent`** and would silently treat a
  genuine multi-way domain association (with an audit edge) as a
  transparent bridge. Any domain 3+-way association that happens to
  carry an Execution provenance FK becomes a false feature.
- **Rule 6:** worsens the false-positive surface for diamond
  detection. **Reject.**

### Option E2 — explicit canonical shape: require Feature_Name FK + Execution FK (recommended)

```python
ml_schema = self.model.ml_schema
execution_fks  = [fk for fk in domain_fks
                  if fk.pk_table.name == "Execution"     and fk.pk_table.schema.name == ml_schema]
feature_name_fks = [fk for fk in domain_fks
                  if fk.pk_table.name == "Feature_Name"  and fk.pk_table.schema.name == ml_schema]
return len(execution_fks) == 1 and len(feature_name_fks) == 1
```

(Equivalently "≥3 domain FKs with exactly one Execution and exactly
one Feature_Name FK"; the count guard becomes redundant once both
required FKs are present, since target + value make ≥4.)

- **False-negative: none** against any `create_feature`-produced
  table — every such table has exactly these two FKs by construction.
  Verified live: returns `True` on
  `Execution_Image_Image_Classification`.
- **False-positive: none** against the existing fixtures.
  `FourWayAssoc` lacks a `Feature_Name` FK → `False` (preserves
  `test_four_fk_assoc_is_not_transparent`'s intent, though that test's
  *rationale* — "4-FK is ambiguous" — would need rewording to
  "4-FK *without* a Feature_Name FK"). `Image_Subject_UnrelatedThing`
  (no Execution, no Feature_Name) → `False`. `Dataset_Image` (2 FK) →
  `False`.
- **Matches `find_features.is_feature`** (`model/catalog.py:653-657`),
  unifying the two predicates on the same definition of "feature"
  rather than leaving the planner with a private, wrong one.
- **Rule 6:** see §7.1.

### Option E3 — key solely off the Feature_Name FK presence (+ Execution)

Functionally identical to E2 here. The only nuance: whether to *also*
require the Execution FK. Requiring both (E2) is safest — a table with
a `Feature_Name` FK but no Execution edge would be a malformed feature
and should not be silently treated as a transparent provenance bridge.
**E2 ≡ E3-with-Execution-guard is the pick.**

### 7.1 Rule 6 dependency and risk (Question F)

`_is_feature_association` feeds `_is_transparent_intermediate`, which
Rule 6 (`_find_path_ambiguities`) consults in three spots:

1. **`_is_downstream_chain`** (`denormalize_planner.py:1306`): if an
   interior node is transparent, the walk hops across it (`i += 2`)
   and treats the `A → bridge → C` span as one downstream step.
2. **`_is_signaled`** (line 1338): transparent intermediates do NOT
   count as user path-signals.
3. **`suggested_intermediates`** (line 1352): transparent nodes are
   not suggested as disambiguators.

**Current (broken-predicate) behavior:** because the feature-assoc
table is non-transparent AND its FK *points at* the target (so the
`target → feat` edge is direction "up", not "down"),
`_is_downstream_chain` rejects any path through it. The feature bridge
is therefore invisible to diamond detection — which is *why*
`_find_path_ambiguities(row_per="Image", ["Image","Image_Class"])`
returns `[]` today and the wide-table call plans cleanly.

**After E2 fix (verified live):** the predicate returns True, the
bridge becomes hoppable, and `_outbound_reachable('Image', S)` now
returns `{'Image_Class'}`. On catalog 27,
`_find_path_ambiguities` still returns `[]` (only one path exists), so
**no false ambiguity** — reproduced. The risk surfaces only on a
schema where the target and the value table are *also* joined by a
second FK path: the now-transparent feature bridge becomes a competing
downstream chain and could raise `DerivaMLDenormalizeAmbiguousPath`
where it previously planned silently. That is arguably *correct*
behavior (two real ways to relate the tables → ask the user), but it
is a behavior change, and it is the reason the fix is not a one-liner
to merge blind. Catalog 27 cannot exhibit it; a multi-feature or
diamond-domain catalog could.

The fix does **not** change behavior in the reverse direction
(mis-classifying a real domain diamond as a feature bridge), because
E2 requires a `Feature_Name` FK that genuine domain associations don't
have.

---

## 8. Test coverage gap (Question G)

The synthetic fixture **does not model the real schema in the one
dimension the predicate measures.**

`tests/local_db/conftest.py:524-577` builds
`Execution_Image_Image_Classification` with:
- `Feature_Name` declared as `{"name":"Feature_Name","type":
  {"typename":"text"}}` — a **plain text column, not an FK** (line 537).
- Exactly **3 foreign keys**: `Image`, `Execution`,
  `Image_Classification` (lines 543-575).

The comment at line 525 even states the intent: "Three domain FKs:
Image, Execution, Image_Classification." Line 526 says it "Includes a
Feature_Name column to match the DerivaML runtime predicate" — but it
satisfies `find_features` (column-presence) while *failing* to model
the real FK, which is what `_is_feature_association` counts.

Consequently `tests/local_db/test_planner_rules.py:226-236`
(`test_feature_assoc_recognized`) asserts the predicate returns `True`
against this 3-FK fixture, and passes. I ran the file live:

```
tests/local_db/test_planner_rules.py … 24 passed in 0.05s
```

All green — against a fixture that contradicts the production schema.
This is precisely why the bug shipped: the test fixture was authored
to match the predicate's (wrong) assumption, not the real catalog. A
single fixture change — promote `Feature_Name` from a text column to
an FK referencing a `Feature_Name` vocab table, making the table 4-FK
— would have turned `test_feature_assoc_recognized` red and caught
this. `test_four_fk_assoc_is_not_transparent` (4-FK + Execution, no
Feature_Name) is the right negative case to keep, just with a clarified
rationale.

There is **no test anywhere** that exercises `_is_feature_association`
against a 4-FK table that includes a `Feature_Name` FK — the exact
real shape. Same fixture-coverage-gap pattern called out in the
earlier audits of this fix-pass.

---

## 9. Risk assessment of each fix option

| Option | Fixes selector? | False-neg (real features) | False-pos (`FourWayAssoc` etc.) | Rule 6 risk | Verdict |
|--------|-----------------|---------------------------|-------------------------------|-------------|---------|
| E1 `>=3` + 1 Execution | Yes | none | **HIGH** — matches any 3+-FK domain assoc with an Execution FK; breaks `test_four_fk_assoc_is_not_transparent` | Worsens diamond false-positives | **Reject** |
| **E2 Feature_Name FK + Execution FK** | **Yes (verified 550 rows)** | **none** | **none** (FourWayAssoc has no Feature_Name FK) | Low: re-enables the bridge hop; can surface *correct* new `AmbiguousPath` only on multi-path domain schemas; none on cat-27 | **Recommend** |
| E3 Feature_Name FK only | Yes | none | low (a Feature_Name-FK table without Execution is malformed; treating it transparent is mildly risky) | same as E2 | Acceptable, but add the Execution guard → collapses to E2 |

**Implementation risk of E2 itself:** small and localized — change the
`if len(domain_fks) != 3: return False` guard to the
Execution-AND-Feature_Name FK check, update the predicate docstring
(it currently describes the 3-FK model and the `Three_Way_Domain_Assoc`
example), and fix the fixture + `test_feature_assoc_recognized` /
`test_four_fk_assoc_is_not_transparent` to model the real FK shape. The
behavioral blast radius is: (a) selector path goes from
always-raise → works; (b) `_outbound_reachable` / `_enumerate_paths`
start hopping the bridge; (c) Rule 6 gains the ability to flag
feature-bridge diamonds. (a) is pure fix; (b) only widens what's
already reachable via the materialization path; (c) is the one to
land behind a test that proves a single-path feature schema still
plans cleanly (verified here) plus a deliberate multi-path case to
pin the new behavior.

**Separately worth flagging (not fixed here):** the
materialization-path duplication. `_prepare_wide_table` reaches
through the feature bridge via raw `_schema_to_paths` endpoint
filtering with **no** transparency predicate, while
`_outbound_reachable` / `_enumerate_paths` *do* gate on the predicate.
After E2 these converge, but the two engines having different notions
of "can I route through this interior table" is a structural hazard
that allowed the predicate to be wrong-and-invisible for so long.

---

## 10. Limitations

1. Catalog 27 has exactly **one** feature, with a single vocab value
   column (`Image_Class`). I did not exercise multi-value features,
   asset-typed features, or multi-target features — the FK count for
   those is ≥5, well clear of the `== 3` guard, so the predicate fails
   the same way, but the *correct*-predicate false-positive analysis
   for exotic shapes (e.g. a feature with two value columns and a
   second domain FK) was not live-tested.
2. The Rule 6 regression risk (§7.1) is **reasoned + partially
   reproduced**: I confirmed catalog 27 stays ambiguity-free after the
   E2 monkeypatch, but I did **not** construct a synthetic
   diamond-with-feature-bridge schema to force the new
   `AmbiguousPath`. That should be a unit fixture in the fix PR.
3. I tested the **stage3a** worktree (current predicate + Stage 1
   selector + Option B). The e2e run's pinned deriva-ml
   (`1.40.2`, git `8af1bdb5`) predates Stage 1, so the selector-path
   DOA finding (§3) applies to the *current/in-flight* selector
   surface, not to anything the e2e personas actually ran. The
   predicate bug itself (`!= 3`) is present in both `1.40.2` and
   stage3a (and on `main`).
4. The relaxed predicate was applied via runtime monkeypatch, not a
   source edit; I did not run the full `tests/local_db/` suite under
   the relaxed predicate, only the targeted `test_planner_rules.py`
   under the *shipped* predicate (to demonstrate the fixture gap) and
   the targeted live probes under the relaxed one.
5. No catalog writes; no deriva-ml code changes; no PRs — per the
   research-only charter.
