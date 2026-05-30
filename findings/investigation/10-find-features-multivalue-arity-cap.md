# Investigation 10 — `find_features` `max_arity=3` cap vs multi-value features

Research-only audit. No code changed. Verified against live
`dev.eye-ai.org` catalog `eye-ai` and live `localhost` catalog `27`
(the e2e CIFAR-10 catalog), plus source.

- Code worktree: `/Users/carl/GitHub/deriva-ml-feature-values-stage3b`
  HEAD `97d85be4` (`refactor(dataset): remove BagFeatureCache …`),
  which carries the current E2 (`_is_feature_association`) code.
- deriva-py `find_associations` / `is_association` read from the
  worktree's `.venv` (deriva-py @ `9501c2b`).

---

## 1. TL;DR

**Is `find_features` broken for multi-value features? YES — but the
mechanism is subtler than "domain-FK count > 3", and the blast radius
on eye-ai is exactly ONE feature, not six.**

The brief's premise — that `max_arity=3` excludes any feature with >3
*domain FKs* — is **wrong about the discriminator**. `find_associations`
does not count domain FKs. It counts **the FKs covered by the
association table's compound uniqueness key** (deriva-py
`Table.is_association`: arity = `len(covered_fkeys)` where
`covered_fkeys` are the FKs whose columns are a subset of the chosen
non-system compound key). Value columns that are *impure metadata*
(NOT part of the key) do not count toward arity.

Consequence on eye-ai (7 feature-shaped tables total):

| feature table | domain FKs | **key-FK arity** | `find_features`? |
|---|---|---|---|
| `Annotation` | 6 | **3** | ✅ found |
| `Image_Diagnosis` | 7 | **3** | ✅ found |
| `Execution_Image_Fundus_Angle` | 4 | 3 | ✅ found |
| `Execution_Image_Fundus_Laterality` | 4 | 3 | ✅ found |
| `Execution_OCT_DICOM_OCT_Embedding` | 4 | 3 | ✅ found |
| `Execution_CGM_Blood_Glucose_CGM_Features` | 4 | 3 | ✅ found |
| **`Execution_Subject_Chart_Label`** | **6** | **4** | ❌ **MISSED** |

So `find_features` on eye-ai misses **1 of 7** features. `Annotation`
(6 FK) and `Image_Diagnosis` (7 FK) are *not* missed — their extra
value FKs are impure payload, not key material, so they read as
arity-3. The real discriminator is whether a value FK is a
**qualifier** (part of the row identity / compound key). On
`Chart_Label`, `Image_Side` is a qualifier: the same
`(Execution, Subject, Feature_Name)` repeats once per eye
(left/right), so `Image_Side` is in the key, pushing key-arity to 4.

**Is `feature_values` ALSO broken for the missed feature? YES.**
Every `feature_values` surface (`FeatureMixin`, `Dataset`,
`DatasetBag`) resolves the feature through
`lookup_feature(table, name)` → `find_features(table)` — the same
arity-capped path. For `Execution_Subject_Chart_Label`,
`ml.lookup_feature("Subject", "Chart_Label")` and
`ml.feature_values("Subject", "Chart_Label")` both raise
`DerivaMLFeatureNotFound` (verified live). The feature is invisible to
discovery AND to value retrieval. This is a bigger deal than discovery
alone — the value-read path is also dead for qualified multi-value
features.

**Recommended fix (discovery-only, NOT deeper):** drop `max_arity=3`
(keep `min_arity=3`) in `find_table_features` so `is_feature` becomes
the sole filter. Verified live: this recovers `Chart_Label`, raises
the eye-ai feature count 6 → 7, and produces **zero false positives**
(`is_feature` still requires `Feature_Name + Execution + target` FK
subset). `Feature.__init__` and `feature_record_class` **already**
handle N value columns (they classify all FKs into asset/term/value
sets — no single-value assumption). `feature_values`' fetch path is
column-agnostic. So the arity-3 assumption is **not** baked deeper
than the discovery gate; the fix is one line plus a regression test.
Risk: low. Test gap: no test exercises a key-qualified multi-value
feature (the usual fixture-lies pattern).

---

## 2. Live confirmation

### eye-ai `Execution_Subject_Chart_Label`

```
find_features("Subject")  ->  []           # count 0
```

Table structure (live):

```
schema: eye-ai
domain FK count: 6
   FK -> deriva-ml.Execution        cols: ['Execution']
   FK -> eye-ai.Subject             cols: ['Subject']
   FK -> deriva-ml.Feature_Name     cols: ['Feature_Name']
   FK -> eye-ai.Condition_Label     cols: ['Condition_Label']
   FK -> eye-ai.Severity_Label      cols: ['Severity_Label']
   FK -> eye-ai.Image_Side          cols: ['Image_Side']
columns: [RID, RCT, RMT, RCB, RMB, Execution, Subject, Feature_Name,
          Condition_Label, Severity_Label, Image_Side, Chart_Label_Provider]
compound key: ['Execution', 'Subject', 'Feature_Name', 'Image_Side']
```

Row count was reported as 1,832 in the brief; the table is real and
populated. The compound key includes `Image_Side` → key-FK arity 4 →
`find_associations(min_arity=3, max_arity=3, pure=False)` rejects it
("too many fkeys in association").

`lookup_feature` / `feature_values` on it (live):

```
ml.lookup_feature("Subject", "Chart_Label")
  -> DerivaMLFeatureNotFound: Feature not found: Chart_Label on Subject
ml.feature_values("Subject", "Chart_Label")
  -> DerivaMLFeatureNotFound: Feature not found: Chart_Label on Subject
```

(`Feature_Name` default on the table confirms the feature name is
`Chart_Label`.)

### Full enumeration — eye-ai (feature-shaped = has FK to deriva-ml.Feature_Name AND deriva-ml.Execution)

7 feature-shaped tables; see the §1 table. **1 missed**
(`Execution_Subject_Chart_Label`, key-FK arity 4). `find_features()`
catalog-wide returns 6.

Why `Annotation` (6 domain FKs) and `Image_Diagnosis` (7 domain FKs)
are NOT missed: their compound keys are
`['Execution', 'Image', 'Feature_Name']` (arity 3). The extra value
FKs are impure metadata columns not covered by the key, so
`pure=False` admits them while the key-arity stays at 3. This is the
*designed* single-value-per-(target,execution) feature shape with
multiple decoration columns — distinct from a *qualified* feature
where a value participates in identity.

### Full enumeration — catalog 27 (localhost, e2e CIFAR-10)

1 feature-shaped table:
`e2e-test-20260528.Execution_Image_Image_Classification`, domain FK 4,
**key-FK arity 3** (`['Execution', 'Image', 'Feature_Name']`),
discoverable. `find_features()` returns 1. **Cat 27 misses zero
features** — CIFAR-10 has no qualified multi-value features. The bug
is only observable on eye-ai.

---

## 3. The arity gate vs `is_feature` (Question B)

**The gate, not the predicate, excludes the feature.** `is_feature`
itself would PASS on `Chart_Label`:

```python
# model/catalog.py, find_features()
def is_feature(a: FindAssociationResult) -> bool:
    return {
        "Feature_Name",
        "Execution",
        a.self_fkey.foreign_key_columns[0].name,   # "Subject"
    }.issubset({c.name for c in a.table.columns})
```

`Chart_Label`'s columns include `Feature_Name`, `Execution`, and
`Subject` → subset check is True. But `is_feature` is never called for
it, because the generator is gated:

```python
def find_table_features(t: Table) -> list[Feature]:
    return [
        Feature(a, self)
        for a in t.find_associations(min_arity=3, max_arity=3, pure=False)
        if is_feature(a)
    ]
```

**`max_arity=3` is an explicit argument deriva-ml passes** — it is NOT
the deriva-py default. deriva-py's signature
(`deriva.core.ermrest_model.Table.find_associations`):

```python
def find_associations(self, min_arity=2, max_arity=2, unqualified=True,
                      pure=True, no_overlap=True) -> Iterable[FindAssociationResult]:
```

Default `max_arity=2`; deriva-ml overrides to 3. `max_arity=None`
means "no upper bound".

**What arity counts (deriva-py `Table.is_association`):** it picks the
longest non-system compound uniqueness key as `row_key`, then
`covered_fkeys = {fk for fk in self.foreign_keys if
set(fk.foreign_key_columns).issubset(row_key)}`, and
`arity = len(covered_fkeys)`. With `pure=False`, non-key columns are
permitted ("impure metadata merely decorates"). So **arity = number of
FKs that are part of the identity key**, not the total FK count. This
is exactly why a 7-FK table (`Image_Diagnosis`) passes and a 6-FK
table (`Chart_Label`) fails.

---

## 4. `Feature` / `FeatureRecord` multi-value support (Question C)

**Multi-value is fully supported in the model layer — no arity-3
assumption is baked in.** `Feature.__init__` (`feature.py`) classifies
**all** non-structural columns:

```python
skip_columns = {"RID","RMB","RCB","RCT","RMT","Feature_Name",
                self.target_table.name, "Execution"}
self.feature_columns = {c for c in self.feature_table.columns
                        if c.name not in skip_columns}
assoc_fkeys = {atable.self_fkey} | atable.other_fkeys
self.asset_columns = {fk.foreign_key_columns[0] for fk in ...
                      if fk not in assoc_fkeys and is_asset(fk.pk_table)}
self.term_columns  = {fk.foreign_key_columns[0] for fk in ...
                      if fk not in assoc_fkeys and is_vocabulary(fk.pk_table)}
self.value_columns = self.feature_columns - (asset|term)
```

These are **sets** of arbitrary cardinality. `feature_record_class()`
builds the pydantic model by iterating `self.feature_columns` and
emitting one optional field per column — N value columns → N fields.
`select_majority_vote` even has an explicit multi-term-column branch
(requires `column=` when `len(term_columns) > 1`). So a generated
`FeatureRecord` for `Chart_Label` would correctly carry
`Condition_Label`, `Severity_Label`, `Image_Side` fields. **The only
thing standing between the user and that record class is the
discovery gate.**

One nuance worth flagging (not a blocker): the `Feature` object groups
all qualifier value FKs as ordinary term/value columns. It has no
notion of "this value FK is part of the row identity" (a qualifier)
vs. "this value FK is a single value per target". For `Chart_Label`
that means the generated record treats `Image_Side` as just another
term column — which is correct for read/write, but consumers that
assume one row per `(target, execution)` (e.g. naive selectors keyed
on target RID alone) would collapse left/right-eye rows. The
`reduce_with_selector` grouping in `feature_values` groups on
`target_col` only (the Subject FK), so two Chart_Label rows for the
same Subject (left + right eye) would land in the same group and a
selector would pick one — silently dropping the other eye. This is a
*semantic* gap for qualified features, separate from the discovery
bug, and would surface only after the discovery gate is lifted. Worth
a follow-up but out of scope here.

---

## 5. Blast radius — every `find_associations` call site (Question D)

```
core/mixins/vocabulary.py:424   find_associations()                 # default arity 2; vocab usage, unrelated
dataset/bag_builder.py:682      find_associations(max_arity=3, pure=False)   # dataset-member assoc walk
dataset/bag_builder.py:878      find_associations()                 # default
dataset/dataset.py:258          find_associations()                 # default
dataset/dataset.py:531          find_associations()                 # default
dataset/dataset.py:1635         find_associations()                 # default
model/catalog.py:516            find_associations(pure=False)       # find_association(): default arity 2
model/catalog.py:662            find_associations(min_arity=3, max_arity=3, pure=False)  # <-- THE BUG
model/catalog.py:909            find_associations()                 # list_dataset_element_types: default
```

Only **one** call site uses `min_arity=3, max_arity=3` — the feature
discovery in `find_table_features`. The `bag_builder.py:682`
`max_arity=3` is a *dataset-member* association walk, not feature
discovery, and is a separate concern (it bounds how many ways a
dataset row can link; not in scope).

**Code paths that assume features are arity-3:** only the discovery
gate. Everything downstream (`Feature`, `feature_record_class`,
`feature_values` fetch) is column-set driven and arity-agnostic
(see §4).

**Does `create_feature` round-trip today?** `create_feature`
(`core/mixins/feature.py`) chains
`metadata=[... for m in chain(assets, terms, metadata)]` into a single
`_define_association` → one association with
`target + Feature_Name + Execution + N value columns`. Its **return**
is `self.feature_record_class(target, name)` →
`lookup_feature(...).feature_record_class()` → `find_features`. So:

- If the N value columns are emitted as **impure metadata** (not in
  the row key) — the deriva-py default for `define_association`
  metadata — key-FK arity stays 3 and the round-trip **works** (this
  is the `Annotation` / `Image_Diagnosis` shape).
- If any value FK becomes part of the compound key (a **qualifier**,
  the `Chart_Label` / `Image_Side` shape), the round-trip **breaks**:
  `create_feature` succeeds in building the table, but its own return
  statement's `lookup_feature` would raise `DerivaMLFeatureNotFound`.

I did NOT create a feature on eye-ai (shared catalog) to test this
directly — reasoned from code + the live `Chart_Label` example, which
is exactly the broken shape. Whether deriva-ml's own `create_feature`
can *produce* a key-qualified feature depends on `define_association`'s
key construction; the `Chart_Label` table exists in eye-ai today, so
*something* produced it (possibly a hand-edited schema or an older
create path). Either way, discovery must handle it.

---

## 6. `lookup_feature` / `feature_values` path for multi-value (Question F) — the critical one

**All three `feature_values` surfaces go through the arity-capped
`find_features` path. Confirmed by tracing:**

- `FeatureMixin.feature_values` (`core/mixins/feature.py:484`):
  `feat = self.lookup_feature(table_obj, feature_name)`
- `Dataset.feature_values` (`dataset/dataset.py:686`):
  `feat = self.lookup_feature(table, feature_name)` →
  `self._ml_instance.lookup_feature(...)` (line 739)
- `DatasetBag.feature_values` (`dataset/dataset_bag.py:624`):
  `feat = self.lookup_feature(table, feature_name)` →
  `self.model.lookup_feature(...)` (line 683)

And `DerivaModel.lookup_feature` (`model/catalog.py:734`):

```python
return [f for f in self.find_features(table) if f.feature_name == feature_name][0]
# IndexError -> DerivaMLFeatureNotFound
```

`find_features(table)` → `find_table_features` → the arity-capped
`find_associations`. So a feature whose key-FK arity > 3 is absent
from `find_features`, `lookup_feature` raises
`DerivaMLFeatureNotFound`, and **all** `feature_values` callers
(online `DerivaML`, `Dataset`, offline `DatasetBag`) fail for it.
Verified live: `ml.feature_values("Subject", "Chart_Label")` →
`DerivaMLFeatureNotFound`.

**Relationship to E2 / the consolidation:** PR #259's claim that E2
`_is_feature_association` "mirrors `find_features.is_feature`" is true
for the **predicate** but false for **`find_features` as a whole**.
E2 keys off `Feature_Name + Execution` FK presence with **no arity
cap** (`denormalize_planner.py:489-496`), so E2 *correctly* recognizes
`Execution_Subject_Chart_Label` as a feature-association and treats it
as transparent. `find_features` does not. The two diverge precisely on
key-qualified multi-value features. For the Stages 3a/3b consolidation
this matters: `feature_values` was refactored to delegate to the
Denormalizer, but the **feature lookup** still happens via
`lookup_feature`→`find_features` before the Denormalizer runs (the
denormalizer is asked to materialize a *known* feature table). So even
though the Denormalizer's planner (E2) would happily traverse
`Chart_Label`, `feature_values` never reaches that code for it —
`lookup_feature` raises first. The consolidation did not fix this;
the divergence is upstream of the delegation boundary.

---

## 7. Fix recommendation + risk + test gap (Questions E, G)

### Fix (discovery-only)

In `model/catalog.py`, `find_table_features`:

```python
# before
for a in t.find_associations(min_arity=3, max_arity=3, pure=False)
# after
for a in t.find_associations(min_arity=3, max_arity=None, pure=False)
```

`is_feature` becomes the sole filter — which is what it always
effectively was for the kept features. Verified live on eye-ai:

- Per-table: `find_features("Subject")` →
  `['Execution_Subject_Chart_Label']` (was `[]`).
- Catalog-wide: 7 features (was 6); the added one is
  `Execution_Subject_Chart_Label`.
- **False positives: zero.** Every table the fix newly admits is
  feature-shaped (`Feature_Name + Execution` FK present). The
  `min_arity=3` floor + `is_feature` subset check
  (`Feature_Name + Execution + target`) is a strictly stronger filter
  than the arity ceiling ever was.

### Is removing the cap safe? (Question E)

Yes. The reasoning in the brief holds against the code: `is_feature`
is the real filter; `max_arity=3` only *pre-excludes* valid features
whose value FKs are key-qualified. Dropping the ceiling cannot admit a
non-feature, because:

1. `min_arity=3` still requires ≥3 key-covered FKs.
2. `is_feature` still requires `Feature_Name`, `Execution`, and the
   target FK column all present.

A non-feature 3+-FK association (e.g. a 4-way domain join) lacks the
`Feature_Name` FK and is rejected by `is_feature` — confirmed by the
zero-false-positive live result and by E2's own design notes
(`denormalize_planner.py:464-466`, the `FourWayAssoc` example).

### Scope: discovery-only, NOT deeper

`Feature`/`feature_record_class`/`feature_values` all handle N value
columns already (§4). The fix is the one-line ceiling removal. The
*semantic* qualifier gap (§4: `reduce_with_selector` groups on target
RID only, collapsing left/right-eye Chart_Label rows) is a separate,
lower-priority follow-up — it only manifests after discovery is fixed,
and only for features that put a value FK in the identity key.

### Test gap

No test exercises a key-qualified multi-value feature. `tests/feature/`
and the cached `tests/dataset/eye-ai-catalog-schema.json` fixture
should be checked: if the fixture omits `Execution_Subject_Chart_Label`
(or includes it but no test asserts `find_features` returns it), that's
the same fixture-lies pattern flagged across this session. A regression
test should: (1) assert `find_features("Subject")` includes
`Chart_Label`; (2) assert `lookup_feature("Subject","Chart_Label")`
succeeds and the record class carries `Image_Side`; (3) assert
`feature_values` returns rows. Ideally with a fixture that has a value
FK in the compound key (the distinguishing structural feature), not
just extra FK count.

---

## 8. Limitations

- Catalog 27 is the localhost e2e CIFAR-10 catalog (per
  `src/configs/deriva.py:29`), not a separate eye-ai catalog. It has
  no qualified multi-value features, so it neither reproduces nor
  contradicts the bug — included for completeness (misses 0).
- The 1,832-row count for `Chart_Label` was taken from the brief; I
  confirmed the table is real, populated, and has the stated 6 domain
  FKs, but did not re-count rows (read-only audit; no need).
- I did NOT create a test feature on eye-ai (shared catalog), so the
  `create_feature` round-trip claim for the *qualified* shape (§5) is
  reasoned from code + the existing `Chart_Label` example, not
  executed end-to-end. The non-qualified round-trip (Annotation shape)
  demonstrably works (those features are discoverable today).
- The fix was validated by simulating `max_arity=None` in-process
  against live eye-ai; I did not edit `catalog.py` or run the test
  suite (research-only). A real patch should add the regression test
  in §7 and run `uv run python -m pytest tests/feature tests/model`.
- The §4 qualifier/selector semantic gap is identified but not
  fully scoped — it warrants its own investigation once discovery is
  fixed.
