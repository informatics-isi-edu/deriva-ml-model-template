# Investigation 12 — `reduce_with_selector` grouping on key-qualified multi-value features

Research-only audit. No code changed in deriva-ml. Verified against
the **real** `Execution_Subject_Chart_Label` feature on live
`dev.eye-ai.org` catalog `eye-ai`, plus source. The arity cap
(`max_arity=3` → `None`, finding 10) was monkeypatched at runtime to
reach the otherwise-undiscoverable feature; nothing was committed to
deriva-ml.

- Code worktree: `/Users/carl/GitHub/deriva-ml-feature-values-stage3b`
  (carries current `reduce_with_selector` + `Feature` code).
- deriva-py `find_associations` / `is_association` read from the
  worktree's `.venv`.
- This is the one consequence finding 10 deferred (its §4 "qualifier /
  selector semantic gap").

---

## 1. TL;DR

**Does the selector drop qualified observations? YES — catastrophically,
on the real Chart_Label feature.**

Live numbers (cap fix simulated, `feature_values("Subject","Chart_Label")`):

| call | rows returned | distinct Subjects | distinct (Subject, Image_Side) |
|---|---|---|---|
| `selector=None` | **1832** | 917 | **1832** |
| `selector=select_newest` | **917** | 917 | 917 |

`select_newest` collapses 1832 rows → 917, **losing 915 rows** — i.e.
nearly every Subject has BOTH eyes (left + right), and `select_newest`
keeps exactly one eye per Subject and silently drops the other. That is
a wrong answer: each (Subject, Image_Side) row is a distinct, valid
clinical observation, and half of them vanish.

**Is `selector=None` correct with just the cap fix? YES.** With no
selector, all 1832 rows come back — both eyes present, no collapse. The
bug is **selector-path-only**. The cap fix alone (finding 10) makes
Chart_Label fully *discoverable AND readable*. Only the selector path
mis-groups.

**Is the group-by-identity fix small or a Feature-model change?**
It is a **small, contained change** — NOT a Feature-model balloon. The
qualifier FKs are already structurally available on the
`FindAssociationResult` (`other_fkeys` minus the two structural FKs
`Feature_Name`/`Execution`). The `Feature` object simply discards that
information today. The fix is: (a) have `Feature` retain its
qualifier-column names (one comprehension in `Feature.__init__`, reusing
`assoc_fkeys` which it *already computes and throws away*), and (b) have
the three `feature_values` callers build the group key from
`(target_col, *qualifier_cols)` instead of `target_col` alone. No
deriva-py change, no schema change.

**What PR-fix-2 must include to be CORRECT (not just discoverable):**
the cap fix (finding 10) **plus** group-by-identity in
`reduce_with_selector` (tractable, above) **plus** a real Chart_Label
selector regression test. Shipping the cap fix *without* the grouping
fix would make a wrong-under-selector feature discoverable — strictly
worse than the status quo, where the feature is invisible but at least
never silently halves your data. See §8.

---

## 2. The feature's identity key (Question A — live)

Read directly from the live model (`Execution_Subject_Chart_Label`,
schema `eye-ai`):

```
columns: [RID, RCT, RMT, RCB, RMB, Execution, Subject, Feature_Name,
          Condition_Label, Severity_Label, Image_Side, Chart_Label_Provider]

KEYS:
  ['RID']                                              # system
  ['Execution', 'Subject', 'Feature_Name', 'Image_Side']   # compound identity key

FOREIGN KEYS:
  Execution        -> deriva-ml.Execution
  Subject          -> eye-ai.Subject            (the target FK)
  Feature_Name     -> deriva-ml.Feature_Name
  Condition_Label  -> eye-ai.Condition_Label    (VALUE — not in key)
  Severity_Label   -> eye-ai.Severity_Label     (VALUE — not in key)
  Image_Side       -> eye-ai.Image_Side         (QUALIFIER — IN key)

Feature_Name default: Chart_Label
```

**Confirmed:**
- `Image_Side` **is in the compound key** → `(Subject, Image_Side)` is
  the per-target identity (the same Subject legitimately has a Left row
  and a Right row).
- `Condition_Label` and `Severity_Label` are FKs but **NOT in the key**
  — they are value/term columns (the actual labels being recorded).
- This is exactly the structure finding 10 predicted, confirmed live:
  key-arity 4, which is why `max_arity=3` misses it.

`Image_Side` is itself a vocabulary table (terms: `Left` / `Right` /
`Unknown`). Note this for §3: even though `Image_Side` is a vocab FK,
the `Feature` object classifies it as a **value_column**, not a
term_column — see §3.

---

## 3. FeatureRecord shape for the qualified feature (Question B)

With the cap fix simulated, building the `Feature` and its record class:

```
Feature object:
  feature_name : Chart_Label
  target_table : Subject
  feature_columns: ['Chart_Label_Provider', 'Condition_Label', 'Image_Side', 'Severity_Label']
  asset_columns  : []
  term_columns   : ['Condition_Label', 'Severity_Label']
  value_columns  : ['Chart_Label_Provider', 'Image_Side']

FeatureRecord fields:
  ['Chart_Label_Provider', 'Condition_Label', 'Execution',
   'Feature_Name', 'Image_Side', 'RCT', 'Severity_Label', 'Subject']
```

**The qualifier IS present and distinguishable in the record:** the
generated record class has a `Subject` field, an `Image_Side` field, and
`Condition_Label` / `Severity_Label` fields. So a consumer reading rows
(selector=None) sees the full identity and both eyes — correct.

**Subtle but important classification fact** (decisive for the fix in
§6): `Image_Side` lands in `value_columns`, NOT `term_columns`, even
though it is a vocabulary table. Why: `Feature.__init__` builds
`assoc_fkeys = {self_fkey} | other_fkeys`, and on this association the
`FindAssociationResult.other_fkeys` is **`[Feature_Name, Execution,
Image_Side]`** (the key-covered FKs, per deriva-py — see §5). The
term/asset classifier skips any FK in `assoc_fkeys`, so the `Image_Side`
FK is excluded from term classification and falls through to
`value_columns` via `feature_columns - (asset|term)`.

That is slightly *wrong* for term classification (it'd be nicer typed as
a term), but it is *exactly the signal the fix needs*: **the qualifier
columns are precisely `other_fkeys − {Feature_Name FK, Execution FK}`**,
which the association result already carries.

---

## 4. Bug reproduction (Question C — verbatim)

Cap fix monkeypatched (`find_features` → `max_arity=None`), real
eye-ai, no writes:

```
=== selector=None ===
  total rows: 1832
  distinct Subjects: 917
  distinct (Subject, Image_Side) pairs: 1832

=== selector=select_newest ===
  total rows: 917
  distinct Subjects in result: 917
  distinct (Subject,Image_Side) pairs in result: 917

  ROWS LOST by selector: 915
```

A specific Subject with both eyes, showing the drop:

```
=== Example Subject with both eyes: 2-7P00 ===
   Image_Side=Right  Condition=POAG  Severity=Unspecified/Indeterminate  RCT=2025-07-22T23:59:06.834979+00:00  Exec=6-9SMR
   Image_Side=Left   Condition=POAG  Severity=Unspecified/Indeterminate  RCT=2025-07-22T23:59:06.834979+00:00  Exec=6-9SMR

select_newest picks Image_Side=Right (RCT=2025-07-22T23:59:06.834979+00:00) -- DROPS the other eye
```

Mechanism: `reduce_with_selector` groups on `target_col == "Subject"`
only (`feature.py:95-99`). Both eye rows for `2-7P00` land in one group;
`select_newest` returns a single record; the other eye is gone.
(Here both rows even share an RCT, so the choice between eyes is
effectively arbitrary — but either way one valid observation is lost.)

This is not specific to `select_newest`: every selector
(`select_first`, `select_latest`, `select_by_execution`,
`select_by_workflow`, `select_majority_vote`) reduces each
target-RID group to a single record, so all of them collapse the two
eyes. `select_majority_vote` is arguably worse — it would mix
Left-eye and Right-eye labels into one vote.

---

## 5. Correct grouping semantics (Question D)

A selector exists to reduce **redundant** records that describe the
*same logical observation* (e.g. the same Image labelled by three
annotators, or re-run by two executions) down to one. The grouping unit
must therefore be **the feature's identity**, not the target RID.

For a normal (unqualified) feature, identity == target RID, so
group-by-target is correct — that is why the current code works for
Image_Classification, Annotation, Image_Diagnosis, etc. (all key-arity
3, no qualifier).

For a **key-qualified** feature, the schema author put `Image_Side` in
the uniqueness key precisely to declare that *(Subject, Image_Side) is
the identity* — a Left-eye Chart_Label and a Right-eye Chart_Label are
two DISTINCT real observations, not redundant copies of one. A user
asking for "the newest Chart_Label per Subject" almost certainly means
"the newest *per eye*", because the two eyes can have different
glaucoma status. Collapsing them is a semantic error, not a tie-break.

**Correct semantics:** `select_newest` should pick the newest Left-eye
row AND the newest Right-eye row, preserving both. I.e. group by the
full identity key `(Subject, Image_Side)` and reduce *within* each
qualifier bucket. On the live data that yields 1832 rows (one per
(Subject, Image_Side)) — identical to `selector=None` here because each
pair currently has exactly one record, but distinct in general (if some
(Subject, eye) had two annotators, the selector would correctly reduce
*that* pair to one while keeping both eyes).

---

## 6. Fix-location options + tractability (Question E)

### Is the qualifier distinguishable from a value column at the reduce layer?

**Yes — structurally, and it's already in hand at Feature-construction
time.** deriva-py's `Table.find_associations` calls
`is_association(..., return_fkeys=True)`, which returns the set of
**key-covered FKs** (FKs whose columns are a subset of the chosen
compound key). It splits that into `self_fkey` (FK to the target) and
`other_fkeys` (the rest). Live for Chart_Label:

```
self_fkey   : ['Subject']
other_fkeys : [['Feature_Name'], ['Execution'], ['Image_Side']]
```

So the **qualifier columns = `other_fkeys` columns − {Feature_Name,
Execution}** = `{Image_Side}`. A non-qualified feature has
`other_fkeys == {Feature_Name FK, Execution FK}` only → qualifier set
empty → group-by-identity degenerates to group-by-target (the current,
correct behavior). This is a clean, structural discriminator: a value
FK that is NOT in the key (e.g. `Condition_Label`) never appears in
`other_fkeys`; a qualifier FK that IS in the key always does.

`Feature.__init__` **already computes** `assoc_fkeys = {self_fkey} |
other_fkeys` (feature.py:615) and uses it only to *exclude* those FKs
from term/asset classification, then discards it. Retaining the
qualifier subset is one extra comprehension on data already in scope.

### Option 1 — `reduce_with_selector` groups by full identity key. **RECOMMENDED.**

Small, two-part change:

1. `Feature.__init__`: add
   ```python
   structural = {"Feature_Name", "Execution"}
   self.qualifier_columns = {
       fk.foreign_key_columns[0]
       for fk in atable.other_fkeys
       if fk.foreign_key_columns[0].name not in structural
   }
   ```
   (Empty for unqualified features → no behavior change for them.)
   Optionally also surface `qualifier_columns()` as a `FeatureRecord`
   classmethod, mirroring the existing `term_columns()` etc.

2. `reduce_with_selector`: accept the qualifier column names and build a
   composite group key:
   ```python
   def reduce_with_selector(records, target_col, selector, qualifier_cols=()):
       grouped = defaultdict(list)
       for rec in records:
           target_rid = getattr(rec, target_col, None)
           if target_rid is None:
               continue
           key = (target_rid, *(getattr(rec, q, None) for q in qualifier_cols))
           grouped[key].append(rec)
       for group in grouped.values():
           chosen = selector(group)
           if chosen is not None:
               yield chosen
   ```
   The three callers (`FeatureMixin.feature_values`,
   `Dataset.feature_values`, `DatasetBag.feature_values`) pass
   `qualifier_cols=[c.name for c in feat.qualifier_columns]`. Default
   `()` keeps unqualified features and any other caller unchanged.

This does **NOT** balloon into a Feature-model change: it reuses an
already-computed structure, touches one helper + one new Feature
attribute + three one-line call sites, and is fully backward-compatible
(qualifier set empty ⇒ identical to today). Tractable for PR-fix-2.

### Option 2 — Guard: detect qualified feature, raise/warn when a selector is applied.

Detectable with the same `qualifier_columns` set (non-empty ⇒
qualified). Safe (never silently drops) but blocks a *legitimate* use:
the user genuinely may want "newest per eye." Strictly worse than
Option 1 for the same detection cost. Only justified if Option 1 were
hard — it isn't.

### Option 3 — Document as a known limitation, defer the fix.

Acceptable *only* if Option 1 turned out to be a deriva-model balloon
(it doesn't). Given that the fix is small, shipping the cap fix with a
selector that silently halves real data — even documented — is the
worst outcome: discoverable + readable + wrong-under-selector. Reject.

---

## 7. selector=None correctness (Question G)

**Confirmed: with just the cap fix and NO selector, Chart_Label reads
correctly.** `feature_values("Subject","Chart_Label", selector=None)`
returns all 1832 rows = all 1832 distinct (Subject, Image_Side) pairs,
both eyes present (see §4). The no-selector path
(`feature.py:518-520`, `yield from records`) does no grouping at all, so
qualifiers can't collapse. The FeatureRecord carries `Image_Side`, so
consumers can group/pivot themselves.

This narrows PR-fix-2's *required* scope: the cap fix alone delivers
**discoverable + readable + correct** for the dominant read pattern
(no selector). The additional grouping work is needed **only** to make
the *selector* path correct.

---

## 8. PR-fix-2 scope recommendation (Question F) — the minimum CORRECT fix

The minimum that is *correct* (not merely complete):

1. **Cap fix** (finding 10): `max_arity=3` → `max_arity=None` in
   `find_table_features`. Makes Chart_Label discoverable + readable.
2. **Group-by-identity in `reduce_with_selector`** (Option 1, §6):
   `Feature.qualifier_columns` + composite group key + 3 call-site
   updates. Makes the *selector* path correct (newest-per-eye, not
   newest-per-Subject).
3. **A real Chart_Label-shaped selector regression test**: a fixture
   with a value FK *in the compound key* (the distinguishing structural
   feature — not just extra FK count), asserting that `select_newest`
   over a Subject-with-two-eyes returns **both** eyes, and that an
   unqualified feature still reduces to one-per-target. (No such fixture
   exists today — confirmed: nothing in `tests/` references
   `Chart_Label` / `Image_Side` / a key-qualified feature. Same
   fixture-lies gap flagged across this session.)

**Why grouping must ride with the cap fix, not defer:** shipping (1)
alone would expose a feature that is *discoverable and readable* but
returns *silently halved* data the moment anyone applies a selector
(the standard "give me the newest label per target" idiom). That is a
correctness regression introduced by the very PR meant to fix a
correctness bug. Today the feature is invisible — annoying, but never
wrong. (1)+(2) together flip it to visible-and-correct on every path.
The cost of (2) is small (§6), so there is no scope argument for
deferring it.

If, against expectation, (2) cannot land in the same PR, the safe
fallback is (1) + **Option 2 guard** (raise when a selector is applied
to a qualified feature) + test + filed follow-up — never (1) alone.

**Recommendation: PR-fix-2 = cap fix + group-by-identity (Option 1) +
real qualified-feature selector test.**

---

## 9. Limitations

- The 1832/917 counts are live as of this session against
  `dev.eye-ai.org` catalog `eye-ai`; the table is mutable by its
  owners, so exact counts may drift. The *structure* (Image_Side in the
  key; selector collapses to one-per-Subject) is what matters and is
  schema-level, not data-level.
- The cap fix was simulated via runtime monkeypatch of
  `find_features` (max_arity=None); `catalog.py` was not edited and the
  test suite was not run (research-only). A real PR-fix-2 must add the
  §8.3 test and run `uv run python -m pytest tests/feature tests/model`.
- The Option-1 patch in §6 is sketched, not implemented or executed.
  The `qualifier_columns` derivation is verified against the live
  `other_fkeys` for Chart_Label (`Image_Side` present; structural FKs
  separable), and the degenerate (unqualified ⇒ empty) case is verified
  by construction, but the composite-key reduction was reproduced by
  hand in this audit, not run through the real `reduce_with_selector`.
- The mild mis-classification of `Image_Side` as a `value_column`
  rather than a `term_column` (§3) is noted but out of scope — it does
  not affect read/write correctness, only the python field type, and is
  orthogonal to the grouping bug.
- I did not test whether deriva-ml's own `create_feature` can *produce*
  a key-qualified feature. It cannot via the documented path:
  `create_feature` → `_define_association(associates=[execution,
  target, feature_name])` puts only those three FKs in the key, so any
  extra term/asset/metadata column is impure (not key-covered).
  Chart_Label's `Image_Side`-in-key shape was therefore produced by a
  different path (hand-built schema or an older/custom create). Either
  way the feature exists in eye-ai today and discovery + selector must
  handle it. (Carried over from finding 10 §5.)
