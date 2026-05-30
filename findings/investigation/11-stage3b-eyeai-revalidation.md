# Investigation 11 — Stage 3b (PR #260) bag-delegation re-validation on eye-ai

**Investigator:** Stage 3b eye-ai re-validation (verification-only)
**Date:** 2026-05-29
**Scope:** Re-run the three-way oracle that validated PR #260
(Stage 3b — `DatasetBag.feature_values` delegates to the Denormalizer
bag path; `BagFeatureCache` deleted) against the richer
`dev.eye-ai.org` / `eye-ai` domain model, to confirm the bag delegation
holds beyond catalog 27's single simple `Image_Classification` feature.

- **Code under test:** worktree
  `/Users/carl/GitHub/deriva-ml-feature-values-stage3b`, branch
  `refactor/feature-values-delegate-stage3b`, HEAD `97d85be4`
  (delegate `bf6e2f12` + remove BagFeatureCache `97d85be4`, on top of
  the #259 E2 predicate fix `ae94e5d7`).
- **No code changed.** Throwaway verification scripts only (deleted).
- **eye-ai is a shared dev catalog — strictly read-only.** No datasets,
  features, executions, or rows were written. Bag *downloads* are
  reads; the only attempted writes (none succeeded) would have been the
  catalog's own export-side effects, which never got that far (the
  build failed during manifest construction — see §3).

---

## 1. TL;DR

**A real bag oracle surfaced a genuine C ≠ A divergence — STOP-worthy
for #260 as written.** On the live eye-ai dataset `6-EKGA` (a
multi-element-type AIREADI dataset: CGM_Blood_Glucose, Subject,
OCT_DICOM members):

- **A** — `Dataset(ml,"6-EKGA").feature_values("CGM_Blood_Glucose",
  "CGM_Features")` against the **live catalog** returns **27 rows**
  (selector=None) / **9 rows** (select_newest). No error.
- **C** — `DatasetBag(...).feature_values("CGM_Blood_Glucose",
  "CGM_Features")` against a **real downloaded bag** of the same
  dataset/version **RAISES `DerivaMLDenormalizeUnrelatedAnchor`**:
  *"Anchors of table(s) ['OCT_DICOM', 'Subject'] have no FK path to any
  table in include_tables=['CGM_Blood_Glucose',
  'Execution_CGM_Blood_Glucose_CGM_Features']."*

Both wrappers delegate to the identical
`Denormalizer(self).feature_records(feat, selector=None)` with
`ignore_unrelated_anchors=False` hard-coded (denormalizer.py:670). The
divergence is in **anchor enumeration**: the bag Denormalizer treats
the bag's physically-present sibling element types (Subject, OCT_DICOM)
as hard-fail anchors for a CGM-target feature; the live-`Dataset`
Denormalizer does not raise on the **same three member tables** (live
`list_dataset_members` confirms 6-EKGA has CGM=10, Subject=10,
OCT_DICOM=226 — identical anchor set). So this is **not** an artifact
of bag construction or of the `exclude_tables={"Image"}` workaround
used to dodge the corrupt asset (§3) — the live dataset carries the
exact same unrelated element types and A tolerates them.

**This means `DatasetBag.feature_values` (the PR #260 code) will raise
on any multi-element-type bag whose dataset has element types unrelated
to the target feature — a common shape.** Catalog 27's single-element
dataset (one `Image_Classification` feature, no unrelated siblings)
could not surface this, which is why C==A held there.

Secondary findings:
- Catalog-side (A) reads on eye-ai's richer Image features
  (`Fundus_Laterality`: vocab + 2 floats + datetime RCT) are bit-clean
  and shape-correct (§5).
- The multi-value `Chart_Label` feature is **not reachable** via
  `lookup_feature` (`DerivaMLFeatureNotFound`), corroborating
  investigation 10's `find_features` arity-cap audit (§6).
- Image-bearing eye-ai bags additionally cannot be built at all (corrupt
  zero-byte asset → 409, §3), independent of #260.

---

## 2. Dataset & feature inventory (eye-ai, live)

`find_features()` returns **6** features (the 7th, `Chart_Label`, is
missed — see investigation 10). Populated feature-table row counts
(live, this run):

| feature table | target | rows | distinct targets |
|---|---|---|---|
| `Execution_Image_Fundus_Angle` | Image | 1972 | 1972 (1:1) |
| `Execution_Image_Fundus_Laterality` | Image | 1972 | — |
| `Execution_OCT_DICOM_OCT_Embedding` | OCT_DICOM | **0** (empty) | — |
| `Execution_CGM_Blood_Glucose_CGM_Features` | CGM_Blood_Glucose | 243 | 117 |

OCT_Embedding is empty → unusable for the oracle.

**Smallest Image-bearing datasets with feature coverage** (member Image
count | covered by Fundus_Angle):

```
5-XW4J  10 images  10 covered   v0.3.1   <- ideal tiny target
5-YKAY   9 images   4 covered   v0.1.1
5-STDM 343 images 343 covered
5-STDA 1397 images 1397 covered
```

`5-XW4J` (10 Image members, all with Fundus_Angle + Fundus_Laterality)
was chosen as the bag target.

---

## 3. Bag download blocked by a corrupt eye-ai asset (NOT #260)

`Dataset("5-XW4J").download_dataset_bag(version="0.3.1")` **fails**:

```
DerivaMLException: Dataset bag export failed: [DerivaDownloadError]
Exception during HEAD request: [HTTPError] 409 Client Error: CONFLICT
for url:
https://dev.eye-ai.org/hatrac/images/scans/subject/1129530/observation/1641361/image/15697382/d41d8cd98f00b204e9800998ecf8427e.jpg
```

The failing object's MD5 is `d41d8cd98f00b204e9800998ecf8427e` — the
**MD5 of zero bytes**. eye-ai's hatrac holds a **zero-byte (empty)
image upload**; the export builder's `HEAD` for the remote-file
manifest gets a `409 CONFLICT` and aborts the whole bag build (in
`bag_fetch_query_processor.createManifestEntry`, before any
materialization).

This is **catalog-side data corruption**, independent of PR #260:

- It fires during *manifest construction*, not in any
  `feature_values` / Denormalizer code.
- `5-YKAY` (a different tiny dataset, disjoint-looking 9-image set)
  hits the **exact same asset URL** — so the corrupt image is a member
  of a shared/parent dataset that both leaf datasets inherit; it
  poisons every Image-bearing bag export on this catalog.
- `use_minid=True` is not an option (`s3_bucket` not configured on
  this DerivaML instance).

**Consequence:** a real bag (C) for an *Image* feature cannot be
produced on eye-ai today without first repairing/removing the corrupt
asset (a catalog-maintenance task, out of scope for this read-only
verification).

---

## 4. Real CGM bag oracle — the C ≠ A divergence

To obtain a real bag despite the corrupt Image asset, I targeted a
**CGM-bearing** dataset and excluded the Image table from FK traversal.
Smallest CGM-covered datasets (members with `CGM_Features`):

```
6-EKGA  cgm=10  cov=9   v0.5.2   <- chosen
6-EQ8C  cgm=15  cov=15  v0.5.0
6-ERV6  cgm=98  cov=93  v0.7.0
```

Bag-build observations:
- `materialize=True, exclude_tables={"Image"}`: manifest built (no 409
  this time — Image excluded), but **file materialization timed out**
  (>540s) fetching CGM data files. eye-ai bag *file* fetch is slow.
- `materialize=False, exclude_tables={"Image"}`: **succeeded** — bag
  downloaded (`DatasetBag rid='6-EKGA' version='0.5.2'
  types=['AIREADI']`). `feature_values` only needs the in-bag feature
  CSV, not the fetched asset blobs, so `materialize=False` is
  sufficient for the oracle.

**Oracle (real bag):**

```
A  Dataset.feature_values("CGM_Blood_Glucose","CGM_Features", None)           -> 27 rows
A  Dataset.feature_values(... select_newest)                                  ->  9 rows
   A[0] keys: ['CGM_Blood_Glucose','CGM_Features','Execution','Feature_Name','RCT']
   A[0]: {'Execution':'6-EKZW','Feature_Name':'CGM_Features',
          'RCT':'2026-02-17T03:03:48.787385+00:00',
          'CGM_Features':'6-EM18','CGM_Blood_Glucose':'6-E5RM'}
   (select_newest meaningfully reduces 27 -> 9: multi-execution feature)

C  DatasetBag.feature_values("CGM_Blood_Glucose","CGM_Features", None)         -> RAISES
C  DatasetBag.feature_values(... select_newest)                               -> RAISES
   DerivaMLDenormalizeUnrelatedAnchor: Anchors of table(s)
   ['OCT_DICOM','Subject'] have no FK path to any table in
   include_tables=['CGM_Blood_Glucose','Execution_CGM_Blood_Glucose_CGM_Features'].
```

**C ≠ A. C raises on both selector cases; A returns rows.**

### Root cause (code-confirmed)

Both surfaces delegate identically:
- `Dataset.feature_values` (dataset.py:694):
  `Denormalizer(self).feature_records(feat, selector=None)`
- `DatasetBag.feature_values` (dataset_bag.py:632): **same call**.

`Denormalizer.feature_records` (denormalizer.py:670) runs
`self._run([target_table, feat_table], …, ignore_unrelated_anchors=False)`
— the `include_tables` is just the CGM target + feature table in **both**
paths. The only variable is **what `self` enumerates as anchors**:

- `Denormalizer(Dataset)` (live): does **not** flag Subject / OCT_DICOM
  as unrelated anchors for the CGM feature → 27 rows.
- `Denormalizer(DatasetBag)` (bag): enumerates the bag's
  physically-present member tables (Subject, OCT_DICOM, CGM) as anchors
  → Subject & OCT_DICOM have no FK path to the CGM include set →
  `ignore_unrelated_anchors=False` makes it **raise**.

Confirmed the live dataset carries the identical anchor set — so this is
**not** a bag-construction or `exclude_tables` artifact:

```
6-EKGA live list_dataset_members():
   CGM_Blood_Glucose 10   Subject 10   OCT_DICOM 226
```

### Why catalog 27 missed it

Catalog 27's validation dataset had a single element type (Image) and
one feature whose target *is* that element type — there were no
unrelated sibling anchors to trip the guard, so the bag Denormalizer's
anchor set was trivially all-related and C==A held. eye-ai's
multi-modal datasets (CGM + Subject + OCT in one dataset) are exactly
the shape that exposes the difference.

### Severity

STOP-worthy for #260 as written: `DatasetBag.feature_values` will raise
`DerivaMLDenormalizeUnrelatedAnchor` for any feature read on a bag whose
dataset has element types unrelated to the feature's target — a common
real-world dataset shape. The likely fix is to pass
`ignore_unrelated_anchors=True` from the `feature_values` wrappers (a
feature read is intentionally scoped to one target + its feature table;
sibling element types are legitimately irrelevant and should be dropped,
not error), and to make the **live** `Dataset` path do the same so both
surfaces stay in lockstep. Whichever way it's resolved, A and C must
agree — today they do not.

---

## 5. Catalog-side (A) reads on richer features — bit-clean

Both Image features read cleanly from the **live catalog** (the
Denormalizer-backed, #259-predicate-fixed ground truth that the bag
path now shares), under both `selector=None` and
`selector=FeatureRecord.select_newest`, on dataset `5-XW4J` (10 Image
members):

**`Fundus_Angle`** (single vocab value column — closest analogue to
catalog 27's `Image_Classification`):

```
A  Image/Fundus_Angle  selector=None:          rows=10
A  Image/Fundus_Angle  selector=select_newest: rows=10
keys: ['Execution', 'Feature_Name', 'Image', 'Image_Angle', 'RCT']
sample: {'Execution': '5-SJG2', 'Feature_Name': 'Fundus_Angle',
         'RCT': '2025-01-18T07:40:14.932987+00:00',
         'Image_Angle': '2', 'Image': '2-BDAM'}
RCT type: str (UTC-aware ISO-8601)   Image_Angle: str
```

**`Fundus_Laterality`** (RICHER than anything in catalog 27 — a vocab
value column `Image_Side` PLUS two `float` probability columns, one
nullable):

```
A  Image/Fundus_Laterality  selector=None:          rows=10
A  Image/Fundus_Laterality  selector=select_newest: rows=10
keys: ['Execution', 'Feature_Name', 'Image', 'Image_Side',
       'Left_Prob', 'Right_Prob', 'RCT']
sample: {'Execution': '5-SPHC', 'Feature_Name': 'Fundus_Laterality',
         'RCT': '2025-01-19T17:40:36.995778+00:00',
         'Image_Side': 'Right', 'Left_Prob': None,
         'Right_Prob': 0.95773345, 'Image': '2-BDAM'}
RCT type: str (UTC-aware ISO-8601)
Image_Side: str   Left_Prob: None (nullable float)   Right_Prob: float
```

This exercises exactly the value-shape concern raised in
investigation 08 §8 on a feature far richer than catalog 27's:

- **Datetime (RCT):** comes back as a UTC-aware ISO-8601 **string**
  (`+00:00`), the canonical shape — matching catalog 27.
- **Floats:** `Right_Prob` is a native Python `float`; `Left_Prob` is a
  proper `None`, not the string `"None"` or `""`.
- **Multiple value columns + vocab term** coexist on one
  FeatureRecord without collision.

Each Image carries exactly one annotation per feature, so
`select_newest` returns the same 10 rows as `selector=None` — the
selector reduction is a no-op here, confirming it does not drop or
duplicate rows on a 1-annotation-per-target feature.

---

## 6. Multi-value `Chart_Label` not reachable via lookup_feature

```
ml.lookup_feature("Subject", "Chart_Label")
  -> DerivaMLFeatureNotFound: Feature not found: Chart_Label on Subject
```

Control: `ml.lookup_feature("Image", "Fundus_Angle")` succeeds.

This **corroborates investigation 10**: the `find_features` /
`lookup_feature` discovery layer misses `Execution_Subject_Chart_Label`
(its compound-key arity is 4 — a value FK participates in the key —
exceeding the `max_arity=3` cap in deriva-py `find_associations`). The
feature exists and is populated (1832 rows per the brief) but is
undiscoverable through the public API, so the oracle cannot target it
without bypassing discovery. **Skipped**, as the brief anticipated.

---

## 7. Verdict for PR #260

**Complicates — does not clear — the merge.** The catalog-27 validation
showed C==A on a single-element-type dataset; this run shows **C ≠ A on
a multi-element-type dataset** (the common case in a real domain like
eye-ai). The bag path raises `DerivaMLDenormalizeUnrelatedAnchor` where
the live catalog returns rows, because the bag Denormalizer enumerates
unrelated sibling element types as hard-fail anchors and the wrappers
delegate with `ignore_unrelated_anchors=False`.

Recommended before merge:
1. Decide the intended semantics of a feature read on a multi-element
   dataset (almost certainly: drop unrelated anchors silently —
   `ignore_unrelated_anchors=True` — since the read is scoped to one
   target + its feature table).
2. Apply it in **both** `Dataset.feature_values` and
   `DatasetBag.feature_values` so the two surfaces agree.
3. Add a regression test with a multi-element-type fixture (≥2 element
   types, feature on one of them) asserting A == C and neither raises.

The secondary results are clean and *do* strengthen confidence in the
shared Denormalizer read itself: rich Image features (multi-column,
float, nullable, datetime) read correctly catalog-side with proper
types and UTC-aware ISO RCT. The problem is narrowly the anchor-guard
behavior on the bag wrapper, not the value materialization.

---

## 8. Limitations / cleanup

- The C ≠ A divergence (§4) was captured on a **CGM** feature via a
  `materialize=False, exclude_tables={"Image"}` bag. The
  `exclude_tables` does **not** confound the result: the live dataset
  carries the identical unrelated anchors (Subject, OCT_DICOM) and A
  tolerates them, so the divergence is in the bag-vs-live anchor
  enumeration, not in which tables were excluded from the bag.
- **C for an Image feature could not be captured** on eye-ai: every
  Image-bearing bag export fails on a **corrupt zero-byte asset**
  (MD5 `d41d8cd9…`, the empty-file hash) → `409 CONFLICT` during
  manifest build (§3). Independent of #260; a catalog-maintenance issue.
- The multi-value `Chart_Label` (datetime + multiple value columns) was
  **unreachable** via `lookup_feature` (`DerivaMLFeatureNotFound`, §6),
  so it could be exercised on neither A nor C — corroborates inv. 10.
- `OCT_Embedding` is empty on eye-ai → no array-column feature available
  to test the array-shape divergence from inv. 08 §8.
- Read-only: no writes to eye-ai. The successful `materialize=False`
  bag download for `6-EKGA` is a pure read (no minid/export persisted —
  `use_minid=True` was unavailable: no `s3_bucket` configured).
- Scratch scripts and logs (`/tmp/eyeai_*.py`, `/tmp/eyeai_bagdl*.py`,
  `/tmp/eyeai_bag4.py`, `/tmp/bagdl*.log`, `/tmp/bag4.log`,
  `/tmp/cgm_bag_info.txt`) and any downloaded bag cache under the
  worktree's deriva-ml working dir deleted after the run (see commit).
