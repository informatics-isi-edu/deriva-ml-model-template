# Duplicate Image_Classification feature rows from retried loader run

**Persona:** Curator
**Phase:** Catalog characterisation — feature-value smell-check

## What happened

Counting `Image_Classification` feature rows on the 1100 Image rows
returns 1600, not 1100. Distribution:

- 600 images carry exactly 1 feature row.
- 500 images carry exactly 2 feature rows.

The 500 doubly-tagged images all carry the **same** `Image_Class` value
across both rows — there is no contradictory ground truth — but the
duplication is real catalog state.

Provenance: the doubly-tagged set comes from execution `854` (500 rows,
2026-05-28 09:52:49, workflow 46T) and execution `HSR` (1100 rows,
2026-05-28 09:54:01, same workflow 46T). Both executions are
`Uploaded`. They are the **first failed loader attempt** (at
`--num-images 500`, the one Phase 0 finding 01 documents) and the
**successful retry** (at `--num-images 1100`) respectively. The first
attempt's images phase succeeded and wrote 500 feature rows; the
datasets phase then failed; the retry at `--num-images 1100` re-ran the
images phase and wrote a fresh 1100 rows over (and including) the same
500 originals.

The 500 images doubly-tagged are precisely the 500 images that were
loaded by the first attempt — they remained in the Image table when
the second attempt ran and added its own 1100 feature rows. The first
attempt's feature rows were never cleaned up.

## Reproduction

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id=27)
ds = ml.catalog.getPathBuilder().schemas["e2e-test-20260528"]
feat = list(ds.Execution_Image_Image_Classification.entities().fetch())
# len(feat) == 1600, but len({r['Image'] for r in feat}) == 1100
```

## Notes

This is **harmless for inference of ground-truth class** (both rows agree
per image), but it has two real downstream consequences:

1. Anything that counts feature rows (`deriva_ml_list_feature_values`,
   `len(ds.feature_table.entities())`) over-reports for the 500
   doubly-tagged images by 1.
2. **It interacts badly with `split_dataset(..., row_per="Execution_Image_Image_Classification", ...)`** — see finding 02. The loader's
   training-derived split (TCC = TCM + TCY, and VAP = VAY + VB8)
   stratifies and partitions feature *rows*, not images. With 250 of
   M16's 550 training images carrying two feature rows, those images
   can land on both sides of the train/test split. That is the
   mechanism behind the leakage in TCM/TCY (33 overlap) and VAY/VB8
   (24 overlap).

Workarounds available to downstream personas:

- When counting "how many labels exist on Image" use
  `len({r['Image'] for r in feat})`, not `len(feat)`.
- When selecting an authoritative class for an image, deduplicate by
  picking the row from execution `HSR` (the full 1100-image pass) and
  ignoring rows from execution `854` — see `create-feature` skill's
  "selectors" section. Both agree, so either picks the same class, but
  filtering to HSR makes the row count match the image count.

The loader could prevent this by deleting prior `Image_Classification`
rows on retry (or by detecting "this image already has a class
recorded" and skipping). Both are scope for fix-pass, not for the
Curator to land mid-arc.

Detected during the 2026-05-28 e2e run with sibling versions:
deriva-ml v1.40.2, deriva-ml-mcp v0.5.9, deriva-mcp-core latest main,
deriva-skills v1.2.4, deriva-ml-skills v1.4.11.
