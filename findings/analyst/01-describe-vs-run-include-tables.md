# `describe_denormalized` accepts feature names that `get_denormalized_as_dataframe` rejects

**Persona:** Analyst
**Phase:** Denormalize end-to-end exercise on CSA, 2026-05-26
**Severity:** Low
**Component:** `deriva-ml` `Dataset.describe_denormalized` / `Dataset.get_denormalized_as_dataframe`
(the underlying `Denormalizer` planner vs `Denormalizer._run`)

## What happened

While exercising `deriva_ml_denormalize_dataset` end-to-end on
catalog 18, dataset `CSA`, the Analyst's first reach was to
pass the **feature name** in `include_tables` because that's the
name used everywhere else in the API (`find_features()`,
`feature_values()`, `lookup_feature()`).

```python
ds = ml.lookup_dataset("CSA")
ds.describe_denormalized(
    include_tables=["Image", "Image_Classification"],
    row_per="Image",
)
# Returns successfully:
# {
#   'row_per': 'Image',
#   'row_per_candidates': ['Image', 'Image_Classification'],
#   'estimated_row_count': {'in_scope_row_per_rows': 50, 'orphan_rows': 0, 'total': 50},
#   'anchors': {'total': 50, 'by_type': {'Image': 50}},
#   ...
# }
```

`describe_denormalized` accepts `"Image_Classification"` as a valid
`include_tables` entry AND lists it as a candidate `row_per`. The
preview is internally consistent — anchors, estimated row count,
join path, all populated.

Then the same call to the actual runner fails:

```python
ds.get_denormalized_as_dataframe(
    include_tables=["Image", "Image_Classification"],
    row_per="Image",
)
# raises:
# DerivaMLException: The table Image_Classification doesn't exist.
```

The runner requires the **feature table name** instead:

```python
ds.get_denormalized_as_dataframe(
    include_tables=["Image", "Execution_Image_Image_Classification"],
)
# works: returns 350-row wide table
```

## Reproduction

Against catalog 18 (`e2e-test-20260526`), dataset `CSA`:

```python
from deriva_ml import DerivaML
ml = DerivaML("localhost", "18")
ds = ml.lookup_dataset("CSA")

# Step 1: describe accepts the feature name.
ds.describe_denormalized(
    include_tables=["Image", "Image_Classification"],
    row_per="Image",
)  # OK

# Step 2: run rejects the feature name.
ds.get_denormalized_as_dataframe(
    include_tables=["Image", "Image_Classification"],
    row_per="Image",
)  # DerivaMLException
```

## Impact on the persona's work

Lost ~5 minutes. The error message
("The table Image_Classification doesn't exist") is correct but
not helpful to a user who reasonably believed `describe_denormalized`'s
preview meant the call would work. A user without
`find_features('Image')` muscle memory might not realize the
feature has a separate table name.

The Analyst's deliverables were unaffected — after one
`find_features('Image')` call, the right name
(`Execution_Image_Image_Classification`) was obvious and everything
worked from there.

## Suggested classification

Inconsistency / UX. Either:

1. **Tighten `describe_denormalized`** so it rejects feature names
   that `_run` would reject. Same validation, applied earlier.
2. **Loosen `get_denormalized_as_dataframe`** so it resolves
   feature names to their feature table at call time. Symmetric
   with how `feature_values('Image', 'Image_Classification')`
   already works.

Option 2 is the friendlier user experience — feature *name* is
already the public identifier on the `Image_Classification` API
surface, and the wide-table user is asking the same conceptual
question.

## Notes for the fix-pass

- Code sites:
  `deriva-ml/src/deriva_ml/model/denormalize_planner.py` (the
  `_prepare_wide_table` validation that rejects with
  `name_to_table(t)`), and `deriva-ml/src/deriva_ml/local_db/
  denormalizer.py` (the planner→runner handoff).
- If option 2 is chosen: at the planner's
  `name_to_table(t)` step, before raising, check
  `ml.find_features(target_table).by_name(t)` and substitute the
  feature table name. Keep the original `t` for error-message
  fidelity.
- Test: `tests/dataset/test_denormalize.py` should grow a case
  where `include_tables` carries a feature name and the runner
  resolves it.

## Related

- Not a regression of the §3.4 denormalize verification — the
  underlying contract (row count, RID set, label distribution all
  reconcile against `list_dataset_members` and `feature_values`)
  holds. This is a discoverability/UX nit on the planner→runner
  interface, not a data-integrity bug.
