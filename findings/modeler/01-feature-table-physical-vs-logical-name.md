# Counting feature rows by MCP needs the *physical* feature-table name, not the logical feature name

**Persona:** Modeler
**Phase:** Verifying predictions landed — counting rows in the Image_Classification feature after training runs

## What happened

To confirm my training runs wrote prediction rows, I tried to count the
`Image_Classification` feature table directly with `count_table`, using
the feature's **logical name** (the name a modeler knows it by — it's
`ml.feature_record_class("Image", "Image_Classification")` in the model
code, and `deriva_ml_list_features` reports `feature_name:
"Image_Classification"`):

```
count_table(hostname=localhost, catalog_id=69,
            schema="e2e-test-20260605", table="Image_Classification")
-> {"error": "409 ... Table Image_Classification does not exist in
              schema e2e-test-20260605."}
```

The error is technically correct but reads as alarming ("does not
exist") for a feature that very much does exist and is actively used.
The catch: the **physical** table backing the feature is named
`Execution_Image_Image_Classification` (the deriva-ml feature-table
naming convention: `Execution_<TargetTable>_<FeatureName>`).
`deriva_ml_list_features` reports both, but in different fields:

```
deriva_ml_list_features(...) ->
  {"feature_name": "Image_Classification",          <- what the model code uses
   "target_table": "Image",
   "feature_table": "Execution_Image_Image_Classification"}  <- what count_table/query_* need
```

Using the `feature_table` value works:

```
count_table(..., table="Execution_Image_Image_Classification") -> 1400  ✓
```

## Why this matters

The logical feature name is the one a modeler carries in their head —
it's what they wrote in the model code and what the feature-discovery
tool surfaces first (`feature_name`). The generic catalog tools
(`count_table`, `query_aggregate`, `query_attribute`, `get_table`) only
speak the physical table name. The gap is one field-lookup wide, but the
error message ("Table X does not exist") points at the wrong conclusion
("did my feature not get created?") rather than the right one ("I used
the logical name where the physical name was required"). A modeler
verifying their own run's output is exactly the person most likely to
reach for the logical name and most likely to be rattled by a
"does not exist" on a feature they just wrote to.

## Reproduction

1. On catalog 69 (`e2e-test-20260605`), after any run that touched the
   `Image_Classification` feature.
2. `count_table(..., schema="e2e-test-20260605", table="Image_Classification")`
   -> 409 "Table Image_Classification does not exist".
3. `deriva_ml_list_features(...)` -> read the `feature_table` field
   (`Execution_Image_Image_Classification`).
4. `count_table(..., table="Execution_Image_Image_Classification")` -> 1400.

## Notes

- Routed around it (used the physical `feature_table` name from
  `deriva_ml_list_features`) rather than treating it as a blocker.
  No fix attempted mid-arc.
- Not necessarily a defect — the generic catalog tier operates on
  physical tables by design, and the deriva-ml tier
  (`deriva_ml_list_feature_values`) is the surface that speaks logical
  feature names. The friction is the *cross-tier handoff*: a modeler who
  knows the logical name from the model code lands on a generic tool
  whose error doesn't hint that a name-translation step was needed.
  Possible mitigations (out of scope for this run): have `count_table`
  /`get_table` recognize a logical feature name and either resolve it or
  emit a "did you mean the feature table `Execution_..._...`?" hint; or
  document the `feature_name` vs `feature_table` distinction at the point
  of use.
- The deriva-ml-native path (`deriva_ml_list_feature_values`) does take
  the logical feature name and is the recommended surface for reading
  feature values; this finding is specifically about the *generic
  catalog count/query tools*, which a modeler naturally reaches for when
  they just want a quick row count.
