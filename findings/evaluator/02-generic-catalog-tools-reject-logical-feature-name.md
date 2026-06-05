# Generic catalog tools (`count_table`/`query_*`/`get_table`) reject the logical feature name with a misleading "does not exist" error

**Severity:** Medium
**Category:** Skill issue / Doc gap (cross-tier handoff)
**Component:** `deriva-mcp-core` generic catalog tools ↔ `deriva-ml-mcp` feature surface
**Persona finding this consolidates:** `findings/modeler/01` (also referenced by tk-007 §2)

## What the evaluator found

The Modeler filed this, and the Analyst hit it too and "routed around it
rather than re-filing" (tk-007). The evaluator promotes it to a
cross-persona finding because it bit **two** of the three personas at the
exact same seam, which makes it a recurring friction, not a one-off.

A modeler/analyst knows the feature by its **logical** name
`Image_Classification` (that is what the model code uses via
`ml.feature_record_class("Image","Image_Classification")` and what
`deriva_ml_list_features` surfaces first as `feature_name`). But the
generic catalog tools speak only the **physical** table name
`Execution_Image_Image_Classification`:

```
count_table(..., table="Image_Classification")
  -> 409 "Table Image_Classification does not exist in schema ..."
count_table(..., table="Execution_Image_Image_Classification")
  -> 1400  ✓   (evaluator confirmed: 1100 GT + 3×100 predictions)
```

The error is technically correct but points at the wrong conclusion —
"did my feature not get created?" — for a feature that demonstrably
exists and was just written to. The person most likely to reach for the
logical name (the modeler verifying their own run) is the person most
likely to be rattled by "does not exist".

## Severity rationale

Medium: friction that slowed two personas but blocked neither (both
recovered via the `feature_table` field from `deriva_ml_list_features`,
or via the deriva-ml-native `deriva_ml_list_feature_values`). Not High
because the correct path exists and is one field-lookup away; not Low
because it recurred across personas and the error message actively
misleads.

## Suggested mitigations (out of scope for the run; for the user to triage)

1. **Error-message hint (cheapest, highest value):** when `count_table`
   / `get_table` / `query_*` get a table name that matches a known
   feature's logical `feature_name`, append "did you mean the feature
   table `Execution_<Target>_<Name>`?" to the 409.
2. **Doc/skill steer:** the `create-feature` / feature-reading skills
   could state at point-of-use that generic catalog tools need the
   physical `feature_table` name while `deriva_ml_list_feature_values`
   takes the logical name. The distinction exists but is not findable at
   the moment of friction.
3. **Resolution:** have the generic tools resolve a logical feature name
   to its physical table transparently. Larger change; the steering
   principle (prefer the deriva-ml surface) argues against teaching the
   generic tier ML-specific names, so (1)+(2) are the better fit.

## Reproduction

```
count_table(hostname=localhost, catalog_id=69,
            schema="e2e-test-20260605", table="Image_Classification")
# -> 409 "... does not exist"
deriva_ml_list_features(...)   # read feature_table = Execution_Image_Image_Classification
count_table(..., table="Execution_Image_Image_Classification")  # -> 1400
```
