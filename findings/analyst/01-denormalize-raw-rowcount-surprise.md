# `get_denormalized_as_dataframe` row count is N_features, not N_images

**Persona:** Analyst
**Phase:** §3 — Denormalize cross-channel parity exercise on JZ8
**Severity:** Polish (documentation / surprise factor, not a defect)
**Component:** `deriva-ml` `Dataset.get_denormalized_as_dataframe` / `dataset-lifecycle` skill

## What happened

Ran:

```python
df = ds.get_denormalized_as_dataframe(
    include_tables=["Image", "Execution_Image_Image_Classification"],
    version="0.1.0.post1.dev3",
)
# JZ8 has 1500 Image members.
# Expected: df.shape == (1500, ...)
# Actual:   df.shape == (1900, 12)
```

The 1900 is exactly correct given the cardinality contract: every
`Execution_Image_Image_Classification` row contributes a denorm row,
and the catalog had 1500 GT features + 400 prediction features
(150+150+100 from XYG/YAP/XNE) = 1900. The denormalize spec is
explicit about this in the Rule 1-8 docstring.

But for an Analyst trying to do the §3.4 parity check (denorm row
count = dataset member count), the first instinct is "uh oh, 1900 ≠
1500, the fetcher is off by 400." Took me a minute to figure out
that this is correct semantics and the right filter is
`df[df['Execution_Image_Image_Classification.Confidence'].isna()]`.

## Reproduction

Any dataset where the same Image appears in multiple
Execution_Image_*feature_name* rows will produce a denorm rowcount
greater than the Image member count. In this catalog, JZ8 plus the
Developer's three training runs is the canonical reproducer (and
will be again on any future catalog where you train models and
denormalize over the prediction feature).

## Impact on the persona's work

Cost a few minutes of "is the parity check failing?" panic. Routed
around with a `.isna()` filter on Confidence. No deliverable
affected; the parity result is what it was always going to be.

## Suggested classification

Polish / Skill issue (documentation).

## Notes for the fix-pass

Two small things, either of which would close the gap:

1. The `dataset-lifecycle` skill's denormalize section could include
   a "trap: rowcount ≠ member count when a feature has multiple
   values per Image" callout with the exact filter idiom.
2. The Denormalizer docstring could add a worked example that
   matches the analyst's typical case (one row per Image after
   filtering on ground-truth-Confidence-is-null).

Neither is a code defect — the cardinality contract is right, just
not surfaced where the Analyst will look first.
