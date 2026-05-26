# `deriva_ml_list_feature_values` returns `next_after_rid: ""` when `truncated=true`

**Persona:** Curator
**Phase:** Cross-channel verification of `Image_Classification` feature values, 2026-05-26
**Severity:** Low
**Component:** `deriva-ml-mcp` `deriva_ml_list_feature_values` tool

## What happened

During audit of `Image_Classification` feature values on catalog 18,
the Curator fetched the first page via MCP:

```text
mcp__dev-localhost__deriva_ml_list_feature_values(
  hostname=localhost, catalog_id=18,
  table=Image, feature_name=Image_Classification, limit=5
)
=> {"count":5,"truncated":true,"next_after_rid":"","records":[...]}
```

`truncated` is `true` (correct — there are 500 records, 5 returned), but
`next_after_rid` is the empty string. The `getting-started` orientation
states the cursor should be the last RID of the page, opaque, passed
back as `after_rid` to advance.

Notably, every record's `RID` field is also `null`:

```json
{"RID":null,"Execution":"854","Feature_Name":"Image_Classification",
 "Image_Class":"truck","Confidence":null,"Image":"47Y", ...}
```

So the per-row RID is missing AND the cursor cannot be advanced.

## Reproduction

```text
ReadMcpResourceTool(server=dev-localhost,
                   uri=deriva://catalog/localhost/18/ml/datasets)
# Confirm Image_Classification feature exists.

mcp__dev-localhost__deriva_ml_list_feature_values(
  hostname=localhost, catalog_id=18,
  table=Image, feature_name=Image_Classification,
  limit=5, preflight_count=false
)
# Returned next_after_rid="" and per-row RID=null.
```

## Impact on the persona's work

Minor for *this* audit — the Curator only needed an existence + sample
spot-check, not a full enumeration via MCP. The direct deriva-ml Python
path (`ml.find_features('Image')` → `feature_table.entities().fetch()`)
returns all 500 rows cleanly and was used for the cross-channel check.

But for any downstream consumer that *does* want to paginate feature
values via MCP (e.g., a future Analyst or the
`compare-model-runs` flow), the cursor is unusable — passing `""` as
`after_rid` either re-fetches the same page or errors out.

## Suggested classification

Bug (`deriva-ml-mcp` feature value pagination). Sub-question: is the
per-row `RID: null` an intentional projection (feature value rows don't
have RIDs in the usual sense?) or a missing column in the serializer?

## Notes for the fix-pass

- Two questions to disentangle: (a) is `Execution_Image_Image_Classification`
  *supposed* to have a per-row RID column? Inspecting the table directly
  in Python shows the rows do carry their own RIDs — so the MCP layer is
  dropping the column. (b) Even if RID is null, the pagination contract
  says the cursor must be advanceable; returning `""` is worse than
  returning `null` (which `truncated=false` would imply).
- The Analyst's CSA/CS0 work in 2026-05-25 used `execution_rids=` filter
  to limit feature_values queries and may have side-stepped this. Worth
  checking whether the new fix lets unfiltered enumeration via cursor
  work.
- Code site: `deriva-ml-mcp/src/deriva_ml_mcp/tools/feature.py` (the
  `list_feature_values` handler), or wherever the per-row serializer
  decides which columns to project.
