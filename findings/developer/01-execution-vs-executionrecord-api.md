# developer/01 — `lookup_execution()` returns ExecutionRecord, not Execution

**When:** 2026-05-27 (Developer arc, catalog 93)
**Severity:** Low — friction, not a bug.

## What happened

During cross-channel verification I wanted to enumerate the
`Execution_Asset` rows linked to each of my 3 training executions
via the direct deriva-ml Python channel. The obvious-looking
call was:

```python
ex = ml.lookup_execution("YAP")
assets = list(ex.execution_assets())
```

This fails with:

```
AttributeError: 'ExecutionRecord' object has no attribute 'execution_assets'
```

`lookup_execution(rid)` returns an **ExecutionRecord** (a
Pydantic read-only metadata model: status, description,
workflow, timestamps). It is *not* the **Execution** handle the
training script uses, which carries the methods that walk
asset/dataset/feature links.

## Reproduction

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id="93")
ex = ml.lookup_execution("YAP")
type(ex)                     # ExecutionRecord (pydantic.BaseModel)
ex.execution_assets()        # AttributeError
ex.status                    # 'Uploaded' — read-only fields do work
```

The "right" way (and what eventually worked) is to use the
path-builder API:

```python
pb = ml.pathBuilder()        # note: pathBuilder is a method, not a property
ml_schema = pb.schemas["deriva-ml"]
ea_link = ml_schema.tables["Execution_Asset_Execution"]
ea_table = ml_schema.tables["Execution_Asset"]
ex_table = ml_schema.tables["Execution"]
rows = list(
    ex_table.filter(ex_table.RID == "YAP")
            .link(ea_link)
            .link(ea_table)
            .entities().fetch()
)
```

That's the working pattern documented elsewhere in deriva-ml,
but it's not what `lookup_execution`'s return type advertises.

## Impact on the persona's work

Roughly 5 minutes lost discovering that
(a) `lookup_execution` returns ExecutionRecord (a name reused for
two different concepts — the live Execution context manager
versus this read-only summary record),
(b) `pathBuilder` is a method, not a property, and
(c) the right link table is `Execution_Asset_Execution`
(I guessed `Execution_Execution_Asset` first; same end-points,
opposite naming convention from the Execution_Execution table
which is a many-to-many self-link).

Verification eventually succeeded via the path-builder + MCP
cross-check. No deliverable was blocked.

## Suggested classification

`docs/API-surface` — naming clarity. Possible cheap fixes:

1. Rename the read-only model to `ExecutionSummary` or
   `ExecutionMetadata` so it doesn't shadow the term "Execution".
2. Add an `execution_assets()` / `list_assets()` convenience
   method to `ExecutionRecord` that does the path-builder walk
   internally. The data is one foreign-key hop away.
3. The deriva-ml capture-tacit-knowledge or execution-lifecycle
   skill could document the "read-the-execution-after-it-finished"
   recipe explicitly. The patterns I found via grep all assume
   you have the *live* Execution handle from `create_execution`
   in the same process.

## Notes for the fix-pass

Low priority. The path-builder workaround is one extra import
once you find it. Worth a docstring tweak more than a code
change.
