# `deriva_ml_get_execution` MCP tool returns `workflow_rid: null` for every execution

**Persona:** Developer
**Phase:** Cross-channel verification of 8 newly-created executions, 2026-05-26
**Severity:** Medium
**Component:** `deriva-ml-mcp` `deriva_ml_get_execution` tool (and `deriva_ml_list_execution_children` — same bug, both surface the same field)

## What happened

After completing two single training runs (`DYC`, `E4A`) and one multirun
parent + 4 children (`EA8` → `EC0, EJ0, ER0, EY0`) plus one degenerate
no-train execution (`F40`), the Developer ran §3.4 cross-channel verification.

Indirect channel (MCP) — every execution returns `workflow_rid: null`:

```text
mcp__dev-localhost__deriva_ml_get_execution(
    hostname=localhost, catalog_id=18, execution_rid="DYC"
)
=> {"rid":"DYC","workflow_rid":null,"status":"Uploaded", ...}
```

Same pattern on `E4A`, `EA8`, `EC0`, `EJ0`, `ER0`, `EY0`, `F40`, AND on the
children of `EA8` returned by `deriva_ml_list_execution_children` — all 8
executions, `workflow_rid: null`.

Direct channel (ermrest via `PathBuilder`) — every execution has
`Workflow="DY6"`:

```text
DYC   Status=Uploaded   Workflow=DY6
E4A   Status=Uploaded   Workflow=DY6
EA8   Status=Uploaded   Workflow=DY6
EC0   Status=Uploaded   Workflow=DY6
EJ0   Status=Uploaded   Workflow=DY6
ER0   Status=Uploaded   Workflow=DY6
EY0   Status=Uploaded   Workflow=DY6
F40   Status=Uploaded   Workflow=DY6
```

Workflow `DY6` is the `cifar10_cnn` workflow created on the first training
run and content-addressed to the deriva-ml-model-template repo URL + commit
SHA at the time of `DYC`. All 8 executions correctly reference it in the
catalog. The MCP serializer is dropping the column.

## Reproduction

Against any catalog with at least one execution that has its `Workflow`
column populated (catalog 18 satisfies this — `DY6` is the workflow row).

```text
mcp__dev-localhost__deriva_ml_get_execution(
    hostname=localhost, catalog_id=18, execution_rid="DYC"
)
# Returns: {"rid":"DYC","workflow_rid":null, ...}
```

Cross-check (direct):

```python
from deriva_ml import DerivaML
ml = DerivaML('localhost', '18')
pb = ml.catalog.getPathBuilder()
exe = pb.schemas['deriva-ml'].tables['Execution']
row = next(exe.filter(exe.column_definitions['RID'] == 'DYC').entities().fetch())
print(row['Workflow'])   # 'DY6'
```

## Impact on the persona's work

Not blocking — the Developer's deliverables don't depend on the MCP
`workflow_rid` field being correct, since the deriva-ml Python path is the
one used inside the training script and that path writes `DY6` correctly.
But it's a *catalog-misreport* on a load-bearing provenance field, which is
exactly the kind of silent disagreement §3.4 was written to surface.

If a downstream tool (a comparison dashboard, an audit script, an analyst
notebook reaching for the MCP surface for "what code ran this?") consumes
`workflow_rid` from `deriva_ml_get_execution`, it will see "no workflow"
for *every* execution in the catalog and conclude the catalog is missing
provenance — when in fact provenance is intact at the ermrest level.

## Suggested classification

Bug (`deriva-ml-mcp` execution serializer).

## Notes for the fix-pass

- The MCP returns `"workflow_rid":null` literally — not "field missing" —
  which means the serializer is *trying* to populate it but reading it
  from a model object where the attribute is unbound. Likely the
  `ExecutionRecord` Pydantic model in deriva-ml has `workflow_rid` as
  an optional field that the lookup path isn't populating from the
  catalog row.
- The `Execution_Record` Pydantic shape exposed by
  `ml.lookup_execution(rid)` directly (Python) has no `workflow_rid`
  field at all (only `execution_rid`, `start_time`, `stop_time`,
  `duration`, `download_duration`, `upload_duration`). So either:
  (a) the MCP layer is augmenting the model with a `workflow_rid`
  field it never fills, or
  (b) the deriva-ml `lookup_execution` Python API is itself missing
  the Workflow FK as a returned field, and the MCP layer is just
  exposing that gap as `null` instead of the real value.
- Most likely site: `deriva-ml-mcp/src/deriva_ml_mcp/tools/execution.py`
  (`get_execution` handler) or wherever the executon record is
  projected to MCP JSON.
- Related: `deriva_ml_list_executions` and `deriva_ml_list_execution_children`
  have the same field on each row and likely suffer the same bug; worth
  checking together.
- Fix the test plan: §3.4 §test for the Developer arc should explicitly
  call out cross-checking `workflow_rid` between channels — without that
  check, this bug stays latent.
