# Finding: MCP resource-read tool unavailable; deriva:// orientation + read resources unreachable

- **Persona:** Curator
- **Date:** 2026-06-01
- **Catalog:** localhost / catalog 2
- **Severity:** Low-Medium (workflow friction; the mandated cold-start could not be performed as written)
- **Category:** Harness / tooling integration

## What I expected

The `using-deriva-mcp` and `deriva-ml-context` skills both require, before the
first MCP call, reading the upstream server's orientation resources:

```
ReadMcpResourceTool(server="dev-localhost", uri="deriva://deriva-ml/concepts")
ReadMcpResourceTool(server="dev-localhost", uri="deriva://deriva-ml/getting-started")
```

and prefer `deriva://catalog/.../deriva-ml/...` **resources** over `*_list_*`
tools for every read-shaped question (cached, page-free, no audit rows).

## What actually happened

No resource-reading tool is exposed in this harness. `ReadMcpResourceTool` /
`ListMcpResourcesTool` are not present in the tool surface, and a `ToolSearch`
for "read mcp resource" returns only unrelated browser/Drive/GitHub tools.
Result:

- The two mandated orientation resources (`deriva://deriva-ml/concepts`,
  `deriva://deriva-ml/getting-started`) could **not** be read at all.
- Every read-side `deriva://catalog/{h}/{c}/deriva-ml/...` resource in the
  resource-vs-tool table is unreachable, so all reads had to route through the
  `deriva_ml_list_*` / `deriva_ml_get_*` / `list_vocabulary_terms` tools
  instead — the exact tools the skills say to avoid for read-shaped questions.

## Repro

In this session: `ToolSearch(query="select:ReadMcpResourceTool,ListMcpResourcesTool")`
→ "No matching deferred tools found". Keyword search for resource-reading
returns no MCP resource tool.

## Impact

- The cold-start discipline the skills are built around cannot be satisfied as
  written; an agent following the skill literally will stall looking for a tool
  that isn't there. I fell back to the `deriva_ml_*` tools, which carry the same
  conventions, so the catalog work was unaffected — but the prescribed
  lower-cost read path (cached, no audit-log entries) was unavailable, and the
  orientation prompts went unread.
- Worth confirming whether this is a harness limitation for this e2e setup
  specifically, or whether the MCP server should also expose the orientation
  material via a callable **tool** / **prompt** (it does expose them as prompts
  `deriva_ml_concepts` / `deriva_ml_getting_started`, but those were not surfaced
  as invokable slash commands here either).

## Workaround applied

Used `deriva_ml_list_datasets`, `deriva_ml_list_dataset_relations`,
`deriva_ml_list_dataset_members`, `deriva_ml_get_feature`,
`deriva_ml_list_features`, `list_vocabulary_terms`, `get_schema`,
`get_catalog_info`, plus direct read-only `deriva-ml` Python for set-algebra
checks. All read-only; no audit-relevant mutations performed for inspection.
