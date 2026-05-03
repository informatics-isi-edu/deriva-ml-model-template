"""Validate multirun catalog state.

Pass the parent execution RID as the first argument (or it auto-discovers
the most recent multirun parent on catalog 1248).
"""

from __future__ import annotations

import sys

from deriva_ml import DerivaML

ml = DerivaML(hostname="localhost", catalog_id="1337")
pb = ml.pathBuilder()
ml.refresh_schema()

# ---------------------------------------------------------------------------
# Discover the multirun parent execution.
# ---------------------------------------------------------------------------
parent_rid = sys.argv[1] if len(sys.argv) > 1 else None

ex_table = pb.schemas["deriva-ml"].tables["Execution"]
ne_table = pb.schemas["deriva-ml"].tables["Execution_Execution"]

if parent_rid is None:
    # Most-recent execution that appears as a *parent* in Nested_Execution
    nest_rows = list(ne_table.entities().fetch())
    parent_set = {r["Execution"] for r in nest_rows}
    parent_recents = list(
        ex_table.filter(ex_table.RID == ex_table.RID).entities().fetch()
    )
    parent_recents = [
        r for r in parent_recents if r["RID"] in parent_set
    ]
    parent_recents.sort(key=lambda r: r.get("RCT", ""), reverse=True)
    if not parent_recents:
        print("No parent executions found in Nested_Execution; pass the RID explicitly.")
        sys.exit(2)
    parent = parent_recents[0]
    parent_rid = parent["RID"]
else:
    parent = list(ex_table.filter(ex_table.RID == parent_rid).entities().fetch())[0]

print(f"=== Parent execution: {parent_rid} ===")
print(f"  Status:       {parent.get('Status')}")
print(f"  Status_Detail:{parent.get('Status_Detail')}")
print(f"  Workflow:     {parent.get('Workflow')}")
print(f"  Description:  {(parent.get('Description') or '')[:200]}")
print(f"  Duration:     {parent.get('Duration')}")
print()

# ---------------------------------------------------------------------------
# Children via Nested_Execution.
# ---------------------------------------------------------------------------
nest_rows = list(
    ne_table.filter(ne_table.Execution == parent_rid).entities().fetch()
)
print(f"=== Nested children: {len(nest_rows)} ===")
child_rids = [r["Nested_Execution"] for r in nest_rows]
for cr in sorted(child_rids):
    rows = list(ex_table.filter(ex_table.RID == cr).entities().fetch())
    if not rows:
        print(f"  {cr}: (record missing!)")
        continue
    c = rows[0]
    print(
        f"  {cr}  status={c.get('Status'):16}  workflow={c.get('Workflow')}  "
        f"desc={(c.get('Description') or '')[:60]}"
    )

# ---------------------------------------------------------------------------
# Asset/metadata coverage per execution (parent + children).
# ---------------------------------------------------------------------------
all_rids = [parent_rid] + sorted(child_rids)

ea_table = pb.schemas["deriva-ml"].tables["Execution_Asset"]
ea_link = pb.schemas["deriva-ml"].tables["Execution_Asset_Execution"]
em_table = pb.schemas["deriva-ml"].tables["Execution_Metadata"]
em_link = pb.schemas["deriva-ml"].tables["Execution_Metadata_Execution"]


def asset_summary(table, link_table, link_col, exec_rid):
    """Return list of (filename, length, has_url) tuples for a given exec."""
    links = list(
        link_table.filter(link_table.Execution == exec_rid).entities().fetch()
    )
    out = []
    for link in links:
        a_rid = link[link_col]
        rows = list(table.filter(table.RID == a_rid).entities().fetch())
        if rows:
            a = rows[0]
            out.append((a.get("Filename"), a.get("Length"), bool(a.get("URL"))))
    return out


print()
print("=== Per-execution Execution_Metadata files ===")
for rid in all_rids:
    items = asset_summary(em_table, em_link, "Execution_Metadata", rid)
    have_url = sum(1 for _, _, ok in items if ok)
    print(f"  {rid}: {len(items)} files ({have_url} uploaded)")
    for fn, length, ok in items:
        ok_mark = "OK" if ok else "NO_URL"
        print(f"    - [{ok_mark}] {fn} ({length} bytes)")

print()
print("=== Per-execution Execution_Asset files ===")
for rid in all_rids:
    items = asset_summary(ea_table, ea_link, "Execution_Asset", rid)
    have_url = sum(1 for _, _, ok in items if ok)
    print(f"  {rid}: {len(items)} files ({have_url} uploaded)")
    for fn, length, ok in items:
        ok_mark = "OK" if ok else "NO_URL"
        print(f"    - [{ok_mark}] {fn} ({length} bytes)")

# ---------------------------------------------------------------------------
# Image_Classification feature records per execution.
# ---------------------------------------------------------------------------
print()
print("=== Image_Classification predictions per execution ===")
fc = pb.schemas[ml.default_schema].tables["Execution_Image_Image_Classification"]
for rid in all_rids:
    rows = list(fc.filter(fc.Execution == rid).entities().fetch())
    print(f"  {rid}: {len(rows)} predictions")
