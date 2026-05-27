"""Cross-channel verify: list all assets attached to execution Y90."""

from deriva_ml import DerivaML

ml = DerivaML(hostname="localhost", catalog_id=96, use_minid=False)

# All output assets of the analysis execution.
exp = ml.lookup_experiment("Y90")
print(f"Execution: Y90 (status: {exp.status if hasattr(exp, 'status') else 'n/a'})")
print(f"Workflow: {exp.workflow_rid if hasattr(exp, 'workflow_rid') else '?'}")
print()

# Get the analysis execution as Execution_Asset rows pointing to it.
pb = ml.catalog.getPathBuilder()
asset_path = pb.schemas["deriva-ml"].tables["Execution_Asset"]
link_path = pb.schemas["deriva-ml"].tables["Execution_Asset_Execution"]

# Find Execution_Asset_Execution rows where Execution = Y90.
links = list(link_path.filter(link_path.Execution == "Y90").entities())
print(f"Linked Execution_Asset rows: {len(links)}")
for link in links:
    asset_rid = link["Execution_Asset"]
    asset_rows = list(asset_path.filter(asset_path.RID == asset_rid).entities())
    if asset_rows:
        a = asset_rows[0]
        print(f"  {asset_rid}: {a.get('Filename') or a.get('URL') or '?'} "
              f"({a.get('Length', 0)} bytes)")
