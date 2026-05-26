"""Analyst: verify the 6 viable Developer executions and their prediction CSV assets.

Cross-channel verification (direct deriva-ml Python). Confirms the
ranking-and-asset-RID mapping recorded in tk-004 against the actual catalog
state for catalog 18.

Output: a JSON dump under findings/analyst/_artifacts/exec_verification.json
plus a stdout table.

Usage:
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/analyst_verify_executions.py
"""
from __future__ import annotations

import json
from pathlib import Path

from deriva_ml import DerivaML

HOSTNAME = "localhost"
CATALOG_ID = "18"

# From tk-004 handoff
EXPECTED = {
    "DYC": {"variant": "quick (lr=1e-3, 3 ep)", "predictions_asset": "E0A", "test_acc": 0.28},
    "E4A": {"variant": "extended (lr=1e-3, 50 ep)", "predictions_asset": "E68", "test_acc": 0.24},
    "EC0": {"variant": "lr_sweep child lr=1e-4", "predictions_asset": "EE0", "test_acc": 0.14},
    "EJ0": {"variant": "lr_sweep child lr=1e-3 (best)", "predictions_asset": "EM0", "test_acc": 0.30},
    "ER0": {"variant": "lr_sweep child lr=1e-2", "predictions_asset": "ET0", "test_acc": 0.12},
    "EY0": {"variant": "lr_sweep child lr=1e-1 (diverged)", "predictions_asset": "F00", "test_acc": 0.10},
}


def main() -> None:
    ml = DerivaML(HOSTNAME, CATALOG_ID)
    print(f"Connected to {HOSTNAME}/catalog/{CATALOG_ID}")

    result = {"executions": {}}
    print()
    print(f"{'RID':>5} {'Status':>10} {'Workflow':>10}  Assets")
    print("-" * 80)
    for exec_rid, meta in EXPECTED.items():
        exe = ml.lookup_execution(exec_rid)
        assets = list(exe.list_assets())
        asset_map = {a.filename: a.asset_rid for a in assets}
        exec_asset_map = {a.filename: a.asset_rid for a in assets if a.asset_table == "Execution_Asset"}
        # Find prediction csv (must come from Execution_Asset table)
        pred_rid = None
        for fn, rid in exec_asset_map.items():
            if "prediction" in fn.lower() and fn.endswith(".csv"):
                pred_rid = rid
                break
        # Read workflow rid directly via ermrest
        pb = ml.catalog.getPathBuilder()
        exec_table = pb.schemas["deriva-ml"].tables["Execution"]
        rows = list(exec_table.filter(exec_table.column_definitions["RID"] == exec_rid).entities().fetch())
        workflow_rid_direct = rows[0]["Workflow"] if rows else None
        result["executions"][exec_rid] = {
            "variant": meta["variant"],
            "expected_predictions_asset": meta["predictions_asset"],
            "actual_predictions_asset": pred_rid,
            "match": pred_rid == meta["predictions_asset"],
            "all_assets": asset_map,
            "execution_assets": exec_asset_map,
            "workflow_rid_direct": workflow_rid_direct,
            "expected_test_acc": meta["test_acc"],
        }
        ok = "OK" if pred_rid == meta["predictions_asset"] else "MISMATCH"
        print(f"{exec_rid:>5} {'Uploaded':>10} {workflow_rid_direct or 'None':>10}  pred_csv={pred_rid} ({ok}), #assets={len(assets)}")

    # Validate F40 is skipped
    f40 = ml.lookup_execution("F40")
    f40_assets = list(f40.list_assets())
    f40_exec_assets = [a for a in f40_assets if a.asset_table == "Execution_Asset"]
    result["F40_skip_check"] = {
        "total_assets": len(f40_assets),
        "execution_assets": [(a.asset_rid, a.filename) for a in f40_exec_assets],
    }
    print(f"\nF40 (degenerate, to skip): {len(f40_exec_assets)} Execution_Asset(s) -> {[(a.asset_rid, a.filename) for a in f40_exec_assets]}")

    out_path = Path("findings/analyst/_artifacts/exec_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
