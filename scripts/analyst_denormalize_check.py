"""Analyst: exercise denormalize on CSA (test partition) and cross-channel verify.

Pulls the wide table via the direct deriva-ml Python API and reconciles
against:
  - dataset.list_dataset_members()
  - ml.feature_values('Image', 'Image_Classification') filtered to CSA members

If any of {row count, member RID set, label distribution} disagree, exits
with code 2 — that becomes a high-severity finding against denormalize.

Note: includes the **feature table name** (`Execution_Image_Image_Classification`),
not the feature name (`Image_Classification`). The describe_denormalized
preview accepts the feature name as a row_per candidate, but `_run`
requires the real table name. That mismatch is a low-severity finding
captured in findings/analyst/02-*.

Usage:
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/analyst_denormalize_check.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from deriva_ml import DerivaML

HOSTNAME = "localhost"
CATALOG_ID = "18"
DATASET_RID = "CSA"
GROUND_TRUTH_EXEC = "854"  # the load-cifar10 bootstrap execution


def main() -> int:
    ml = DerivaML(HOSTNAME, CATALOG_ID)
    ds = ml.lookup_dataset(DATASET_RID)

    # ---- Channel A: direct deriva-ml denormalize, no row_per (one row per feature value) ----
    df = ds.get_denormalized_as_dataframe(
        include_tables=["Image", "Execution_Image_Image_Classification"],
    )
    print(f"denormalize df shape: {df.shape}")
    fv_table_prefix = "Execution_Image_Image_Classification"
    img_rid_col = "Image.RID"
    exec_col = f"{fv_table_prefix}.Execution"
    label_col = f"{fv_table_prefix}.Image_Class"

    # ---- Cardinality breakdown by Execution ----
    print(f"\nrows per Execution:")
    per_exec = df.groupby(exec_col).size().sort_index()
    for e, n in per_exec.items():
        print(f"  {e}: {n}")

    # Ground-truth subset
    gt = df[df[exec_col] == GROUND_TRUTH_EXEC]
    gt_rids = set(gt[img_rid_col].dropna().tolist())
    gt_labels = Counter(gt[label_col].dropna().tolist())
    print(f"\nground truth (Execution={GROUND_TRUTH_EXEC}): {len(gt)} rows, unique image RIDs={len(gt_rids)}")
    print(f"ground truth label distribution: {dict(sorted(gt_labels.items()))}")

    # ---- Channel B: list_dataset_members (dict: member_type -> list of records) ----
    members = ds.list_dataset_members()
    print(f"\nlist_dataset_members returned dict with keys: {list(members.keys())}")
    print(f"  Dataset members: {len(members.get('Dataset', []))}")
    print(f"  File members:    {len(members.get('File', []))}")
    print(f"  Image members:   {len(members.get('Image', []))}")
    image_rows = members.get("Image", [])

    member_rids = set()
    for m in image_rows:
        if isinstance(m, dict):
            rid = m.get("RID")
            if rid:
                member_rids.add(rid)
    print(f"member_rids set size (Image only): {len(member_rids)}")

    # ---- Channel C: feature_values filtered to CSA's image RIDs ----
    all_fv = [r.model_dump() for r in ml.feature_values("Image", "Image_Classification")]
    fv_df = pd.DataFrame(all_fv)
    print(f"\nfeature_values total rows: {len(fv_df)}")
    fv_gt = fv_df[(fv_df["Execution"] == GROUND_TRUTH_EXEC) & (fv_df["Image"].isin(gt_rids))]
    fv_labels = Counter(fv_gt["Image_Class"].tolist())
    print(f"feature_values gt subset for CSA: {len(fv_gt)} rows; labels: {dict(sorted(fv_labels.items()))}")

    # ---- Reconciliation ----
    print("\n=== RECONCILIATION ===")
    checks: dict[str, object] = {}

    checks["denorm_total_rows"] = len(df)
    checks["denorm_gt_row_count"] = len(gt)
    checks["denorm_gt_unique_image_rids"] = len(gt_rids)
    checks["denorm_per_execution_counts"] = {str(k): int(v) for k, v in per_exec.items()}

    checks["passes_gt_row_count_50"] = len(gt) == 50
    print(f"GT row count == 50: {checks['passes_gt_row_count_50']}")

    checks["passes_gt_unique_image_rids_50"] = len(gt_rids) == 50
    print(f"GT unique image RIDs == 50: {checks['passes_gt_unique_image_rids_50']}")

    checks["passes_50_per_class_balanced"] = all(v == 5 for v in gt_labels.values()) and len(gt_labels) == 10
    print(f"GT balanced 5-per-class across 10 classes: {checks['passes_50_per_class_balanced']}")

    rid_set_match = (member_rids == gt_rids) if member_rids else None
    checks["rid_set_match_denorm_vs_members"] = rid_set_match if rid_set_match is not None else "skipped (members not RID-bearing)"
    print(f"denorm GT RIDs == member RIDs: {rid_set_match}")

    label_match = gt_labels == fv_labels
    checks["label_match_denorm_vs_feature_values"] = label_match
    print(f"denorm GT labels == feature_values labels: {label_match}")

    out_path = Path("findings/analyst/_artifacts/denormalize_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checks, indent=2, default=str))
    print(f"\nWrote {out_path}")

    failures = []
    if not checks["passes_gt_row_count_50"]:
        failures.append("GT row count")
    if not checks["passes_gt_unique_image_rids_50"]:
        failures.append("GT unique image RIDs")
    if not checks["passes_50_per_class_balanced"]:
        failures.append("balanced label distribution")
    if rid_set_match is False:
        failures.append("RID set vs member RID set")
    if not label_match:
        failures.append("label distribution vs feature_values")

    if failures:
        print(f"\nFAIL: {failures}")
        return 2
    print("\nPASS: all reconciliation checks agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
