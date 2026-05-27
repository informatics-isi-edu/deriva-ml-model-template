"""Standalone analyst script: rank the Modeler's triplet and build the
single wide joined table for the e2e 2026-05-27-e run.

Reads three prediction CSVs (asset RIDs Y1M / Z3P / 105R) plus the
ground-truth rows from the Image_Classification feature (loader
execution FZC). Joins them by Image_RID, writes:

  - findings/analyst/wide_joined_K16.csv  -- one row per K16 image,
    all three models' predictions and per-class probabilities
  - findings/analyst/ranking.csv          -- top-1 accuracy + micro-AUC
    per execution
  - findings/analyst/per_class_recall.csv -- recall per class per
    execution, to feed the report's confusion-matrix discussion

This script is intentionally not a notebook execution -- it's the
*derivation* of the numbers the notebook will plot, written so the
Evaluator can re-derive them without trusting the notebook output.

Usage:
    DERIVA_ML_ALLOW_DIRTY=true uv run python findings/analyst/rank_and_join.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from deriva_ml import DerivaML

HOST = "localhost"
CATALOG = "2"
GT_EXECUTION = "FZC"

# (asset_rid, execution_rid, label) for the Family-A triplet.
TRIPLET = [
    ("Y1M", "XZP", "cifar10_quick (3 epochs, 32-64)"),
    ("Z3P", "Z1R", "default_model (10 epochs, 32-64)"),
    ("105R", "103T", "cifar10_large (20 epochs, 64-128)"),
]

OUT_DIR = Path(__file__).parent


def main() -> None:
    ml = DerivaML(hostname=HOST, catalog_id=CATALOG, use_minid=False)

    # ---- Ground truth ----
    # Pull rows from Image_Classification, then scope to the loader
    # execution per tk-003. Confidence is NULL on GT rows.
    feature_rows = [r.model_dump() for r in ml.feature_values("Image", "Image_Classification")]
    feat = pd.DataFrame(feature_rows)
    print(f"Image_Classification rows in catalog: {len(feat)}")
    print(feat.groupby("Execution").size().rename("rows"))
    gt = feat[(feat["Execution"] == GT_EXECUTION) & feat["Confidence"].isna()][["Image", "Image_Class"]]
    gt = gt.rename(columns={"Image": "Image_RID", "Image_Class": "True_Class"})
    assert len(gt) == 1500, f"expected 1500 GT rows, got {len(gt)}"
    assert gt["Image_RID"].is_unique, "GT contains duplicate Image RIDs"
    class_names = sorted(gt["True_Class"].unique())
    n_classes = len(class_names)
    print(f"GT: {len(gt)} rows, {n_classes} classes: {class_names}")

    # ---- Per-model predictions ----
    dfs: dict[str, pd.DataFrame] = {}
    for asset_rid, exec_rid, label in TRIPLET:
        # download asset directly via the asset surface (no execution)
        asset = ml.lookup_asset(asset_rid)
        local_path = asset.download(dest_dir=str(OUT_DIR / "downloads" / asset_rid))
        df = pd.read_csv(local_path)
        # standardize required columns and add provenance prefix
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert prob_cols, f"{asset_rid}: no prob_* columns"
        # use the exec RID short label as prefix
        prefix = f"{exec_rid}"
        keep = df[["Image_RID", "Predicted_Class"] + prob_cols].copy()
        keep = keep.rename(
            columns={
                "Predicted_Class": f"{prefix}_pred",
                **{p: f"{prefix}_{p}" for p in prob_cols},
            }
        )
        dfs[exec_rid] = keep
        print(f"{exec_rid} ({label}): {len(df)} predictions from asset {asset_rid}")

    # ---- Wide join: K16-test side, one row per Image ----
    # Start from the first model's row set (all three were predicted on
    # the same K16 partition, but join inner so any disagreement on the
    # row set shows up as a row-count drop).
    wide = None
    for exec_rid, df in dfs.items():
        wide = df if wide is None else wide.merge(df, on="Image_RID", how="inner")
    assert wide is not None
    # Now bring in GT
    wide = wide.merge(gt, on="Image_RID", how="inner")
    print(f"Wide joined table: {len(wide)} rows, {len(wide.columns)} cols")
    wide_path = OUT_DIR / "wide_joined_K16.csv"
    wide.to_csv(wide_path, index=False)
    print(f"  -> {wide_path}")

    # ---- Ranking: top-1 accuracy + micro-AUC per execution ----
    rank_rows = []
    for asset_rid, exec_rid, label in TRIPLET:
        prefix = exec_rid
        prob_cols = [c for c in wide.columns if c.startswith(f"{prefix}_prob_")]
        # accuracy
        acc = (wide[f"{prefix}_pred"] == wide["True_Class"]).mean()
        # micro AUC (OvR)
        y_true_idx = wide["True_Class"].map({c: i for i, c in enumerate(class_names)}).values
        y_true_bin = label_binarize(y_true_idx, classes=range(n_classes))
        # reorder prob cols to match class_names order
        prob_col_by_class = {c.split("prob_", 1)[1]: c for c in prob_cols}
        y_score = wide[[prob_col_by_class[c] for c in class_names]].values
        micro_auc = roc_auc_score(y_true_bin, y_score, average="micro")
        macro_auc = roc_auc_score(y_true_bin, y_score, average="macro")
        rank_rows.append(
            {
                "Execution_RID": exec_rid,
                "Asset_RID": asset_rid,
                "Label": label,
                "N": len(wide),
                "Accuracy": round(acc * 100, 2),
                "Micro_AUC": round(micro_auc, 4),
                "Macro_AUC": round(macro_auc, 4),
            }
        )
    rank_df = pd.DataFrame(rank_rows).sort_values("Accuracy", ascending=False)
    rank_path = OUT_DIR / "ranking.csv"
    rank_df.to_csv(rank_path, index=False)
    print("\nRanking:")
    print(rank_df.to_string(index=False))
    print(f"\n  -> {rank_path}")

    # ---- Per-class recall per execution ----
    pcr_rows = []
    for asset_rid, exec_rid, label in TRIPLET:
        for cls in class_names:
            mask = wide["True_Class"] == cls
            true_n = int(mask.sum())
            correct = int(((wide.loc[mask, f"{exec_rid}_pred"]) == cls).sum())
            pcr_rows.append(
                {
                    "Execution_RID": exec_rid,
                    "Class": cls,
                    "True_N": true_n,
                    "Correct": correct,
                    "Recall": round(correct / true_n, 4) if true_n else float("nan"),
                }
            )
    pcr_df = pd.DataFrame(pcr_rows)
    pcr_path = OUT_DIR / "per_class_recall.csv"
    pcr_df.to_csv(pcr_path, index=False)
    print("\nPer-class recall (head):")
    print(pcr_df.head(15).to_string(index=False))
    print(f"\n  -> {pcr_path}")

    # ---- Confusion summary: which class pairs confuse which model? ----
    # Top-3 off-diagonal entries per execution
    print("\nTop confusions per model (true -> predicted):")
    for asset_rid, exec_rid, label in TRIPLET:
        cm = (
            wide.groupby(["True_Class", f"{exec_rid}_pred"])
            .size()
            .reset_index(name="N")
        )
        off = cm[cm["True_Class"] != cm[f"{exec_rid}_pred"]].sort_values("N", ascending=False).head(5)
        print(f"\n{exec_rid} ({label}):")
        print(off.to_string(index=False))


if __name__ == "__main__":
    main()
