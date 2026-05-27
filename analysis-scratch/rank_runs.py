"""Independent ranking of the Modeler's three runs (XDP, XPR, XZT).

Pulls each prediction CSV from the catalog (via PathBuilder asset
download), joins to ground-truth Image_Classification, and computes
top-1 accuracy + per-class recall from scratch — i.e. without
trusting the emission-time numbers in tk-003.

Run from the e2e worktree (catalog 96 must be reachable):

    DERIVA_ML_ALLOW_DIRTY=true uv run python analysis-scratch/rank_runs.py
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from deriva_ml import DerivaML

ASSET_RIDS = {
    "XDP": "XFM",  # cifar10_quick,  3 epochs
    "XPR": "XRP",  # default_model, 10 epochs
    "XZT": "Y1R",  # cifar10_extended, 50 epochs
}

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ml = DerivaML(hostname="localhost", catalog_id=96, use_minid=False)

    # Ground truth — single, fully labeled feature per tk-001.
    gt_rows = [r.model_dump() for r in ml.feature_values("Image", "Image_Classification")]
    gt_df = pd.DataFrame(gt_rows)
    # Ground truth = rows with no Confidence (manually labeled).
    gt_subset = gt_df[gt_df["Confidence"].isna()][["Image", "Image_Class", "Execution"]]
    gt_lookup = dict(zip(gt_subset["Image"], gt_subset["Image_Class"]))
    print(f"Ground truth labels loaded: {len(gt_lookup)} images "
          f"(execution={gt_subset['Execution'].iloc[0]})")
    print(f"Class distribution (catalog-wide): "
          f"{dict(Counter(gt_subset['Image_Class']))}")

    rankings = []

    for exec_rid, asset_rid in ASSET_RIDS.items():
        asset = ml.lookup_asset(asset_rid)
        path = asset.download(dest_dir=OUT / f"asset_{asset_rid}")
        df = pd.read_csv(path)
        df["True_Class"] = df["Image_RID"].map(gt_lookup)

        n_total = len(df)
        n_matched = df["True_Class"].notna().sum()
        matched = df.dropna(subset=["True_Class"]).copy()
        top1 = (matched["Predicted_Class"] == matched["True_Class"]).mean() * 100

        # Per-class recall.
        per_class = {}
        for cls, sub in matched.groupby("True_Class"):
            per_class[cls] = {
                "n": int(len(sub)),
                "recall_pct": round(
                    (sub["Predicted_Class"] == sub["True_Class"]).mean() * 100, 2
                ),
            }

        # Most-confused pairs (top 5 off-diagonal).
        cm = (
            matched.groupby(["True_Class", "Predicted_Class"]).size()
            .reset_index(name="n")
        )
        confused = cm[cm["True_Class"] != cm["Predicted_Class"]].sort_values(
            "n", ascending=False
        ).head(5)
        confused_pairs = [
            f"{r.True_Class}->{r.Predicted_Class}: {int(r.n)}" for r in confused.itertuples()
        ]

        # Confidence calibration: avg max-prob on correct vs wrong predictions.
        prob_cols = [c for c in matched.columns if c.startswith("prob_")]
        matched["max_prob"] = matched[prob_cols].max(axis=1)
        correct_mask = matched["Predicted_Class"] == matched["True_Class"]
        avg_conf_correct = round(matched.loc[correct_mask, "max_prob"].mean() * 100, 2)
        avg_conf_wrong = round(matched.loc[~correct_mask, "max_prob"].mean() * 100, 2)

        rankings.append({
            "execution_rid": exec_rid,
            "asset_rid": asset_rid,
            "csv_path": str(path),
            "n_predictions": n_total,
            "n_matched_to_gt": int(n_matched),
            "top1_accuracy_pct": round(top1, 2),
            "per_class": per_class,
            "top_confused_pairs": confused_pairs,
            "avg_confidence_on_correct_pct": avg_conf_correct,
            "avg_confidence_on_wrong_pct": avg_conf_wrong,
        })

        print(f"\n=== {exec_rid} (asset {asset_rid}) ===")
        print(f"  predictions: {n_total}, matched-to-gt: {n_matched}")
        print(f"  top-1 accuracy: {top1:.2f}%")
        print(f"  per-class recall: {per_class}")
        print(f"  top confused: {confused_pairs}")
        print(f"  confidence when correct: {avg_conf_correct}%; when wrong: {avg_conf_wrong}%")

    # Save the ranking JSON for the report.
    rankings_sorted = sorted(rankings, key=lambda r: -r["top1_accuracy_pct"])
    (OUT / "rankings.json").write_text(json.dumps(rankings_sorted, indent=2))
    print("\nWrote rankings.json")
    print("\nRanking by top-1 accuracy:")
    for i, r in enumerate(rankings_sorted, 1):
        print(f"  {i}. {r['execution_rid']} ({r['asset_rid']}): "
              f"{r['top1_accuracy_pct']:.2f}%")


if __name__ == "__main__":
    main()
