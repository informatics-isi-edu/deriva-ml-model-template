"""Analyst arc: rank Developer's training executions (XYG/YAP/XNE).

Loads each execution's prediction_probabilities.csv asset, computes
accuracy against the Image_Classification ground-truth feature, and
emits a ranked table. Cross-verifies against Developer's tk-004
reported numbers.

Run with::

    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/analyst_rank_executions.py

"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import pandas as pd

from deriva_ml import DerivaML

HOST = "localhost"
CATALOG = "93"

# Developer's runs (tk-004): exec_rid -> {variant, predictions_asset_rid,
# expected_test_acc}.
RUNS = {
    "XYG": {"variant": "cifar10_default (TX0, 10 ep, seed 123)",
            "pred_rid": "Y0E", "expected_test_acc": 42.00},
    "YAP": {"variant": "cifar10_regularized (TX0+XEM, 10 ep, seed 2026)",
            "pred_rid": "YCP", "expected_test_acc": 37.33,
            "expected_val_acc": 43.00},
    "XNE": {"variant": "cifar10_quick (WD2, 3 ep, seed 42)",
            "pred_rid": "XQC", "expected_test_acc": 24.00},
}


def main() -> int:
    ml = DerivaML(hostname=HOST, catalog_id=CATALOG)
    print(f"Connected: {ml.host_name}:{ml.catalog_id}")

    # 1. Load ground-truth feature values (Image_Classification).
    print("\n[1/3] Loading Image_Classification feature values...")
    fv = [r.model_dump() for r in ml.feature_values("Image", "Image_Classification")]
    fv_df = pd.DataFrame(fv)
    print(f"  Total feature rows: {len(fv_df)}")
    # GT execution = the one with no confidence (manual labels from loader).
    grouped = fv_df.groupby("Execution").agg(
        n=("Image", "count"), with_conf=("Confidence", lambda x: x.notna().sum())
    )
    print(f"  Per-Execution feature summary:\n{grouped}")
    gt_exec = grouped[grouped["with_conf"] == 0]["n"].idxmax()
    print(f"  Ground-truth execution: {gt_exec}")
    gt_rows = fv_df[(fv_df["Execution"] == gt_exec) & fv_df["Confidence"].isna()]
    gt_lookup = dict(zip(gt_rows["Image"], gt_rows["Image_Class"]))
    print(f"  GT labels: {len(gt_lookup)} (over {gt_rows['Image_Class'].nunique()} classes)")

    # 2. For each run: download the predictions CSV, compute accuracy.
    print("\n[2/3] Downloading prediction CSVs and computing metrics...")
    workdir = Path(tempfile.mkdtemp(prefix="analyst_rank_"))
    rows = []
    for exec_rid, meta in RUNS.items():
        pred_rid = meta["pred_rid"]
        asset = ml.lookup_asset(pred_rid)
        # Use the catalog's get_asset / download_asset API.
        # On the deriva-ml v1.39.4 API, lookup_asset returns an Asset with a
        # download method, but the most portable path is via the
        # Execution.download_asset; do it via the ml.download_asset shim if
        # present.
        try:
            local_path = asset.download(dest_dir=workdir / pred_rid)
        except AttributeError:
            # Fallback: use object metadata + hatrac directly.
            from deriva_ml.core.enums import MLAsset  # noqa: F401
            local_path = ml.download_asset(pred_rid, dest_dir=workdir / pred_rid)

        if isinstance(local_path, list):
            local_path = local_path[0]
        local_path = Path(local_path)
        if local_path.is_dir():
            candidates = list(local_path.rglob("prediction_probabilities.csv"))
            if candidates:
                local_path = candidates[0]
        df = pd.read_csv(local_path)
        n_total = len(df)
        df["True_Class"] = df["Image_RID"].map(gt_lookup)
        matched = df.dropna(subset=["True_Class"])
        n_matched = len(matched)
        acc = (matched["Predicted_Class"] == matched["True_Class"]).mean() * 100.0
        # Per-class breakdown.
        per_class = (
            matched.assign(correct=lambda d: d["Predicted_Class"] == d["True_Class"])
            .groupby("True_Class")["correct"].mean() * 100.0
        )
        rows.append({
            "Exec_RID": exec_rid,
            "Variant": meta["variant"],
            "Pred_RID": pred_rid,
            "Pred_rows": n_total,
            "Matched_to_GT": n_matched,
            "Accuracy_pct": round(acc, 2),
            "Expected_pct": meta["expected_test_acc"],
            "Delta_pct": round(acc - meta["expected_test_acc"], 2),
        })
        print(f"  {exec_rid} [{pred_rid}]: {n_matched}/{n_total} matched,"
              f" acc={acc:.2f}% (expected {meta['expected_test_acc']:.2f}%)")
        print(f"    Per-class accuracy:")
        for cls, v in per_class.sort_values(ascending=False).items():
            print(f"      {cls:>10}: {v:5.1f}%")

    # 3. Emit ranking table.
    print("\n[3/3] Ranked table (by computed accuracy):")
    rank_df = pd.DataFrame(rows).sort_values("Accuracy_pct", ascending=False)
    rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))
    print(rank_df.to_string(index=False))

    out = Path("scripts/_artifacts/analyst_ranked_executions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(out, index=False)
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
