"""Read-only exploration of the three Modeler e2e runs (QK8, QWA, R5C) on PK6.

Analyst arc of the multi-persona e2e run. NOT a recorded execution — this is a
read-only data dive to form an independent judgment about the three runs before
running the recorded ROC notebook. Downloads each run's prediction CSV, joins to
the Image_Classification ground truth, and prints accuracy, per-class metrics,
AUC, confusion-pair structure, and calibration.

Run with:
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/analyst_explore.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from deriva_ml import DerivaML

# Asset RIDs for the three prediction CSVs (from src/configs/assets.py).
RUNS = {
    "QK8 smoke (quick, 3ep)": "QN6",
    "QWA regularized (20ep, dropout)": "QY8",
    "R5C fast_lr (1e-2, 15ep)": "R7A",
}

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main() -> None:
    ml = DerivaML(hostname="localhost", catalog_id="2")
    cache = Path("/tmp/analyst_explore")
    cache.mkdir(parents=True, exist_ok=True)

    # ---- Ground truth from the Image_Classification feature ----
    fv = [r.model_dump() for r in ml.feature_values("Image", "Image_Classification")]
    fdf = pd.DataFrame(fv)
    print(f"Image_Classification feature: {len(fdf)} total rows")
    summary = fdf.groupby("Execution").agg(
        n=("Image", "count"),
        with_conf=("Confidence", lambda x: x.notna().sum()),
    )
    print("\nFeature rows by Execution (n, with_confidence):")
    print(summary.to_string())

    # GT = the Confidence-NULL execution covering the most images.
    gt_mask = summary["with_conf"] == 0
    gt_exec = summary[gt_mask]["n"].idxmax() if gt_mask.any() else summary["n"].idxmax()
    gt_rows = fdf[(fdf["Execution"] == gt_exec) & fdf["Confidence"].isna()][
        ["Image", "Image_Class"]
    ]
    gt = dict(zip(gt_rows["Image"], gt_rows["Image_Class"]))
    print(f"\nGround-truth execution: {gt_exec} ({len(gt)} labels)")

    # ---- Per-run analysis ----
    fingerprints = {}
    rows = []
    confusion = {}
    for label, asset_rid in RUNS.items():
        d = cache / asset_rid
        d.mkdir(exist_ok=True)
        path = ml.lookup_asset(asset_rid).download(d)
        fingerprints[asset_rid] = hashlib.md5(Path(path).read_bytes()).hexdigest()
        df = pd.read_csv(path)
        # Reconcile: how many distinct Source_Label values / rows?
        src = df["Source_Label"].unique().tolist()
        df["True_Class"] = df["Image_RID"].map(gt)
        n_total = len(df)
        df = df.dropna(subset=["True_Class"])
        n_matched = len(df)

        acc = (df["Predicted_Class"] == df["True_Class"]).mean() * 100

        # ROC / AUC (one-vs-rest)
        cls_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        y_idx = df["True_Class"].map(cls_to_idx).values
        y_bin = label_binarize(y_idx, classes=range(len(CLASS_NAMES)))
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        y_score = df[prob_cols].values
        per_class_auc = {}
        for i, c in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            per_class_auc[c] = auc(fpr, tpr)
        micro_fpr, micro_tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        micro_auc = auc(micro_fpr, micro_tpr)
        macro_auc = float(np.mean(list(per_class_auc.values())))

        # Confusion matrix (counts)
        cm = confusion_matrix(df["True_Class"], df["Predicted_Class"], labels=CLASS_NAMES)
        confusion[label] = cm

        # Calibration: mean predicted confidence on correct vs wrong
        correct = df["Predicted_Class"] == df["True_Class"]
        conf_correct = df.loc[correct, "Confidence"].mean()
        conf_wrong = df.loc[~correct, "Confidence"].mean()
        mean_conf = df["Confidence"].mean()

        # Prediction distribution: does the model collapse to few classes?
        pred_counts = df["Predicted_Class"].value_counts().to_dict()
        n_classes_predicted = len(pred_counts)

        rows.append({
            "run": label,
            "asset": asset_rid,
            "source_labels": ",".join(src),
            "n_total": n_total,
            "n_matched": n_matched,
            "accuracy": round(acc, 2),
            "micro_auc": round(micro_auc, 4),
            "macro_auc": round(macro_auc, 4),
            "mean_conf": round(mean_conf, 4),
            "conf_correct": round(conf_correct, 4),
            "conf_wrong": round(conf_wrong, 4) if not np.isnan(conf_wrong) else None,
            "n_classes_predicted": n_classes_predicted,
        })
        print(f"\n{'='*70}\n{label}  (asset {asset_rid})")
        print(f"  Source_Label(s): {src}  |  rows {n_total}, matched GT {n_matched}")
        print(f"  Accuracy: {acc:.2f}%   Micro-AUC: {micro_auc:.4f}   Macro-AUC: {macro_auc:.4f}")
        print(f"  Mean confidence: {mean_conf:.4f}  (correct {conf_correct:.4f} / wrong {conf_wrong:.4f})")
        print(f"  Distinct classes predicted: {n_classes_predicted}/10")
        print(f"  Prediction distribution: {dict(sorted(pred_counts.items(), key=lambda kv: -kv[1]))}")
        print("  Per-class AUC: " + ", ".join(f"{c}={v:.3f}" for c, v in per_class_auc.items()))

    print(f"\n{'='*70}\nCSV fingerprints (must be distinct): {fingerprints}")
    assert len(set(fingerprints.values())) == len(fingerprints), "Identical CSVs!"

    print(f"\n{'='*70}\nLEADERBOARD")
    lb = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    print(lb.to_string(index=False))

    # ---- Confusion-pair structure for the best run ----
    print(f"\n{'='*70}\nCONFUSION MATRICES (rows=true, cols=pred; off-diagonal hot spots)")
    for label, cm in confusion.items():
        print(f"\n{label}")
        cmdf = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        print(cmdf.to_string())
        # Top off-diagonal confusions
        offdiag = []
        for i, t in enumerate(CLASS_NAMES):
            for j, p in enumerate(CLASS_NAMES):
                if i != j and cm[i, j] > 0:
                    offdiag.append((cm[i, j], t, p))
        offdiag.sort(reverse=True)
        print("  Top confusions (true -> pred): " +
              ", ".join(f"{t}->{p}:{n}" for n, t, p in offdiag[:6]))


if __name__ == "__main__":
    main()
