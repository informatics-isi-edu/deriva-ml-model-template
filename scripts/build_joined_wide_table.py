"""Build the joined wide table of M1G test images x ground truth x per-model predictions.

This is the Analyst-arc canonical artifact: one row per Image_RID in
the M1G held-out partition, columns =

    Image_RID, True_Class,
    W76_pred, W76_conf, W76_prob_<class>...,   # cifar10_quick (3 ep)
    XCE_pred, XCE_conf, XCE_prob_<class>...,   # default_model (10 ep)
    YHP_pred, YHP_conf, YHP_prob_<class>...,   # cifar10_large (20 ep)

Saved to ``docs/reports/joined-wide-table.csv`` for the human-readable
deliverable, and uploaded to the catalog as an ``Execution_Asset`` of
the analysis execution this script creates for provenance.

Ground-truth source: ``Image_Classification`` feature filtered to
execution ``HSR`` (per Curator [tk-001] — the 1100-row successful
loader retry, which covers all of M1G). The shipped roc_analysis
notebook uses a ``Confidence IS NULL`` filter that picks the first
candidate GT execution, which on this catalog is the partial 500-row
attempt ``854`` rather than ``HSR``; see ``findings/analyst/02``.

Usage:
    cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e
    uv run python scripts/build_joined_wide_table.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from deriva_ml import (
    DerivaML,
    ExecutionConfiguration,
    Workflow,
    MLAsset,
    ExecAssetType,
)

# --- catalog wiring (matches src/configs/deriva.py default_deriva) -----------
HOSTNAME = "localhost"
CATALOG_ID = 27

# Toronto-pair training executions and their prediction-CSV assets.
RUNS = [
    ("W76", "W96", "cifar10_quick (3 ep, 32-64-128 ch, batch 128)"),
    ("XCE", "XEE", "cifar10_default (10 ep, 32-64-128 ch, batch 64)"),
    ("YHP", "YKP", "cifar10_large (20 ep, 64-128-256 ch, batch 64)"),
]
EXEC_PREDS_LABEL_FOR = {exec_rid: exec_rid for exec_rid, _, _ in RUNS}

# Ground-truth execution per Curator tk-001 (covers all 1100 catalog images).
GT_EXECUTION = "HSR"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "docs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    committed_csv = out_dir / "joined-wide-table.csv"

    ml = DerivaML(hostname=HOSTNAME, catalog_id=CATALOG_ID)

    # --- create an analysis execution for provenance ------------------------
    # We bundle this script's source under a workflow so the catalog records
    # the git URL + commit hash; the produced wide-table CSV becomes an
    # Execution_Asset linked to the new execution.
    workflow = Workflow(
        name="Analyst arc: joined wide table",
        url="https://github.com/informatics-isi-edu/deriva-ml-model-template/"
        "blob/e2e-test/2026-05-28/scripts/build_joined_wide_table.py",
        workflow_type="ROC Analysis Notebook",
        description=(
            "Build M1G x ground-truth x {W76,XCE,YHP} joined wide table; "
            "filter ground truth to HSR per Curator tk-001 (the shipped "
            "ROC notebook silently picks the partial 500-row GT execution "
            "854 instead)."
        ),
    )
    config = ExecutionConfiguration(
        assets=["W96", "XEE", "YKP"],
        workflow=workflow,
        description=(
            "Analyst arc: materialise the joined wide table for the "
            "Toronto-pair runs (W76 quick / XCE default / YHP large) on M1G. "
            "Filters ground truth to execution HSR (Curator tk-001)."
        ),
    )

    with ml.create_execution(config) as exe:
        # --- ground truth ----------------------------------------------------
        feature_rows = list(ml.feature_values("Image", "Image_Classification"))
        gt_df = pd.DataFrame(r.model_dump() for r in feature_rows)
        gt_df = gt_df[gt_df["Execution"] == GT_EXECUTION][["Image", "Image_Class"]]
        gt_df = gt_df.rename(columns={"Image": "Image_RID", "Image_Class": "True_Class"})
        # The HSR execution writes exactly one row per image; assert that.
        if gt_df["Image_RID"].duplicated().any():
            dups = gt_df[gt_df["Image_RID"].duplicated(keep=False)]
            raise RuntimeError(
                f"GT execution {GT_EXECUTION} has duplicate Image_RIDs:\n{dups}"
            )

        class_names: list[str] = sorted(gt_df["True_Class"].unique())
        if len(class_names) != 10:
            raise RuntimeError(
                f"Expected 10 CIFAR-10 classes, got {len(class_names)}: {class_names}"
            )

        # --- per-model predictions ------------------------------------------
        # The execution context downloaded W96/XEE/YKP into asset_paths.
        prediction_csvs: dict[str, Path] = {}
        for asset_path in exe.asset_paths.get("Execution_Asset", []):
            if asset_path.file_name.name != "prediction_probabilities.csv":
                continue
            asset = ml.lookup_asset(asset_path.asset_rid)
            producer = asset.list_executions(asset_role="Output")[0].execution_rid
            prediction_csvs[producer] = asset_path.file_name

        if set(prediction_csvs) != {r[0] for r in RUNS}:
            raise RuntimeError(
                f"Did not get all three Toronto prediction CSVs. "
                f"Got: {sorted(prediction_csvs)} expected: "
                f"{sorted(r[0] for r in RUNS)}"
            )

        # --- join on Image_RID, restricted to M1G test images ---------------
        # M1G has 550 images; W96/XEE/YKP each emit a row per M1G image.
        # Use the prediction CSV's Image_RID set as the join scope. (If GT
        # is missing for any prediction row, that's a real gap and worth
        # surfacing — assert below.)
        wide = gt_df.set_index("Image_RID").copy()
        for exec_rid, _, _label in RUNS:
            df = pd.read_csv(prediction_csvs[exec_rid]).set_index("Image_RID")
            wide = wide.join(
                df[["Predicted_Class", "Confidence"]].rename(
                    columns={
                        "Predicted_Class": f"{exec_rid}_pred",
                        "Confidence": f"{exec_rid}_conf",
                    }
                ),
                how="inner",  # only keep images present in BOTH GT and predictions
            )
            for c in class_names:
                wide[f"{exec_rid}_prob_{c}"] = df[f"prob_{c}"]

        wide = wide.reset_index()
        # M1G is 550 images; verify the inner join didn't drop anyone.
        if len(wide) != 550:
            raise RuntimeError(
                f"Joined wide table has {len(wide)} rows; expected 550 "
                f"(M1G test partition size). GT={len(gt_df)} rows; "
                f"per-model CSV sizes={ {e: len(pd.read_csv(p)) for e, p in prediction_csvs.items()} }"
            )

        # --- write the committed CSV in docs/reports/ ------------------------
        wide.to_csv(committed_csv, index=False)
        print(f"Wrote {committed_csv} ({len(wide)} rows, {wide.shape[1]} cols)")

        # --- mirror to the execution asset directory for catalog upload ------
        asset_csv = exe.asset_file_path(
            MLAsset.execution_asset,
            "joined-wide-table.csv",
            ExecAssetType.output_file,
        )
        wide.to_csv(asset_csv, index=False)

        # --- compute summary metrics on the full 550-image set ---------------
        # These numbers are what the report ties back to: any reader can
        # re-derive them from joined-wide-table.csv with two pandas lines.
        summary_rows = []
        for exec_rid, asset_rid, label in RUNS:
            preds = wide[f"{exec_rid}_pred"]
            correct = (preds == wide["True_Class"]).sum()
            n = len(wide)
            top1_conf = wide[f"{exec_rid}_conf"].astype(float)
            summary_rows.append(
                {
                    "Execution_RID": exec_rid,
                    "Asset_RID": asset_rid,
                    "Label": label,
                    "Samples": int(n),
                    "Top1_Correct": int(correct),
                    "Top1_Accuracy": round(correct / n, 4),
                    "Mean_Top1_Confidence": round(float(top1_conf.mean()), 4),
                    "Median_Top1_Confidence": round(float(top1_conf.median()), 4),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = exe.asset_file_path(
            MLAsset.execution_asset,
            "joined-wide-table-summary.csv",
            ExecAssetType.output_file,
        )
        summary_df.to_csv(summary_csv, index=False)
        # Also commit a copy in docs/reports/ for the readable deliverable.
        summary_repo = out_dir / "joined-wide-table-summary.csv"
        summary_df.to_csv(summary_repo, index=False)
        print(summary_df.to_string(index=False))

        # --- per-class confusion (counts) for each model --------------------
        # Long format so a domain expert can pivot it any direction:
        # one row per (Execution, True_Class, Predicted_Class, count).
        rows = []
        for exec_rid, asset_rid, _label in RUNS:
            cm = pd.crosstab(
                wide["True_Class"], wide[f"{exec_rid}_pred"],
                rownames=["True_Class"], colnames=["Predicted_Class"], dropna=False,
            ).reindex(index=class_names, columns=class_names, fill_value=0)
            for true_c in class_names:
                for pred_c in class_names:
                    rows.append(
                        {
                            "Execution_RID": exec_rid,
                            "Asset_RID": asset_rid,
                            "True_Class": true_c,
                            "Predicted_Class": pred_c,
                            "Count": int(cm.loc[true_c, pred_c]),
                        }
                    )
        confusion_long = pd.DataFrame(rows)
        confusion_csv = exe.asset_file_path(
            MLAsset.execution_asset,
            "per-class-confusion-long.csv",
            ExecAssetType.output_file,
        )
        confusion_long.to_csv(confusion_csv, index=False)
        confusion_long.to_csv(out_dir / "per-class-confusion-long.csv", index=False)
        print(f"Wrote per-class confusion (long): {len(confusion_long)} rows")

        # --- per-class accuracy (recall) wide -------------------------------
        per_class_rows = []
        for exec_rid, asset_rid, _label in RUNS:
            for c in class_names:
                mask = wide["True_Class"] == c
                n = int(mask.sum())
                hits = int((wide.loc[mask, f"{exec_rid}_pred"] == c).sum())
                per_class_rows.append(
                    {
                        "Execution_RID": exec_rid,
                        "Asset_RID": asset_rid,
                        "Class": c,
                        "N": n,
                        "Correct": hits,
                        "Recall": round(hits / n, 4) if n else 0.0,
                    }
                )
        per_class_df = pd.DataFrame(per_class_rows)
        per_class_csv = exe.asset_file_path(
            MLAsset.execution_asset,
            "per-class-recall.csv",
            ExecAssetType.output_file,
        )
        per_class_df.to_csv(per_class_csv, index=False)
        per_class_df.to_csv(out_dir / "per-class-recall.csv", index=False)
        print(per_class_df.pivot(index="Class", columns="Execution_RID", values="Recall").to_string())

        # Commit all output assets to hatrac + write catalog rows.
        exe.commit_output_assets()
        print(f"\nAnalysis execution committed: {exe.execution_rid}")
        print(f"Citation URL: {ml.cite(exe.execution_rid)}")


if __name__ == "__main__":
    main()
