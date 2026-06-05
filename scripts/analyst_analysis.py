"""Analyst evaluation: join the Modeler's three runs to ground truth and score.

Standalone driver for the 2026-06-05 multipersona e2e Analyst arc. It compares
the three capacity-sweep training runs (SR8 quick / T1A default / TAC large) the
Modeler recorded on catalog 69's small labeled split, against the shared
ground-truth labels, and materialises:

  - ``analyst_joined_predictions.csv`` — the wide joined table, one row per
    evaluation image carrying the ground-truth class plus each run's predicted
    class, confidence, correctness flag, and per-class softmax probabilities.
    This is the team deliverable: a domain expert can open it in any tool and
    re-derive every number in the report.
  - ``analyst_run_metrics.csv`` — per-run leaderboard (top-1 accuracy, macro &
    micro ROC AUC, correct count) reconciled against the catalog-recorded
    test_acc.
  - ``analyst_per_class_accuracy.csv`` — per-run per-class accuracy.
  - Confusion-matrix plots (one per run) and a micro-averaged ROC overlay.

The run is captured as a DerivaML *execution* that consumes the three
prediction-CSV assets as inputs, so the lineage of every output traces back to
the runs it scored. All number-crunching lives in
``src/scripts/analyst_join.py`` (pure, RID-free, unit-tested); this driver only
wires catalog RIDs and IO.

Usage (commit the tree first — provenance records the git hash):

    uv run python scripts/analyst_analysis.py

    # Fast dev iteration on a dirty tree (do NOT use for the recording run):
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/analyst_analysis.py --dry-run

The catalog target and the run/asset RIDs are baked in because this is a
catalog-69-specific analysis ([E2E-DROP]). The reusable join/metric logic is in
``src/scripts/analyst_join.py`` and carries no RIDs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the e2e host
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from deriva_ml import DerivaML, MLAsset  # noqa: E402
from deriva_ml.execution import ExecutionConfiguration, Workflow  # noqa: E402

# Make ``src/scripts/analyst_join.py`` importable when run as a standalone
# script (the package root is ``src``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scripts.analyst_join import (  # noqa: E402
    build_joined_table,
    confusion,
    macro_micro_auc,
    micro_roc_curve,
    per_class_accuracy,
    per_run_accuracy,
    split_ground_truth_and_predictions,
)
from models.cifar10_classes import CIFAR10_CLASS_NAMES  # noqa: E402

# --- Catalog-69-specific wiring (e2e-test-20260605) -------------------------
HOSTNAME = "localhost"
CATALOG_ID = "69"
GROUND_TRUTH_EXECUTION = "CVP"  # loader execution; Confidence IS NULL rows

# Modeler's three capacity-sweep runs: execution RID -> (label, prediction-CSV
# asset RID, catalog-recorded final-epoch test_acc %). The recorded test_acc
# comes from tk-004 / the training logs and is what we reconcile against.
RUNS: dict[str, dict] = {
    "SR8": {
        "label": "quick",
        "pred_csv": "ST6",
        "recorded_test_acc": 20.0,
        "config": "cifar10_quick (3 epochs, 32->64, 128h)",
    },
    "T1A": {
        "label": "default",
        "pred_csv": "T38",
        "recorded_test_acc": 26.0,
        "config": "cifar10_small_default (10 epochs, 32->64, 128h)",
    },
    "TAC": {
        "label": "large",
        "pred_csv": "TCA",
        "recorded_test_acc": 24.0,
        "config": "cifar10_small_large (20 epochs, 64->128, 256h)",
    },
}

RANDOM_BASELINE_PCT = 10.0  # 10 balanced classes
TEST_PARTITION = "RR6 (small labeled testing, 100 images)"


def pull_feature_frame(ml: DerivaML) -> pd.DataFrame:
    """Pull the full Image_Classification feature as a DataFrame.

    Args:
        ml: Connected DerivaML client.

    Returns:
        DataFrame with one row per feature value (ground truth + every run's
        predictions), columns include ``Image``, ``Image_Class``,
        ``Confidence``, ``Execution``.
    """
    exec_rids = [GROUND_TRUTH_EXECUTION] + list(RUNS.keys())
    records = [
        r.model_dump()
        for r in ml.feature_values(
            "Image", "Image_Classification", execution_rids=exec_rids
        )
    ]
    return pd.DataFrame(records)


def load_prediction_csvs(execution) -> dict[str, pd.DataFrame]:
    """Download each run's prediction_probabilities.csv into per-RID subdirs.

    Per the compare-model-runs skill: every run names its prediction file
    identically, so each download goes into its own subdirectory to avoid
    collision.

    Args:
        execution: Open DerivaML execution (carries credential-aware download).

    Returns:
        Mapping of run label -> DataFrame of the prediction CSV (one row per
        evaluation image, with ``Image_RID``, ``Predicted_Class``,
        ``Confidence``, ``prob_<class>`` columns).
    """
    base = Path(execution.working_dir) / "per_asset_downloads"
    base.mkdir(parents=True, exist_ok=True)
    csvs: dict[str, pd.DataFrame] = {}
    for rid, meta in RUNS.items():
        dest = base / meta["pred_csv"]
        dest.mkdir(parents=True, exist_ok=True)
        path = execution.download_asset(
            asset_rid=meta["pred_csv"], dest_dir=dest, update_catalog=False
        )
        csvs[meta["label"]] = pd.read_csv(path)
    return csvs


def attach_probabilities(
    joined: pd.DataFrame, pred_csvs: dict[str, pd.DataFrame], classes: list[str]
) -> pd.DataFrame:
    """Widen the joined table with each run's per-class softmax probabilities.

    Args:
        joined: Output of :func:`build_joined_table`.
        pred_csvs: Run label -> prediction CSV DataFrame.
        classes: Class ordering.

    Returns:
        ``joined`` with extra columns ``<label>_prob_<class>`` per run/class.
    """
    out = joined.copy()
    for label, df in pred_csvs.items():
        prob_cols = {f"prob_{c}": f"{label}_prob_{c}" for c in classes}
        slim = df.rename(columns={"Image_RID": "Image", **prob_cols})[
            ["Image", *prob_cols.values()]
        ]
        out = out.merge(slim, on="Image", how="left")
    return out


def render_confusion(
    joined: pd.DataFrame, label: str, classes: list[str], out_path: Path
) -> None:
    """Save a confusion-matrix heatmap for one run (rows=true, cols=pred)."""
    cm = confusion(joined, label, classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix — {label} run")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = int(cm.iat[i, j])
            if v:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    color="white" if v > cm.to_numpy().max() / 2 else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_roc_overlay(
    joined: pd.DataFrame, classes: list[str], out_path: Path
) -> None:
    """Save a micro-averaged one-vs-rest ROC overlay across all runs."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for rid, meta in RUNS.items():
        label = meta["label"]
        prob_df = joined[[f"{label}_prob_{c}" for c in classes]].rename(
            columns={f"{label}_prob_{c}": c for c in classes}
        )
        fpr, tpr, auc_val = micro_roc_curve(joined["true_class"], prob_df, classes)
        ax.plot(fpr, tpr, label=f"{label} (micro-AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Micro-averaged ROC (one-vs-rest) across runs")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_metrics(
    joined: pd.DataFrame, classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the per-run leaderboard and the per-class accuracy table.

    Args:
        joined: Wide joined table with predictions + probabilities.
        classes: Class ordering.

    Returns:
        ``(run_metrics_df, per_class_df)``. ``run_metrics_df`` has one row per
        run with computed accuracy, macro/micro AUC, and the recorded test_acc
        for reconciliation. ``per_class_df`` is class x run accuracy.
    """
    labels = [m["label"] for m in RUNS.values()]
    acc = per_run_accuracy(joined, labels)

    rows = []
    for rid, meta in RUNS.items():
        label = meta["label"]
        prob_df = joined[[f"{label}_prob_{c}" for c in classes]].rename(
            columns={f"{label}_prob_{c}": c for c in classes}
        )
        aucs = macro_micro_auc(joined["true_class"], prob_df, classes)
        computed_pct = round(acc[label] * 100, 1)
        rows.append(
            {
                "run": label,
                "execution_rid": rid,
                "config": meta["config"],
                "n_images": len(joined),
                "correct": int(joined[f"{label}_correct"].sum()),
                "computed_test_acc_pct": computed_pct,
                "recorded_test_acc_pct": meta["recorded_test_acc"],
                "reconciles": computed_pct == meta["recorded_test_acc"],
                "macro_auc": round(aucs["macro_auc"], 4),
                "micro_auc": round(aucs["micro_auc"], 4),
                "random_baseline_pct": RANDOM_BASELINE_PCT,
            }
        )
    run_metrics = (
        pd.DataFrame(rows)
        .sort_values("computed_test_acc_pct", ascending=False)
        .reset_index(drop=True)
    )

    per_class = pd.DataFrame(
        {
            meta["label"]: per_class_accuracy(joined, meta["label"], classes)
            for meta in RUNS.values()
        }
    )
    per_class.index.name = "true_class"
    return run_metrics, per_class.reset_index()


def main() -> None:
    """Run the analysis as a DerivaML execution and commit outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build everything but don't create a catalog execution / upload.",
    )
    args = parser.parse_args()

    classes = list(CIFAR10_CLASS_NAMES)
    ml = DerivaML(hostname=HOSTNAME, catalog_id=CATALOG_ID)

    workflow = Workflow(
        name="Analyst capacity-sweep evaluation",
        workflow_type=["Analysis", "Testing"],
        description=(
            "Join the three capacity-sweep training runs (SR8/T1A/TAC) to "
            "ground truth, compute per-run/per-class accuracy, confusion "
            "matrices, and ROC/AUC, and materialise the joined wide table."
        ),
    )
    config = ExecutionConfiguration(
        workflow=workflow,
        assets=[meta["pred_csv"] for meta in RUNS.values()],
        description=(
            "Analyst evaluation of SR8 (quick) / T1A (default) / TAC (large) "
            f"on {TEST_PARTITION}, scored against ground truth (exec "
            f"{GROUND_TRUTH_EXECUTION})."
        ),
    )

    with ml.create_execution(config, dry_run=args.dry_run) as exe:
        print(f"Execution: {exe.execution_rid}")

        feature_df = pull_feature_frame(ml)
        gt_df, pred_df = split_ground_truth_and_predictions(feature_df)
        print(f"Ground-truth rows: {len(gt_df)}  prediction rows: {len(pred_df)}")

        run_labels = {rid: meta["label"] for rid, meta in RUNS.items()}
        joined = build_joined_table(gt_df, pred_df, run_labels)
        print(f"Joined table: {len(joined)} evaluation images")

        pred_csvs = load_prediction_csvs(exe)
        joined = attach_probabilities(joined, pred_csvs, classes)

        run_metrics, per_class = compute_metrics(joined, classes)
        print("\n=== Per-run leaderboard ===")
        print(run_metrics.to_string(index=False))
        print("\n=== Per-class accuracy ===")
        print(per_class.to_string(index=False))

        # Write output assets.
        joined_path = exe.asset_file_path(
            MLAsset.execution_asset,
            "analyst_joined_predictions.csv",
            description=(
                "Wide joined table: one row per evaluation image with the "
                "ground-truth class and each run's predicted class, "
                "confidence, correctness, and per-class softmax probabilities."
            ),
        )
        joined.to_csv(joined_path, index=False)

        metrics_path = exe.asset_file_path(
            MLAsset.execution_asset,
            "analyst_run_metrics.csv",
            description="Per-run leaderboard with computed-vs-recorded accuracy and ROC AUC.",
        )
        run_metrics.to_csv(metrics_path, index=False)

        per_class_path = exe.asset_file_path(
            MLAsset.execution_asset,
            "analyst_per_class_accuracy.csv",
            description="Per-run per-class top-1 accuracy.",
        )
        per_class.to_csv(per_class_path, index=False)

        for rid, meta in RUNS.items():
            cm_path = exe.asset_file_path(
                MLAsset.execution_asset,
                f"confusion_{meta['label']}.png",
                description=f"Confusion matrix for the {meta['label']} run ({rid}).",
            )
            render_confusion(joined, meta["label"], classes, Path(cm_path))

        roc_path = exe.asset_file_path(
            MLAsset.execution_asset,
            "roc_micro_overlay.png",
            description="Micro-averaged one-vs-rest ROC overlay across the three runs.",
        )
        render_roc_overlay(joined, classes, Path(roc_path))

    # Commit AFTER the with block (execution-lifecycle rule 4): uploads staged
    # bytes and transitions the execution to Uploaded. Idempotent on re-call.
    if not args.dry_run:
        report = exe.commit_output_assets()
        print(f"\nUploaded assets: {report}")
    else:
        print("\n[dry-run] skipped commit_output_assets()")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
