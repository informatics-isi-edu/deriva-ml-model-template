"""Analyst arc: denormalize parity check on JZ8 (1500 Image members).

This is the headline #246 PagedFetcher row-completeness exercise. The
test plan §3.4 demands: row count matches member count, no spurious
nulls, label distribution matches direct catalog query, no missing
RIDs.

Run with::

    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/analyst_denormalize_parity.py

Outputs a JSON-ish summary to scripts/_artifacts/analyst_denormalize_jz8.txt.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from deriva_ml import DerivaML

HOST = "localhost"
CATALOG = "93"
TARGET_RID = "JZ8"
TARGET_VERSION = "0.1.0.post1.dev3"


def main() -> int:
    out_dir = Path("scripts/_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "analyst_denormalize_jz8.txt"
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    log(f"# Denormalize parity check — {TARGET_RID} v{TARGET_VERSION}")
    log(f"# Catalog: {HOST}:{CATALOG}, deriva-ml v1.39.4 (post-#246 fix)")
    log("")

    ml = DerivaML(hostname=HOST, catalog_id=CATALOG)
    ds = ml.lookup_dataset(TARGET_RID)

    # === Step 1: Direct channel — list_dataset_members ===
    log("[1] Direct: ds.list_dataset_members()")
    members = ds.list_dataset_members()
    # Filter to Image members only.
    image_members = [m for m in members.get("Image", [])] if isinstance(members, dict) else members
    if isinstance(members, dict):
        log(f"  member-type buckets: {sorted(members.keys())}")
        for k, v in members.items():
            log(f"    {k}: {len(v)}")
        image_members = members.get("Image", [])
    else:
        log(f"  flat list length: {len(image_members)}")
    direct_image_rids = {m.get("RID") if isinstance(m, dict) else m.RID for m in image_members}
    log(f"  direct Image RID count: {len(direct_image_rids)}")
    log("")

    # === Step 2: Indirect / new fetcher — get_denormalized_as_dataframe ===
    log("[2] Indirect: ds.get_denormalized_as_dataframe(...)")
    df = ds.get_denormalized_as_dataframe(
        include_tables=["Image", "Execution_Image_Image_Classification"],
        version=TARGET_VERSION,
    )
    log(f"  shape: {df.shape}")
    log(f"  columns: {list(df.columns)}")
    log("")

    # === Step 3: Row-count parity ===
    log("[3] Row-count parity:")
    df_rids = set(df["Image.RID"]) if "Image.RID" in df.columns else set()
    # try other key naming conventions
    image_rid_col = next(
        (c for c in df.columns if c.endswith(".RID") and c.lower().startswith("image")),
        None,
    )
    if image_rid_col:
        df_rids = set(df[image_rid_col].dropna())
    log(f"  denorm distinct Image RIDs: {len(df_rids)}")
    log(f"  direct  distinct Image RIDs: {len(direct_image_rids)}")
    missing = direct_image_rids - df_rids
    extra = df_rids - direct_image_rids
    log(f"  missing (in direct, not denorm): {len(missing)}")
    log(f"  extra   (in denorm, not direct): {len(extra)}")
    if missing:
        log(f"  first 5 missing: {sorted(missing)[:5]}")
    if extra:
        log(f"  first 5 extra: {sorted(extra)[:5]}")
    parity_row_count = (len(missing) == 0 and len(extra) == 0)
    log(f"  PARITY (row set equal): {parity_row_count}")
    log("")

    # === Step 4: Null audit ===
    log("[4] Null audit per column:")
    null_counts = df.isna().sum()
    for col, n in null_counts.items():
        pct = n / len(df) * 100.0 if len(df) else 0.0
        log(f"  {col}: {n} ({pct:.1f}%)")
    log("")

    # === Step 5: Class distribution parity ===
    log("[5] Class distribution parity:")
    class_col = next(
        (c for c in df.columns if c.endswith("Image_Class")),
        None,
    )
    if class_col:
        denorm_dist = df[class_col].value_counts().sort_index()
        log(f"  denorm class distribution ({class_col}):")
        for cls, n in denorm_dist.items():
            log(f"    {cls}: {n}")
    else:
        log("  (no Image_Class column found in denorm output)")

    # Direct: query feature values from the catalog using deriva-ml feature_values.
    fv = [r.model_dump() for r in ml.feature_values("Image", "Image_Classification")]
    fv_df = pd.DataFrame(fv)
    # restrict to JZ8 image set, ground-truth execution (Confidence is null).
    gt_df = fv_df[fv_df["Confidence"].isna() & fv_df["Image"].isin(direct_image_rids)]
    direct_dist = gt_df["Image_Class"].value_counts().sort_index()
    log(f"  direct class distribution (feature_values for JZ8 image set):")
    for cls, n in direct_dist.items():
        log(f"    {cls}: {n}")

    if class_col:
        parity_class = denorm_dist.equals(direct_dist)
        log(f"  PARITY (class dist equal): {parity_class}")
    log("")

    # === Step 6: Class balance sanity ===
    log("[6] Balance check:")
    if class_col:
        balanced = denorm_dist.nunique() == 1
        log(f"  all-classes-equal (denorm): {balanced} (counts: {set(denorm_dist.tolist())})")
    log("")

    # === Step 7: Summary verdict ===
    log("[7] Verdict:")
    verdict = {
        "row_count_parity": parity_row_count,
        "no_missing_rows": len(missing) == 0,
        "no_extra_rows": len(extra) == 0,
        "row_count_denorm": len(df_rids),
        "row_count_direct": len(direct_image_rids),
        "n_columns": df.shape[1],
        "any_null_columns": int((null_counts > 0).sum()),
    }
    log(json.dumps(verdict, indent=2))

    out_path.write_text("\n".join(lines) + "\n")
    log("")
    log(f"Wrote: {out_path}")
    return 0 if parity_row_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
