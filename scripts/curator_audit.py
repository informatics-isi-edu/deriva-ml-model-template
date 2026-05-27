"""Curator audit script for catalog 93 (2026-05-27 e2e run).

Direct deriva-ml channel verification of dataset count, members per
dataset, and Image_Classification feature value distribution. Pair
with MCP-side reports for the cross-channel check required by
test-plan §3.4.

Example:
    DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/curator_audit.py
"""

from __future__ import annotations

import warnings
from collections import Counter

from deriva_ml import DerivaML

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def main() -> None:
    """Audit catalog 93 and print dataset + feature distribution."""
    ml = DerivaML(hostname="localhost", catalog_id="93")

    # Dataset inventory.
    datasets = ml.find_datasets()
    print(f"\n=== DATASETS (direct deriva-ml) ===")
    print(f"Total datasets: {len(datasets)}")
    rows = []
    for ds in datasets:
        rid = ds.dataset_rid
        types = ds.dataset_types
        version = ds.current_version
        members = ds.list_dataset_members()
        counts = {tname: len(rs) for tname, rs in members.items() if rs}
        rows.append((rid, types, version, counts, ds.description[:60]))
    rows.sort(key=lambda r: r[0])
    for rid, types, ver, counts, desc in rows:
        print(f"  {rid} v={ver} types={types} counts={counts}  {desc!r}")

    # Image_Classification feature value distribution.
    print(f"\n=== Image_Classification distribution (direct deriva-ml) ===")
    feature_vals = list(ml.feature_values("Image", "Image_Classification"))
    print(f"Total feature records: {len(feature_vals)}")
    dist = Counter(fv.Image_Class for fv in feature_vals)
    for cls in sorted(dist):
        print(f"  {cls}: {dist[cls]}")
    print(f"\nClass count: {len(dist)}; distinct labels balanced? {len(set(dist.values())) == 1}")


if __name__ == "__main__":
    main()
