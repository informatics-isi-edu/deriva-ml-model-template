"""Download the Y90 analysis outputs locally for the report."""

from pathlib import Path
from deriva_ml import DerivaML

OUT = Path(__file__).parent / "y90_outputs"
OUT.mkdir(parents=True, exist_ok=True)

ml = DerivaML(hostname="localhost", catalog_id=96, use_minid=False)

# The 10 output assets that Y90 produced.
RIDS = ["YB4", "YB6", "YB8", "YBA", "YBC", "YBE", "YBG", "YBJ", "YCT", "YCW"]

for rid in RIDS:
    asset = ml.lookup_asset(rid)
    path = asset.download(dest_dir=OUT)
    print(f"{rid}: {path}")
