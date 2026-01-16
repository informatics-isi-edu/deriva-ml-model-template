#!/usr/bin/env python3
"""Test script to verify dataset download and restructure_assets functionality.

This script downloads a dataset bag, checks that asset files exist at the expected
paths, and tests the restructure_assets method to verify data is properly organized
for ML training.

Usage:
    uv run scripts/test_restructure_assets.py
    uv run scripts/test_restructure_assets.py --dataset 3YM --host localhost --catalog 25
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

from deriva_ml import DerivaML


def check_database_paths(bag, verbose: bool = True) -> dict:
    """Check the paths stored in the SQLite database.

    Returns a dict with path analysis results.
    """
    results = {
        "db_path": None,
        "filename_column_has_paths": False,
        "sample_db_path": None,
        "sample_actual_path": None,
        "path_mismatch": False,
    }

    model = bag.model
    dbase_dir = model.dbase_path
    results["db_path"] = dbase_dir

    # Find the domain schema database (not deriva-ml.db or main.db)
    db_files = [f for f in os.listdir(dbase_dir) if f.endswith(".db")]
    domain_dbs = [f for f in db_files if f not in ["deriva-ml.db", "main.db"]]

    if not domain_dbs:
        if verbose:
            print("  WARNING: No domain schema database found")
        return results

    # Check the first domain database for Image table
    db_path = os.path.join(dbase_dir, domain_dbs[0])
    if verbose:
        print(f"  Checking database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if Image table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Image';")
    if not cursor.fetchone():
        if verbose:
            print("  WARNING: Image table not found in database")
        conn.close()
        return results

    # Get Image table columns
    cursor.execute("PRAGMA table_info(Image);")
    columns = cursor.fetchall()
    col_names = [c[1] for c in columns]

    # Get a sample row
    cursor.execute("SELECT * FROM Image LIMIT 1;")
    row = cursor.fetchone()

    if row and "Filename" in col_names:
        filename_idx = col_names.index("Filename")
        filename_value = row[filename_idx]
        results["sample_db_path"] = filename_value

        # Check if Filename contains a full path (bug) or just filename (correct)
        if filename_value and "/" in filename_value:
            results["filename_column_has_paths"] = True

            # Check if the path exists locally
            if not os.path.exists(filename_value):
                results["path_mismatch"] = True

                # Try to find the actual path
                rid_idx = col_names.index("RID")
                rid = row[rid_idx]
                actual_path = model.bag_path / "data" / "asset" / rid / "Image"
                if actual_path.exists():
                    actual_files = list(actual_path.iterdir())
                    if actual_files:
                        results["sample_actual_path"] = str(actual_files[0])

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Test dataset download and restructure_assets")
    parser.add_argument("--host", default="localhost", help="DerivaML hostname")
    parser.add_argument("--catalog", default="25", help="Catalog ID")
    parser.add_argument("--dataset", default="3YM", help="Dataset RID to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 70)
    print("TEST: Dataset Download and Restructure Assets")
    print("=" * 70)

    # Connect to catalog
    print("\n1. Connecting to catalog...")
    ml = DerivaML(hostname=args.host, catalog_id=args.catalog)
    print(f"   Connected to {ml.host_name}, catalog {ml.catalog_id}")

    # Lookup the dataset
    print(f"\n2. Looking up dataset {args.dataset}...")
    dataset = ml.lookup_dataset(args.dataset)
    print(f"   Dataset: {dataset}")
    print(f"   Current version: {dataset.current_version}")
    print(f"   Types: {dataset.dataset_types}")

    # Download the dataset bag
    print("\n3. Downloading dataset bag...")
    bag = dataset.download_dataset_bag(version=dataset.current_version)
    print(f"   Dataset RID: {bag.dataset_rid}")
    print(f"   Version: {bag.current_version}")
    print(f"   Description: {bag.description}")
    print(f"   Bag path: {bag.model.bag_path}")

    # Check database paths
    print("\n4. Checking database paths...")
    db_results = check_database_paths(bag, verbose=True)
    if db_results["path_mismatch"]:
        print("   ERROR: Path mismatch detected!")
        print(f"   Database stores: {db_results['sample_db_path']}")
        print(f"   Actual file at: {db_results['sample_actual_path']}")
        print("   This is a bug - the MCP download stored server paths in the database.")
    elif db_results["filename_column_has_paths"]:
        print("   Filename column contains full paths (checking if valid...)")
        if os.path.exists(db_results["sample_db_path"]):
            print("   Paths are valid on this system.")
    else:
        print("   Filename column contains filenames only (correct behavior).")

    # List available tables
    print("\n5. Listing tables in bag...")
    tables = bag.list_tables()
    print(f"   Found {len(tables)} tables")
    if args.verbose:
        for t in tables:
            print(f"      {t}")

    # Get Image table
    print("\n6. Getting Image table...")
    image_data = list(bag.get_table_as_dict("Image"))
    print(f"   Row count: {len(image_data)}")
    if image_data:
        print(f"   Columns: {list(image_data[0].keys())}")

    # Check asset files exist on disk
    print("\n7. Checking asset files on disk...")
    bag_path = bag.model.bag_path
    asset_dir = bag_path / "data" / "asset"

    if asset_dir.exists():
        asset_rids = list(asset_dir.iterdir())
        print(f"   Asset directories: {len(asset_rids)}")

        # Sample check
        files_found = 0
        files_missing = 0
        for rid_dir in asset_rids[:10]:
            image_dir = rid_dir / "Image"
            if image_dir.exists():
                files = list(image_dir.iterdir())
                if files:
                    files_found += 1
                else:
                    files_missing += 1
            else:
                files_missing += 1

        print(f"   Sample check (first 10): {files_found} found, {files_missing} missing")

        # Full count
        total_found = 0
        for rid_dir in asset_rids:
            image_dir = rid_dir / "Image"
            if image_dir.exists() and any(image_dir.iterdir()):
                total_found += 1
        print(f"   Total asset files found: {total_found}/{len(asset_rids)}")
    else:
        print(f"   WARNING: Asset directory not found: {asset_dir}")

    # Check nested datasets
    print("\n8. Checking nested datasets (children)...")
    children = bag.list_dataset_children()
    print(f"   Found {len(children)} child datasets")
    for child in children:
        print(f"      {child.dataset_rid}: types={child.dataset_types}, version={child.current_version}")
        members = child.list_dataset_members()
        for table, recs in members.items():
            if recs:
                print(f"         {table}: {len(recs)} members")

    # Check for Image_Classification features
    print("\n9. Checking Image_Classification feature values...")
    try:
        features = list(bag.list_feature_values("Image", "Image_Classification"))
        print(f"   Found {len(features)} feature values in parent dataset")
    except Exception as e:
        print(f"   Error: {e}")

    for child in children:
        try:
            child_features = list(child.list_feature_values("Image", "Image_Classification"))
            print(f"   Child {child.dataset_rid}: {len(child_features)} feature values")
        except Exception as e:
            print(f"   Child {child.dataset_rid}: Error - {e}")

    # Test restructure_assets
    print("\n10. Testing restructure_assets...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        print(f"    Output dir: {output_dir}")

        def type_selector(types: list[str]) -> str:
            type_lower = [t.lower() for t in types]
            if "training" in type_lower:
                return "training"
            elif "testing" in type_lower:
                return "testing"
            return types[0].lower() if types else "unknown"

        try:
            result = bag.restructure_assets(
                asset_table="Image",
                output_dir=output_dir,
                group_by=["Image_Classification"],
                use_symlinks=True,
                type_selector=type_selector,
            )
            print(f"    restructure_assets returned: {result}")

            # Check output structure
            print("\n11. Checking restructure output structure...")
            if not any(output_dir.iterdir()):
                print("    WARNING: Output directory is empty!")
            else:
                for item in sorted(output_dir.iterdir()):
                    if item.is_dir():
                        subdirs = list(item.iterdir())
                        file_count = sum(
                            len(list(sd.iterdir())) for sd in subdirs if sd.is_dir()
                        )
                        print(f"    {item.name}/ ({len(subdirs)} subdirs, {file_count} files)")
                        if args.verbose:
                            for subdir in sorted(subdirs)[:5]:
                                if subdir.is_dir():
                                    files = list(subdir.iterdir())
                                    print(f"       {subdir.name}/ ({len(files)} files)")
                    else:
                        print(f"    {item.name}")

            # Count totals
            training_dir = output_dir / "training"
            testing_dir = output_dir / "testing"

            train_count = 0
            if training_dir.exists():
                for class_dir in training_dir.iterdir():
                    if class_dir.is_dir():
                        train_count += len(list(class_dir.iterdir()))

            test_count = 0
            if testing_dir.exists():
                for class_dir in testing_dir.iterdir():
                    if class_dir.is_dir():
                        test_count += len(list(class_dir.iterdir()))

            print(f"\n    Training files: {train_count}")
            print(f"    Testing files: {test_count}")
            print(f"    Total restructured: {train_count + test_count}")

        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if db_results["path_mismatch"]:
        issues.append("Database contains wrong paths (MCP server paths instead of local)")
    if not children:
        issues.append("No child datasets found")
    else:
        empty_types = [c for c in children if not c.dataset_types]
        if empty_types:
            issues.append(f"{len(empty_types)} child datasets have empty dataset_types")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues found.")

    print("=" * 70)


if __name__ == "__main__":
    main()
