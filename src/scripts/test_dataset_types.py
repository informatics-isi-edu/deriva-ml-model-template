#!/usr/bin/env python3
"""Test script to verify dataset types are properly downloaded in bags.

This script downloads a split dataset and prints the dataset types for the
downloaded bag and all its children.

Usage:
    uv run src/scripts/test_dataset_types.py
    uv run src/scripts/test_dataset_types.py --dataset 3YM --host localhost --catalog 25
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from deriva_ml import DerivaML


def inspect_bag_database(bag_path: Path) -> None:
    """Inspect the bag's SQLite database for dataset type information."""
    dbase_dir = bag_path / "data" / "dbase"

    print("\n   Inspecting bag database...")
    print(f"   Database directory: {dbase_dir}")

    # Find all .db files
    db_files = list(dbase_dir.glob("*.db"))
    print(f"   Database files: {[f.name for f in db_files]}")

    # Check deriva-ml.db for Dataset and Dataset_Type tables
    deriva_ml_db = dbase_dir / "deriva-ml.db"
    if deriva_ml_db.exists():
        print(f"\n   Examining {deriva_ml_db.name}...")
        conn = sqlite3.connect(deriva_ml_db)
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"   Tables: {tables}")

        # Check Dataset table
        if "Dataset" in tables:
            cursor.execute("SELECT RID, Description FROM Dataset;")
            datasets = cursor.fetchall()
            print(f"\n   Dataset table ({len(datasets)} rows):")
            for rid, desc in datasets:
                print(f"      {rid}: {desc[:50] if desc else '(no description)'}...")

        # Check for Dataset_Type association table
        type_table = None
        for t in tables:
            if "Dataset_" in t and "Type" in t:
                type_table = t
                break

        if type_table:
            print(f"\n   {type_table} table:")
            cursor.execute(f"SELECT * FROM [{type_table}];")
            rows = cursor.fetchall()

            # Get column names
            cursor.execute(f"PRAGMA table_info([{type_table}]);")
            columns = [col[1] for col in cursor.fetchall()]
            print(f"   Columns: {columns}")
            print(f"   Rows ({len(rows)}):")
            for row in rows:
                print(f"      {dict(zip(columns, row))}")
        else:
            print("\n   WARNING: No Dataset_*Type* association table found!")
            print(f"   Available tables with 'Dataset': {[t for t in tables if 'Dataset' in t]}")

        # Check Dataset_Dataset table (parent-child relationships)
        if "Dataset_Dataset" in tables:
            print(f"\n   Dataset_Dataset table (parent-child):")
            cursor.execute("SELECT * FROM Dataset_Dataset;")
            rows = cursor.fetchall()
            cursor.execute("PRAGMA table_info(Dataset_Dataset);")
            columns = [col[1] for col in cursor.fetchall()]
            print(f"   Columns: {columns}")
            print(f"   Rows ({len(rows)}):")
            for row in rows:
                print(f"      {dict(zip(columns, row))}")

        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test dataset types in downloaded bags"
    )
    parser.add_argument("--host", default="localhost", help="DerivaML hostname")
    parser.add_argument("--catalog", default="25", help="Catalog ID")
    parser.add_argument("--dataset", default="3YM", help="Dataset RID to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show database details")
    args = parser.parse_args()

    print("=" * 70)
    print("TEST: Dataset Types in Downloaded Bags")
    print("=" * 70)

    # Connect to catalog
    print("\n1. Connecting to catalog...")
    ml = DerivaML(hostname=args.host, catalog_id=args.catalog)
    print(f"   Connected to {ml.host_name}, catalog {ml.catalog_id}")

    # Lookup the dataset
    print(f"\n2. Looking up dataset {args.dataset}...")
    dataset = ml.lookup_dataset(args.dataset)
    print(f"   Dataset: {dataset}")
    print(f"   Description: {dataset.description}")
    print(f"   Current version: {dataset.current_version}")
    print(f"   Types (from catalog): {dataset.dataset_types}")

    # Also lookup child datasets from catalog for comparison
    print("\n3. Looking up child datasets from catalog...")
    children_from_catalog = dataset.list_dataset_children()
    for child in children_from_catalog:
        print(f"   {child.dataset_rid}: types={child.dataset_types}, desc={child.description[:50] if child.description else '(none)'}...")

    # Download the dataset bag
    print("\n4. Downloading dataset bag...")
    bag = dataset.download_dataset_bag(version=dataset.current_version)
    print(f"   Bag path: {bag.model.bag_path}")

    # Inspect the raw database
    if args.verbose:
        inspect_bag_database(bag.model.bag_path)

    # Print bag dataset types
    print("\n5. Dataset types from downloaded bag:")
    print(f"   Parent ({bag.dataset_rid}): {bag.dataset_types}")

    # Get children and print their types
    print("\n6. Child dataset types from downloaded bag:")
    children = bag.list_dataset_children()
    if not children:
        print("   No child datasets found.")
    else:
        for child in children:
            print(f"   {child.dataset_rid}: {child.dataset_types}")
            print(f"      Description: {child.description}")
            print(f"      Version: {child.current_version}")
            members = child.list_dataset_members()
            for table, recs in members.items():
                if recs:
                    print(f"      Members in {table}: {len(recs)}")

    # Compare catalog vs bag
    print("\n7. Comparison (catalog vs bag):")
    catalog_types = {c.dataset_rid: c.dataset_types for c in children_from_catalog}
    bag_types = {c.dataset_rid: c.dataset_types for c in children}

    for rid in catalog_types:
        cat_t = catalog_types.get(rid, [])
        bag_t = bag_types.get(rid, [])
        match = "✓" if cat_t == bag_t else "✗ MISMATCH"
        print(f"   {rid}: catalog={cat_t}, bag={bag_t} {match}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_datasets = [bag] + children
    issues = []

    for ds in all_datasets:
        if not ds.dataset_types:
            issues.append(f"Dataset {ds.dataset_rid} has empty dataset_types")

    # Check for mismatches
    for rid in catalog_types:
        if catalog_types.get(rid) != bag_types.get(rid):
            issues.append(f"Dataset {rid} types mismatch: catalog={catalog_types[rid]}, bag={bag_types.get(rid, [])}")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All datasets have dataset_types populated and match catalog.")

    print("=" * 70)


if __name__ == "__main__":
    main()
