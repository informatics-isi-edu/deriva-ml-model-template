#!/usr/bin/env python3
"""CIFAR-10 Dataset Loader for DerivaML.

This script downloads the CIFAR-10 image classification dataset from Kaggle
and loads it into a Deriva catalog using the DerivaML library. It demonstrates
how to set up a complete ML data pipeline with proper provenance tracking.

CIFAR-10 is a widely-used benchmark dataset containing 60,000 32x32 color images
in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
The dataset is split into 50,000 training images and 10,000 test images.

What This Script Creates:
    Domain Model:
        - ``Image`` asset table for storing image files
        - ``Image_Class`` vocabulary with the 10 CIFAR-10 class labels
        - ``Image_Classification`` feature linking images to their class labels

    Dataset Hierarchy:
        - ``Complete``: All images (training + testing)
        - ``Split``: Parent dataset containing Training and Testing as children
        - ``Training``: 50,000 labeled training images
        - ``Testing``: 10,000 unlabeled test images
        - ``Small_Split``: Parent dataset for small train/test split (1,000 images)
        - ``Small_Training``: 500 labeled training images (subset)
        - ``Small_Testing``: 500 test images (subset)

    Provenance Tracking:
        - All datasets are created within an Execution context
        - Classification labels are added via a separate Execution
        - Each Execution has a unique RID for reproducibility

Prerequisites:
    1. **Kaggle CLI**: Install and configure with your API credentials::

           pip install kaggle
           # Create ~/.kaggle/kaggle.json with your API key
           # See: https://www.kaggle.com/docs/api#authentication

    2. **7-Zip**: Required to extract CIFAR-10 archives::

           # macOS
           brew install p7zip

           # Ubuntu/Debian
           sudo apt-get install p7zip-full

    3. **Deriva Authentication**: Login to your Deriva server::

           deriva-globus-auth-utils login --host <hostname>

Usage:
    Create a new catalog and load CIFAR-10::

        load-cifar10 --hostname localhost --create-catalog cifar10_demo

    Load into an existing catalog::

        load-cifar10 --hostname ml.derivacloud.org --catalog-id 99

    Load a subset of images for testing::

        load-cifar10 --hostname localhost --create-catalog test --num-images 100

    Dry run (schema setup only, no image download)::

        load-cifar10 --hostname localhost --create-catalog test --dry-run

    Show Chaise URLs for created datasets::

        load-cifar10 --hostname localhost --create-catalog demo --show-urls

Attributes:
    --hostname (str): Deriva server hostname (e.g., 'localhost', 'ml.derivacloud.org').
    --catalog-id (str): Catalog ID to connect to (mutually exclusive with --create-catalog).
    --create-catalog (str): Create a new catalog with this project/schema name
        (mutually exclusive with --catalog-id).
    --domain-schema (str, optional): Domain schema name. Auto-detected if not provided.
    --num-images (int, optional): Limit the number of images to upload. Useful for testing.
        The limit is split evenly between training and testing images.
    --batch-size (int): Number of images to process per batch during upload. Default: 500.
    --dry-run (bool): Set up schema and datasets without downloading/uploading images.
    --show-urls (bool): Display Chaise web interface URLs for each dataset in the summary.

Example:
    >>> # Create catalog and load 100 images for quick testing
    >>> load-cifar10 --hostname localhost --create-catalog cifar10_test --num-images 100

    >>> # Full load into existing catalog
    >>> load-cifar10 --hostname ml.derivacloud.org --catalog-id 52

    This module can also be imported and used programmatically::

        from scripts.load_cifar10 import main
        import argparse
        args = argparse.Namespace(
            hostname='localhost',
            create_catalog='my_catalog',
            catalog_id=None,
            domain_schema=None,
            num_images=100,
            batch_size=500,
            dry_run=False,
            show_urls=True
        )
        main(args)

Note:
    - The Kaggle CIFAR-10 test set does not include labels, so only training
      images receive ``Image_Classification`` feature values.
    - Large uploads (50,000+ images) may take 30+ minutes depending on network speed.
    - The script uses DerivaML's execution system for provenance tracking,
      creating separate executions for data loading and labeling.

See Also:
    - DerivaML documentation: https://github.com/informatics-isi-edu/deriva-ml
    - CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
    - Kaggle CIFAR-10: https://www.kaggle.com/c/cifar-10
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

from deriva.core.ermrest_model import Schema
from deriva_ml import DerivaML
from deriva_ml.core.ermrest import ColumnDefinition, UploadProgress
from deriva_ml.core.enums import BuiltinTypes
from deriva_ml.dataset import VersionPart
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.schema import create_ml_catalog

# =============================================================================
# Logging Configuration
# =============================================================================

# Configure logging with explicit handler to avoid DerivaML overriding root level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler directly to this logger so it works regardless of root logger config
_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.propagate = False  # Don't propagate to root logger

# Also configure the deriva_ml logger to show status messages during execution
_deriva_ml_logger = logging.getLogger("deriva_ml")
_deriva_ml_logger.setLevel(logging.INFO)
_deriva_ml_logger.addHandler(_handler)
_deriva_ml_logger.propagate = False

# Ensure stdout/stderr are unbuffered for real-time output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# =============================================================================
# Constants
# =============================================================================

#: CIFAR-10 class definitions: (name, description, synonyms)
CIFAR10_CLASSES: list[tuple[str, str, list[str]]] = [
    ("airplane", "Fixed-wing aircraft", ["plane", "aeroplane"]),
    ("automobile", "Motor vehicle with four wheels", ["car", "auto"]),
    ("bird", "Feathered flying vertebrate", []),
    ("cat", "Small domestic feline", ["kitten"]),
    ("deer", "Hoofed ruminant mammal", []),
    ("dog", "Domestic canine", ["puppy"]),
    ("frog", "Tailless amphibian", ["toad"]),
    ("horse", "Large domesticated hoofed mammal", ["pony"]),
    ("ship", "Large watercraft", ["boat", "vessel"]),
    ("truck", "Motor vehicle for transporting cargo", ["lorry"]),
]

#: Dataset types used for organizing CIFAR-10 data
DATASET_TYPES: list[tuple[str, str, list[str]]] = [
    ("Complete", "A complete dataset containing all data", ["complete", "entire"]),
    ("Training", "A dataset subset used for model training", ["training", "train", "Train"]),
    ("Testing", "A dataset subset used for model testing/evaluation", ["test", "Test"]),
    ("Split", "A dataset that contains nested dataset splits", ["split"]),
    ("Labeled", "A dataset containing records with ground truth labels", ["labeled", "annotated"]),
    ("Unlabeled", "A dataset containing records without ground truth labels", ["unlabeled", "unannotated"]),
]


# =============================================================================
# Credential and Download Functions
# =============================================================================


def verify_kaggle_credentials() -> bool:
    """Verify that Kaggle API credentials are configured.

    Checks for the existence of ~/.kaggle/kaggle.json which contains
    the API key required for downloading datasets from Kaggle.

    Returns:
        True if credentials file exists, False otherwise.

    Note:
        To configure Kaggle credentials:
        1. Create a Kaggle account at https://www.kaggle.com
        2. Go to Account settings and click "Create New API Token"
        3. Save the downloaded kaggle.json to ~/.kaggle/
        4. Set permissions: chmod 600 ~/.kaggle/kaggle.json
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle credentials not found. Please configure ~/.kaggle/kaggle.json\n"
            "See: https://www.kaggle.com/docs/api#authentication"
        )
        return False
    return True


def download_cifar10(temp_dir: Path) -> Path:
    """Download and extract CIFAR-10 dataset from Kaggle.

    Downloads the CIFAR-10 competition data using the Kaggle CLI,
    then extracts the nested archives (zip and 7z formats).

    Args:
        temp_dir: Temporary directory to download and extract files into.

    Returns:
        Path to the extracted dataset directory containing train/ and test/
        subdirectories with PNG images.

    Raises:
        RuntimeError: If Kaggle download fails or 7z extraction fails.

    Note:
        The CIFAR-10 Kaggle dataset structure:
        - cifar-10.zip (outer archive)
          - train.7z -> train/*.png (50,000 images)
          - test.7z -> test/*.png (10,000 images)
          - trainLabels.csv (image_id -> class mappings)

        Requires 7-zip (p7zip) to be installed for extracting .7z archives.
    """
    download_dir = temp_dir / "cifar10"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CIFAR-10 from Kaggle...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "cifar-10", "-p", str(download_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Kaggle download failed: {result.stderr}")
        raise RuntimeError(f"Failed to download CIFAR-10: {result.stderr}")

    # Extract the outer zip file
    zip_files = list(download_dir.glob("*.zip"))
    if zip_files:
        logger.info("Extracting outer zip archive...")
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(download_dir)

    # CIFAR-10 from Kaggle uses 7z archives for train/test data
    seven_z_files = list(download_dir.glob("*.7z"))
    if seven_z_files:
        logger.info("Extracting 7z archives (train.7z, test.7z)...")
        for seven_z_file in seven_z_files:
            # Try '7z' command first, fall back to '7za'
            result = subprocess.run(
                ["7z", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                result = subprocess.run(
                    ["7za", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to extract {seven_z_file.name}. "
                        "Please install 7-zip: brew install p7zip (macOS) or "
                        "apt-get install p7zip-full (Ubuntu)"
                    )

    return download_dir


def load_train_labels(data_dir: Path) -> dict[str, str]:
    """Load training image labels from trainLabels.csv.

    Args:
        data_dir: Directory containing the extracted CIFAR-10 data.

    Returns:
        Mapping of image ID (filename without extension) to class name.
        Example: {"1": "frog", "2": "truck", ...}

    Note:
        The trainLabels.csv file has two columns: 'id' and 'label'.
        Test images do not have labels in the Kaggle version of CIFAR-10.
    """
    labels = {}
    labels_file = data_dir / "trainLabels.csv"

    if labels_file.exists():
        with open(labels_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["id"]] = row["label"]
    else:
        logger.warning("trainLabels.csv not found")

    return labels


def iter_images(
    data_dir: Path, split: str, labels: dict[str, str]
) -> tuple[Path, str | None, str]:
    """Iterate over images in a dataset split.

    Generator that yields image paths with their class labels and IDs.

    Args:
        data_dir: Directory containing train/ and test/ subdirectories.
        split: Either "train" or "test".
        labels: Mapping of image IDs to class names (from trainLabels.csv).

    Yields:
        Tuple of (image_path, class_name, image_id).
        class_name is None for test images (no labels available).

    Example:
        >>> labels = load_train_labels(data_dir)
        >>> for path, class_name, img_id in iter_images(data_dir, "train", labels):
        ...     print(f"{img_id}: {class_name}")
        1: frog
        2: truck
    """
    if split == "train":
        train_dir = data_dir / "train"
        if train_dir.exists():
            for img_path in sorted(train_dir.glob("*.png")):
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name:
                    yield img_path, class_name, image_id
    else:
        test_dir = data_dir / "test"
        if test_dir.exists():
            for img_path in sorted(test_dir.glob("*.png")):
                image_id = img_path.stem
                yield img_path, None, image_id


# =============================================================================
# Schema Setup Functions
# =============================================================================


def setup_table_annotations(ml: DerivaML) -> None:
    """Configure display annotations for domain tables.

    Sets up Chaise display annotations to improve the web interface:
    - Friendly display names for tables
    - Visible columns configuration
    - Row name patterns for dropdowns
    - Image thumbnail display

    Args:
        ml: Connected DerivaML instance.
    """
    from deriva_ml.model import (
        TableHandle,
        Display,
        VisibleColumns,
        TableDisplay,
        TableDisplayOptions,
        ColumnDisplay,
        ColumnDisplayOptions,
        PseudoColumn,
        SortKey,
    )

    logger.info("Configuring table display annotations...")

    # Configure Image table
    image_table = ml.model.name_to_table("Image")
    if image_table:
        handle = TableHandle(image_table)

        # Set display name and description
        handle.set_annotation(Display(
            name="Images",
            comment="CIFAR-10 32x32 RGB images for classification"
        ))

        # Configure visible columns with thumbnail pseudo-column
        # Create a pseudo-column for thumbnail display
        thumbnail_column = PseudoColumn(
            source="URL",
            markdown_name="Thumbnail",
            display=ColumnDisplayOptions(
                markdown_pattern="[![{{{Filename}}}]({{{URL}}})]({{{URL}}})"
            )
        )

        vc = VisibleColumns()
        vc.compact([
            "RID",
            thumbnail_column,
            "Filename",
            "Length",
        ])
        vc.detailed([
            "RID",
            thumbnail_column,
            "Filename",
            "URL",
            "Length",
            "MD5",
            "Description",
            "RCT",
            "RMT"
        ])
        vc.entry(["Filename", "Description"])
        handle.set_annotation(vc)

        # Set row name pattern (shows filename in dropdowns)
        td = TableDisplay()
        td.row_name("{{{Filename}}}")
        td.compact(TableDisplayOptions(
            row_order=[SortKey("Filename")],
            page_size=25
        ))
        handle.set_annotation(td)

        logger.info("  Configured Image table annotations")

    # Configure Image_Class vocabulary
    image_class_table = ml.model.name_to_table("Image_Class")
    if image_class_table:
        handle = TableHandle(image_class_table)

        # Set display name
        handle.set_annotation(Display(
            name="Image Classes",
            comment="CIFAR-10 classification categories (10 classes)"
        ))

        # Set row name pattern
        td = TableDisplay()
        td.row_name("{{{Name}}}")
        handle.set_annotation(td)

        logger.info("  Configured Image_Class vocabulary annotations")

    # Configure Image_Classification feature table
    # Look up the feature to get its table (avoids hardcoding internal naming convention)
    try:
        feature = ml.lookup_feature("Image", "Image_Classification")
        image_classification_table = feature.feature_table
    except Exception:
        image_classification_table = None
    if image_classification_table:
        handle = TableHandle(image_classification_table)

        # Set display name
        handle.set_annotation(Display(
            name="Image Classifications",
            comment="Class labels and confidence scores for images"
        ))

        # Configure visible columns
        vc = VisibleColumns()
        vc.compact(["RID", "Image", "Image_Class", "Confidence"])
        vc.detailed(["RID", "Image", "Image_Class", "Confidence", "Execution", "RCT"])
        handle.set_annotation(vc)

        # Set row name pattern
        td = TableDisplay()
        td.row_name("{{{Image}}} → {{{Image_Class}}}")
        td.compact(TableDisplayOptions(
            row_order=[SortKey("Image")],
            page_size=50
        ))
        handle.set_annotation(td)

        logger.info("  Configured Image_Classification table annotations")


def setup_domain_model(ml: DerivaML) -> dict[str, Any]:
    """Create the CIFAR-10 domain model in the catalog.

    Sets up the necessary tables, vocabularies, and features for storing
    CIFAR-10 image data with classification labels.

    Args:
        ml: Connected DerivaML instance.

    Returns:
        Status dictionary with keys:
        - vocabulary: {status, name} for Image_Class vocabulary
        - asset_table: {status, table_name} for Image table
        - feature: {status, feature_name} for Image_Classification

    Note:
        This function is idempotent - it checks for existing objects before
        creating them, so it's safe to run multiple times.

        Creates:
        - `Image_Class` vocabulary with 10 CIFAR-10 class terms
        - `Image` asset table for storing image files
        - `Image_Classification` feature linking images to classes
    """
    results = {}

    # Check existing vocabularies across both schemas
    vocabs = [
        v.name
        for schema in [ml.ml_schema, ml.domain_schema]
        for v in ml.model.schemas[schema].tables.values()
        if ml.model.is_vocabulary(v)
    ]

    # Create Image_Class vocabulary if needed
    if "Image_Class" not in vocabs:
        logger.info("Creating Image_Class vocabulary...")
        ml.create_vocabulary(
            vocab_name="Image_Class",
            comment="CIFAR-10 image classification categories",
        )
        results["vocabulary"] = {"status": "created", "name": "Image_Class"}
    else:
        logger.info("Image_Class vocabulary already exists")
        results["vocabulary"] = {"status": "exists", "name": "Image_Class"}

    # Add class terms
    existing_terms = {t.name for t in ml.list_vocabulary_terms("Image_Class")}

    logger.info("Adding CIFAR-10 class terms...")
    for class_name, description, synonyms in CIFAR10_CLASSES:
        if class_name not in existing_terms:
            ml.add_term(
                table="Image_Class",
                term_name=class_name,
                description=description,
                synonyms=synonyms,
            )
            logger.info(f"  Added term: {class_name}")
        else:
            logger.info(f"  Term exists: {class_name}")

    # Check existing tables in domain schema
    tables = [t.name for t in ml.model.schemas[ml.domain_schema].tables.values()]

    # Create Image asset table if needed
    if "Image" not in tables:
        logger.info("Creating Image asset table...")
        ml.create_asset(
            asset_name="Image",
            column_defs=[],
            comment="CIFAR-10 32x32 RGB images",
        )
        results["asset_table"] = {"status": "created", "table_name": "Image"}
    else:
        logger.info("Image asset table already exists")
        results["asset_table"] = {"status": "exists", "table_name": "Image"}

    # Enable Image as dataset element type
    logger.info("Enabling Image as dataset element type...")
    element_types = {t.name for t in ml.list_dataset_element_types()}
    if "Image" not in element_types:
        ml.add_dataset_element_type("Image")

    # Create Image_Classification feature with Confidence score
    # - Image_Class: vocabulary term for the predicted/actual class
    # - Confidence: float value (0-1) representing prediction confidence/probability
    logger.info("Creating Image_Classification feature...")
    confidence_column = ColumnDefinition(
        name="Confidence",
        type=BuiltinTypes.float4,
        nullok=True,
        comment="Prediction confidence/probability (0-1)",
    )
    try:
        ml.create_feature(
            target_table="Image",
            feature_name="Image_Classification",
            comment="CIFAR-10 class label and optional confidence score for each image",
            terms=["Image_Class"],
            metadata=[confidence_column],
            optional=["Confidence"],
        )
        results["feature"] = {"status": "created", "feature_name": "Image_Classification"}
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("Image_Classification feature already exists")
            results["feature"] = {"status": "exists", "feature_name": "Image_Classification"}
        else:
            raise

    return results


def setup_workflow_type(ml: DerivaML) -> None:
    """Ensure required workflow types exist.

    Creates workflow types in the Workflow_Type vocabulary for tracking
    CIFAR-10 data loading and model training executions.

    Args:
        ml: Connected DerivaML instance.
    """
    existing_types = {t.name for t in ml.list_vocabulary_terms("Workflow_Type")}

    if "CIFAR_Data_Load" not in existing_types:
        logger.info("Creating CIFAR_Data_Load workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="CIFAR_Data_Load",
            description="Workflow for loading CIFAR-10 dataset into catalog",
        )

    if "Image Classification" not in existing_types:
        logger.info("Creating Image Classification workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="Image Classification",
            description="Workflows for training and evaluating image classification models",
        )

    if "ROC Analysis Notebook" not in existing_types:
        logger.info("Creating ROC Analysis Notebook workflow type...")
        ml.add_term(
            table="Workflow_Type",
            term_name="ROC Analysis Notebook",
            description="Jupyter notebook that computes ROC curves and AUC scores for classification experiments",
        )


def setup_dataset_types(ml: DerivaML) -> None:
    """Ensure required dataset types exist in Dataset_Type vocabulary.

    Creates the dataset type terms needed for organizing CIFAR-10 data
    into Complete, Training, Testing, and Split datasets.

    Args:
        ml: Connected DerivaML instance.
    """
    logger.info("Setting up dataset types...")

    existing_terms = {t.name for t in ml.list_vocabulary_terms("Dataset_Type")}

    for type_name, description, synonyms in DATASET_TYPES:
        if type_name not in existing_terms:
            try:
                ml.add_term(
                    table="Dataset_Type",
                    term_name=type_name,
                    description=description,
                    synonyms=synonyms,
                )
                logger.info(f"  Added dataset type: {type_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not add dataset type {type_name}: {e}")


# =============================================================================
# Dataset Creation Functions
# =============================================================================


def create_dataset_hierarchy(ml: DerivaML, exe: Any = None) -> dict[str, str]:
    """Create the CIFAR-10 dataset hierarchy.

    Creates datasets organized in a hierarchy:
    - Complete: Contains all images (Labeled type since training images have labels)
    - Split: Parent dataset for train/test split
      - Training: Child of Split, contains training images (Labeled)
      - Testing: Child of Split, contains test images (Unlabeled - no ground truth)
    - Small_Split: Parent dataset for small train/test split (1,000 images)
      - Small_Training: Child of Small_Split, 500 randomly selected training images (Labeled)
      - Small_Testing: Child of Small_Split, 500 randomly selected test images (Unlabeled)
    - Labeled_Split: Parent dataset for split from labeled images only (enables ROC analysis)
      - Labeled_Training: 80% of labeled training images
      - Labeled_Testing: 20% of labeled training images (has ground truth!)
    - Small_Labeled_Split: Smaller version of labeled split for quick testing
      - Small_Labeled_Training: 400 labeled images for training
      - Small_Labeled_Testing: 100 labeled images for testing

    Args:
        ml: Connected DerivaML instance.
        exe: Execution context for provenance tracking. If provided, datasets
            are created within this execution. If None, datasets are created
            directly without execution tracking.

    Returns:
        Mapping of dataset names to their RIDs.

    Note:
        The Split datasets use nested datasets to organize their Training and
        Testing children. The Labeled_Split datasets are created from training
        images only (which have ground truth) to enable metrics like ROC curves.
    """
    datasets = {}

    logger.info("Creating dataset hierarchy...")

    # Helper to create dataset with or without execution context
    def create_ds(description: str, types: list[str]):
        if exe:
            return exe.create_dataset(description=description, dataset_types=types)
        return ml.create_dataset(description=description, dataset_types=types)

    # Create Complete dataset (Labeled since training images have labels)
    complete_ds = create_ds(
        "Complete CIFAR-10 dataset with all labeled images", ["Complete", "Labeled"]
    )
    datasets["complete"] = complete_ds.dataset_rid
    logger.info(f"  Created Complete dataset: {complete_ds.dataset_rid}")

    # Create Split dataset
    split_ds = create_ds(
        "CIFAR-10 dataset split into training and testing subsets", ["Split"]
    )
    datasets["split"] = split_ds.dataset_rid
    logger.info(f"  Created Split dataset: {split_ds.dataset_rid}")

    # Create Training dataset (Labeled - has ground truth)
    training_ds = create_ds(
        "CIFAR-10 training set with 50,000 labeled images", ["Training", "Labeled"]
    )
    datasets["training"] = training_ds.dataset_rid
    logger.info(f"  Created Training dataset: {training_ds.dataset_rid}")

    # Create Testing dataset (Unlabeled - Kaggle test set has no ground truth)
    testing_ds = create_ds(
        "CIFAR-10 testing set (unlabeled)", ["Testing", "Unlabeled"]
    )
    datasets["testing"] = testing_ds.dataset_rid
    logger.info(f"  Created Testing dataset: {testing_ds.dataset_rid}")

    # Add Training and Testing as children of Split
    split_ds.add_dataset_members(
        [training_ds.dataset_rid, testing_ds.dataset_rid], validate=False
    )
    logger.info("  Linked Training and Testing to Split dataset")

    # Create Small_Split dataset (1,000 images total for quick testing)
    small_split_ds = create_ds(
        "Small CIFAR-10 dataset split with 1,000 randomly selected images for testing",
        ["Split"],
    )
    datasets["small_split"] = small_split_ds.dataset_rid
    logger.info(f"  Created Small_Split dataset: {small_split_ds.dataset_rid}")

    # Create Small_Training dataset (500 images, Labeled)
    small_training_ds = create_ds(
        "Small CIFAR-10 training set with 500 randomly selected labeled images for quick testing and development",
        ["Training", "Labeled"],
    )
    datasets["small_training"] = small_training_ds.dataset_rid
    logger.info(f"  Created Small_Training dataset: {small_training_ds.dataset_rid}")

    # Create Small_Testing dataset (500 images, Unlabeled)
    small_testing_ds = create_ds(
        "Small CIFAR-10 testing set with 500 randomly selected images for quick testing and development",
        ["Testing", "Unlabeled"],
    )
    datasets["small_testing"] = small_testing_ds.dataset_rid
    logger.info(f"  Created Small_Testing dataset: {small_testing_ds.dataset_rid}")

    # Add Small_Training and Small_Testing as children of Small_Split
    small_split_ds.add_dataset_members(
        [small_training_ds.dataset_rid, small_testing_ds.dataset_rid], validate=False
    )
    logger.info("  Linked Small_Training and Small_Testing to Small_Split dataset")

    # Create Labeled_Split dataset (from training images only - all have ground truth)
    # This enables ROC analysis since both train and test have labels
    labeled_split_ds = create_ds(
        "CIFAR-10 labeled split: train/test from training images (all have ground truth)",
        ["Split", "Labeled"],
    )
    datasets["labeled_split"] = labeled_split_ds.dataset_rid
    logger.info(f"  Created Labeled_Split dataset: {labeled_split_ds.dataset_rid}")

    # Create Labeled_Training dataset (80% of labeled images)
    labeled_training_ds = create_ds(
        "CIFAR-10 labeled training subset (80% of labeled images, with ground truth)",
        ["Training", "Labeled"],
    )
    datasets["labeled_training"] = labeled_training_ds.dataset_rid
    logger.info(f"  Created Labeled_Training dataset: {labeled_training_ds.dataset_rid}")

    # Create Labeled_Testing dataset (20% of labeled images - has ground truth!)
    labeled_testing_ds = create_ds(
        "CIFAR-10 labeled test subset (20% of labeled images, with ground truth for ROC analysis)",
        ["Testing", "Labeled"],
    )
    datasets["labeled_testing"] = labeled_testing_ds.dataset_rid
    logger.info(f"  Created Labeled_Testing dataset: {labeled_testing_ds.dataset_rid}")

    # Add Labeled_Training and Labeled_Testing as children of Labeled_Split
    labeled_split_ds.add_dataset_members(
        [labeled_training_ds.dataset_rid, labeled_testing_ds.dataset_rid], validate=False
    )
    logger.info("  Linked Labeled_Training and Labeled_Testing to Labeled_Split dataset")

    # Create Small_Labeled_Split dataset (500 labeled images for quick testing)
    small_labeled_split_ds = create_ds(
        "Small CIFAR-10 labeled split with 500 images (all have ground truth)",
        ["Split", "Labeled"],
    )
    datasets["small_labeled_split"] = small_labeled_split_ds.dataset_rid
    logger.info(f"  Created Small_Labeled_Split dataset: {small_labeled_split_ds.dataset_rid}")

    # Create Small_Labeled_Training dataset (400 labeled images)
    small_labeled_training_ds = create_ds(
        "Small CIFAR-10 labeled training subset (400 images with ground truth)",
        ["Training", "Labeled"],
    )
    datasets["small_labeled_training"] = small_labeled_training_ds.dataset_rid
    logger.info(f"  Created Small_Labeled_Training dataset: {small_labeled_training_ds.dataset_rid}")

    # Create Small_Labeled_Testing dataset (100 labeled images)
    small_labeled_testing_ds = create_ds(
        "Small CIFAR-10 labeled test subset (100 images with ground truth for ROC analysis)",
        ["Testing", "Labeled"],
    )
    datasets["small_labeled_testing"] = small_labeled_testing_ds.dataset_rid
    logger.info(f"  Created Small_Labeled_Testing dataset: {small_labeled_testing_ds.dataset_rid}")

    # Add Small_Labeled_Training and Small_Labeled_Testing as children of Small_Labeled_Split
    small_labeled_split_ds.add_dataset_members(
        [small_labeled_training_ds.dataset_rid, small_labeled_testing_ds.dataset_rid], validate=False
    )
    logger.info("  Linked Small_Labeled_Training and Small_Labeled_Testing to Small_Labeled_Split dataset")

    return datasets


# =============================================================================
# Upload Progress Tracking
# =============================================================================


def create_upload_progress_callback(
    total_files: int,
) -> tuple[Callable[[UploadProgress], None], dict[str, Any]]:
    """Create a progress callback for upload monitoring.

    Creates a callback function that logs upload progress at appropriate
    intervals based on the total number of files being uploaded.

    Args:
        total_files: Total number of files to be uploaded.

    Returns:
        Tuple of (callback_function, state_dict).
        - callback_function: Pass to upload_execution_outputs()
        - state_dict: Contains tracking state (for debugging)

    Note:
        Reporting frequency scales with upload size:
        - < 20 files: Report every file
        - 20-100 files: Report every 10%
        - > 100 files: Report every 5%
    """
    state = {
        "last_reported_percent": -1,
        "started": False,
        "callback_count": 0,
    }

    # Determine reporting interval as percentage
    if total_files < 20:
        report_every_percent = max(1, 100 // total_files) if total_files > 0 else 10
    elif total_files <= 100:
        report_every_percent = 10
    else:
        report_every_percent = 5

    def progress_callback(progress: UploadProgress) -> None:
        """Handle upload progress updates."""
        state["callback_count"] += 1

        # Report start once
        if not state["started"]:
            state["started"] = True
            logger.info(
                f"  [Upload] Starting upload (reporting every ~{report_every_percent}%)..."
            )

        # Extract current file number from message if available
        match = re.search(r"Uploading file (\d+) of (\d+)", progress.message)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            percent = progress.percent_complete

            # Round to nearest reporting interval for cleaner output
            report_percent = int(percent // report_every_percent) * report_every_percent

            # Report at interval boundaries
            if report_percent > state["last_reported_percent"]:
                state["last_reported_percent"] = report_percent
                logger.info(f"  [Upload] {percent:.0f}% ({current}/{total} files)")

    return progress_callback, state


# =============================================================================
# Main Image Loading Function
# =============================================================================


def load_images(
    ml: DerivaML,
    data_dir: Path,
    batch_size: int = 500,
    max_images: int | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load CIFAR-10 images into the catalog with full provenance tracking.

    This is the main data loading function that:
    1. Creates an execution for provenance tracking
    2. Registers and uploads images to the Image asset table
    3. Adds classification labels as features (training images only)
    4. Creates the dataset hierarchy within the labeling execution
    5. Assigns images to appropriate datasets (Complete, Training, Testing)

    The datasets are created AFTER features are loaded so that when datasets
    are first created, the images already have their feature values attached.
    This ensures dataset bags include feature data from the start.

    Args:
        ml: Connected DerivaML instance.
        data_dir: Directory containing extracted CIFAR-10 data with train/
            and test/ subdirectories.
        batch_size: Number of images to process per batch when assigning
            to datasets. Defaults to 500.
        max_images: Maximum total images to upload. If specified, the limit
            is split evenly between training and testing images. Useful for
            testing.

    Returns:
        Tuple of (datasets, load_result):
        - datasets: Mapping of dataset names to RIDs
        - load_result: Statistics dict with keys:
          - total_images: Total images uploaded
          - training_images: Number of training images
          - testing_images: Number of test images
          - uploaded_assets: Total assets in Image table

    Note:
        The function creates two separate executions:
        1. Data Load Execution: Uploads images to the catalog
        2. Labeling Execution: Adds Image_Classification features and creates datasets

        This separation allows tracking which execution produced which artifacts.
        Datasets are created in the labeling execution so they include images
        with features already attached.

        Training images are labeled with their class from trainLabels.csv.
        Test images have no labels (Kaggle CIFAR-10 format).
    """
    # Ensure CIFAR_Data_Load workflow type exists
    setup_workflow_type(ml)

    # Create workflow for data loading
    logger.info("Creating execution for data loading...")
    workflow = ml.create_workflow(
        name="CIFAR-10 Data Load",
        workflow_type="CIFAR_Data_Load",
        description="Load CIFAR-10 dataset images into DerivaML catalog",
    )

    # Create execution configuration
    config = ExecutionConfiguration(workflow=workflow)

    # Track images for dataset assignment by their filenames
    train_filenames: list[str] = []
    test_filenames: list[str] = []
    filename_to_class: dict[str, str] = {}

    # Calculate limits for training and testing
    if max_images:
        train_limit = max_images // 2
        test_limit = max_images - train_limit
        logger.info(f"Loading {train_limit} training + {test_limit} testing images")
    else:
        train_limit = None
        test_limit = None

    # Use execution context manager for data loading
    with ml.create_execution(config) as exe:
        logger.info(f"  Execution RID: {exe.execution_rid}")

        # Clear working directory to avoid uploading stale files
        working_dir = exe.working_dir
        if working_dir.exists():
            for item in working_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            logger.info(f"  Cleared working directory: {working_dir}")

        # Load training labels
        labels = load_train_labels(data_dir)
        logger.info(f"Loaded {len(labels)} training labels")

        # Process training images
        logger.info("Registering training images for upload...")
        train_count = 0
        for img_path, class_name, image_id in iter_images(data_dir, "train", labels):
            if class_name is None:
                continue

            if train_limit and train_count >= train_limit:
                break

            # Create unique filename with train_ prefix and class
            new_filename = f"train_{class_name}_{image_id}.png"

            # Register file for upload
            exe.asset_file_path(
                asset_name="Image",
                file_name=str(img_path),
                asset_types=["Image"],
                copy_file=True,
                rename_file=new_filename,
            )

            train_filenames.append(new_filename)
            filename_to_class[new_filename] = class_name
            train_count += 1

            if train_count % 1000 == 0:
                logger.info(f"  Registered {train_count} training images...")

        logger.info(f"  Total training images registered: {train_count}")

        # Process test images (no labels in Kaggle CIFAR-10)
        logger.info("Registering test images for upload...")
        test_count = 0
        for img_path, _, image_id in iter_images(data_dir, "test", labels):
            if test_limit and test_count >= test_limit:
                break

            new_filename = f"test_{image_id}.png"

            exe.asset_file_path(
                asset_name="Image",
                file_name=str(img_path),
                asset_types=["Image"],
                copy_file=True,
                rename_file=new_filename,
            )

            test_filenames.append(new_filename)
            test_count += 1

            if test_count % 1000 == 0:
                logger.info(f"  Registered {test_count} test images...")

        logger.info(f"  Total test images registered: {test_count}")

    # Upload outputs (after context manager exits)
    total_count = train_count + test_count
    logger.info(f"Uploading {total_count} images to catalog (this may take a while)...")

    progress_callback, callback_state = create_upload_progress_callback(total_count)
    upload_result = exe.upload_execution_outputs(
        clean_folder=True, progress_callback=progress_callback
    )

    logger.info("  [Upload] 100% complete")
    logger.debug(f"  [Upload] Callback invoked {callback_state['callback_count']} times")

    uploaded_count = sum(len(files) for files in upload_result.values())
    logger.info(f"  Upload complete: {uploaded_count} files uploaded")
    for asset_type, files in upload_result.items():
        logger.info(f"    {asset_type}: {len(files)} files")

    # Get uploaded image RIDs
    logger.info("Getting uploaded image RIDs...")
    assets = ml.list_assets("Image")
    logger.info(f"  Found {len(assets)} uploaded images")

    # Build filename -> RID mapping
    filename_to_rid = {a["Filename"]: a["RID"] for a in assets}

    # Separate RIDs by dataset membership
    train_rids = [filename_to_rid[f] for f in train_filenames if f in filename_to_rid]
    test_rids = [filename_to_rid[f] for f in test_filenames if f in filename_to_rid]
    all_rids = train_rids + test_rids

    # Add Image_Classification features for training images BEFORE creating datasets
    # This ensures datasets include images with features already attached
    datasets = {}
    label_exe = None

    if train_rids and filename_to_class:
        logger.info("Adding Image_Classification features...")

        ImageClassification = ml.feature_record_class("Image", "Image_Classification")

        # Create separate execution for labeling and dataset creation
        label_workflow = ml.create_workflow(
            name="CIFAR-10 Labeling",
            workflow_type="CIFAR_Data_Load",
            description="Add class labels to CIFAR-10 training images and create datasets",
        )
        label_config = ExecutionConfiguration(workflow=label_workflow)

        with ml.create_execution(label_config) as label_exe:
            logger.info(f"  Labeling execution RID: {label_exe.execution_rid}")

            feature_records = []
            for filename, rid in filename_to_rid.items():
                if filename in filename_to_class:
                    class_name = filename_to_class[filename]
                    feature_records.append(
                        ImageClassification(
                            Image=rid,
                            Image_Class=class_name,
                        )
                    )

            logger.info(f"  Adding {len(feature_records)} classification labels...")
            label_exe.add_features(feature_records)

        label_exe.upload_execution_outputs(clean_folder=True)
        logger.info(f"  Added {len(feature_records)} Image_Classification features")

    # Now create datasets AFTER features are loaded
    # This ensures dataset bags include feature data from the start
    logger.info("Creating dataset hierarchy (after features are loaded)...")

    # Create a new execution for dataset creation if we don't have one from labeling
    if label_exe is None:
        ds_workflow = ml.create_workflow(
            name="CIFAR-10 Dataset Creation",
            workflow_type="CIFAR_Data_Load",
            description="Create CIFAR-10 dataset hierarchy",
        )
        ds_config = ExecutionConfiguration(workflow=ds_workflow)
        with ml.create_execution(ds_config) as ds_exe:
            logger.info(f"  Dataset creation execution RID: {ds_exe.execution_rid}")
            datasets = create_dataset_hierarchy(ml, ds_exe)
        ds_exe.upload_execution_outputs(clean_folder=True)
    else:
        # Create another execution for dataset creation
        ds_workflow = ml.create_workflow(
            name="CIFAR-10 Dataset Creation",
            workflow_type="CIFAR_Data_Load",
            description="Create CIFAR-10 dataset hierarchy",
        )
        ds_config = ExecutionConfiguration(workflow=ds_workflow)
        with ml.create_execution(ds_config) as ds_exe:
            logger.info(f"  Dataset creation execution RID: {ds_exe.execution_rid}")
            datasets = create_dataset_hierarchy(ml, ds_exe)
        ds_exe.upload_execution_outputs(clean_folder=True)

    # Assign images to datasets
    logger.info("Assigning images to datasets...")

    # Add all images to Complete dataset
    if all_rids:
        complete_ds = ml.lookup_dataset(datasets["complete"])
        logger.info("  Adding images to Complete dataset...")
        added = 0
        for i in range(0, len(all_rids), batch_size):
            batch = all_rids[i : i + batch_size]
            complete_ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
            logger.info(f"    Added {added}/{len(all_rids)} images")

    # Add training images to Training dataset
    if train_rids:
        training_ds = ml.lookup_dataset(datasets["training"])
        logger.info("  Adding images to Training dataset...")
        added = 0
        for i in range(0, len(train_rids), batch_size):
            batch = train_rids[i : i + batch_size]
            training_ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
            logger.info(f"    Added {added}/{len(train_rids)} images")

    # Add test images to Testing dataset
    if test_rids:
        testing_ds = ml.lookup_dataset(datasets["testing"])
        logger.info("  Adding images to Testing dataset...")
        added = 0
        for i in range(0, len(test_rids), batch_size):
            batch = test_rids[i : i + batch_size]
            testing_ds.add_dataset_members({"Image": batch}, validate=False)
            added += len(batch)
            logger.info(f"    Added {added}/{len(test_rids)} images")

    # Add randomly selected images to Small_Training and Small_Testing datasets
    small_train_size = 500
    small_test_size = 500

    if train_rids and len(train_rids) >= small_train_size:
        small_train_rids = random.sample(train_rids, small_train_size)
        small_training_ds = ml.lookup_dataset(datasets["small_training"])
        logger.info(f"  Adding {small_train_size} randomly selected images to Small_Training dataset...")
        small_training_ds.add_dataset_members({"Image": small_train_rids}, validate=False)
        logger.info(f"    Added {len(small_train_rids)} images")
    elif train_rids:
        # If we have fewer than 500 training images, use all of them
        small_training_ds = ml.lookup_dataset(datasets["small_training"])
        logger.info(f"  Adding {len(train_rids)} images to Small_Training dataset (all available)...")
        small_training_ds.add_dataset_members({"Image": train_rids}, validate=False)

    if test_rids and len(test_rids) >= small_test_size:
        small_test_rids = random.sample(test_rids, small_test_size)
        small_testing_ds = ml.lookup_dataset(datasets["small_testing"])
        logger.info(f"  Adding {small_test_size} randomly selected images to Small_Testing dataset...")
        small_testing_ds.add_dataset_members({"Image": small_test_rids}, validate=False)
        logger.info(f"    Added {len(small_test_rids)} images")
    elif test_rids:
        # If we have fewer than 500 test images, use all of them
        small_testing_ds = ml.lookup_dataset(datasets["small_testing"])
        logger.info(f"  Adding {len(test_rids)} images to Small_Testing dataset (all available)...")
        small_testing_ds.add_dataset_members({"Image": test_rids}, validate=False)

    # Create labeled split datasets from training images only (all have ground truth)
    # This enables ROC analysis since both train and test partitions have labels
    if train_rids:
        # Shuffle training RIDs for random split
        shuffled_train_rids = train_rids.copy()
        random.shuffle(shuffled_train_rids)

        # Full labeled split: 80/20 split of all training images
        split_point = int(len(shuffled_train_rids) * 0.8)
        labeled_train_rids = shuffled_train_rids[:split_point]
        labeled_test_rids = shuffled_train_rids[split_point:]

        labeled_training_ds = ml.lookup_dataset(datasets["labeled_training"])
        logger.info(f"  Adding {len(labeled_train_rids)} images to Labeled_Training dataset...")
        for i in range(0, len(labeled_train_rids), batch_size):
            batch = labeled_train_rids[i : i + batch_size]
            labeled_training_ds.add_dataset_members({"Image": batch}, validate=False)
        logger.info(f"    Added {len(labeled_train_rids)} images")

        labeled_testing_ds = ml.lookup_dataset(datasets["labeled_testing"])
        logger.info(f"  Adding {len(labeled_test_rids)} images to Labeled_Testing dataset...")
        labeled_testing_ds.add_dataset_members({"Image": labeled_test_rids}, validate=False)
        logger.info(f"    Added {len(labeled_test_rids)} images")

        # Small labeled split: 500 images (400 train / 100 test) from training images
        small_labeled_total = 500
        small_labeled_train_size = 400
        small_labeled_test_size = 100

        if len(train_rids) >= small_labeled_total:
            small_labeled_sample = random.sample(train_rids, small_labeled_total)
            small_labeled_train_rids = small_labeled_sample[:small_labeled_train_size]
            small_labeled_test_rids = small_labeled_sample[small_labeled_train_size:]

            small_labeled_training_ds = ml.lookup_dataset(datasets["small_labeled_training"])
            logger.info(f"  Adding {len(small_labeled_train_rids)} images to Small_Labeled_Training dataset...")
            small_labeled_training_ds.add_dataset_members({"Image": small_labeled_train_rids}, validate=False)
            logger.info(f"    Added {len(small_labeled_train_rids)} images")

            small_labeled_testing_ds = ml.lookup_dataset(datasets["small_labeled_testing"])
            logger.info(f"  Adding {len(small_labeled_test_rids)} images to Small_Labeled_Testing dataset...")
            small_labeled_testing_ds.add_dataset_members({"Image": small_labeled_test_rids}, validate=False)
            logger.info(f"    Added {len(small_labeled_test_rids)} images")
        else:
            # If fewer than 500 training images, use 80/20 split of what we have
            small_split_point = int(len(train_rids) * 0.8)
            small_labeled_train_rids = train_rids[:small_split_point]
            small_labeled_test_rids = train_rids[small_split_point:]

            small_labeled_training_ds = ml.lookup_dataset(datasets["small_labeled_training"])
            logger.info(f"  Adding {len(small_labeled_train_rids)} images to Small_Labeled_Training dataset...")
            small_labeled_training_ds.add_dataset_members({"Image": small_labeled_train_rids}, validate=False)

            small_labeled_testing_ds = ml.lookup_dataset(datasets["small_labeled_testing"])
            logger.info(f"  Adding {len(small_labeled_test_rids)} images to Small_Labeled_Testing dataset...")
            small_labeled_testing_ds.add_dataset_members({"Image": small_labeled_test_rids}, validate=False)

    return datasets, {
        "total_images": len(all_rids),
        "training_images": len(train_rids),
        "testing_images": len(test_rids),
        "uploaded_assets": len(assets),
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main(args: argparse.Namespace | None = None) -> int:
    """Main entry point for the CIFAR-10 loader.

    Orchestrates the complete loading process: catalog connection/creation,
    schema setup, data download, and image loading.

    Args:
        args: Parsed command-line arguments. If None, arguments are parsed
            from sys.argv.

    Returns:
        Exit code: 0 for success, 1 for failure.

    Example:
        Command-line usage::

            load-cifar10 --hostname localhost --create-catalog demo --num-images 100

        Programmatic usage::cre

            >>> args = argparse.Namespace(
            ...     hostname='localhost',
            ...     create_catalog='demo',
            ...     catalog_id=None,
            ...     domain_schema=None,
            ...     num_images=100,
            ...     batch_size=500,
            ...     dry_run=False,
            ...     show_urls=False
            ... )
            >>> exit_code = main(args)
    """
    if args is None:
        args = parse_args()
        # Verify Kaggle credentials (only if not dry run)
        if not args.dry_run and not verify_kaggle_credentials():
            return 1

    # Either create a new catalog or connect to existing one
    if args.create_catalog:
        logger.info(
            f"Creating new catalog on {args.hostname} with project name: {args.create_catalog}"
        )

        # Create the catalog
        catalog = create_ml_catalog(args.hostname, args.create_catalog)
        model = catalog.getCatalogModel()

        # Create domain schema
        model.create_schema(Schema.define(args.create_catalog))

        catalog_id = catalog.catalog_id
        domain_schema = args.create_catalog

        print(f"\n{'='*60}")
        print("  CREATED NEW CATALOG")
        print(f"  Hostname:    {args.hostname}")
        print(f"  Catalog ID:  {catalog_id}")
        print(f"  Schema:      {domain_schema}")
        print(f"{'='*60}\n")

        # Connect to the newly created catalog
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(catalog_id),
            domain_schema=domain_schema,
            check_auth=True,
        )
    else:
        logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(args.catalog_id),
            domain_schema=args.domain_schema,
            check_auth=True,
        )
        catalog_id = args.catalog_id
        domain_schema = ml.domain_schema
        logger.info(f"Connected to catalog, domain schema: {domain_schema}")

    # Set up domain model
    logger.info("Setting up domain model...")
    setup_domain_model(ml)
    logger.info("Domain model setup complete")

    # Configure table display annotations
    setup_table_annotations(ml)

    # Apply catalog annotations for Chaise web interface
    logger.info("Applying catalog annotations...")
    project_name = args.create_catalog if args.create_catalog else domain_schema
    ml.apply_catalog_annotations(
        navbar_brand_text=f"CIFAR-10 ({project_name})",
        head_title="CIFAR-10 ML Catalog",
    )

    # Setup dataset types
    setup_dataset_types(ml)

    # Load images or dry run
    datasets = None
    load_result = None
    if not args.dry_run:
        # Download CIFAR-10 from Kaggle
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = download_cifar10(temp_path)
            logger.info(f"Downloaded CIFAR-10 to: {data_dir}")

            # Load images (also creates datasets within execution for provenance)
            datasets, load_result = load_images(
                ml, data_dir, args.batch_size, max_images=args.num_images
            )
            logger.info(f"Loading complete: {load_result}")
    else:
        # In dry run mode, create datasets without images or features
        logger.info("Dry run mode - creating empty datasets without image upload")
        datasets = create_dataset_hierarchy(ml)

    # Get Chaise URLs for datasets if requested
    dataset_urls = {}
    if args.show_urls:
        logger.info("Fetching Chaise URLs for datasets...")
        for name, rid in datasets.items():
            try:
                url = ml.chaise_url(rid)
                dataset_urls[name] = url
                logger.info(f"  {name}: {url}")
            except Exception as e:
                logger.warning(f"  Failed to get URL for {name}: {e}")
                dataset_urls[name] = ""

    # Print summary
    print("\n" + "=" * 60)
    print("  CIFAR-10 LOADING COMPLETE")
    print("=" * 60)
    print(f"  Hostname:      {args.hostname}")
    print(f"  Catalog ID:    {catalog_id}")
    print(f"  Schema:        {domain_schema}")
    print("")
    print("  Datasets created:")
    # Define all datasets to display
    dataset_display = [
        ("Complete (Labeled)", "complete"),
        ("Split", "split"),
        ("Training (Labeled)", "training"),
        ("Testing (Unlabeled)", "testing"),
        ("Small_Split", "small_split"),
        ("Small_Training (Labeled)", "small_training"),
        ("Small_Testing (Unlabeled)", "small_testing"),
        ("Labeled_Split", "labeled_split"),
        ("Labeled_Training", "labeled_training"),
        ("Labeled_Testing", "labeled_testing"),
        ("Small_Labeled_Split", "small_labeled_split"),
        ("Small_Labeled_Training", "small_labeled_training"),
        ("Small_Labeled_Testing", "small_labeled_testing"),
    ]

    for display_name, key in dataset_display:
        if key in datasets:
            rid = datasets[key]
            if args.show_urls and dataset_urls:
                print(f"    - {display_name}: {rid}")
                print(f"      URL: {dataset_urls.get(key, 'N/A')}")
            else:
                print(f"    - {display_name}: {rid}")
    if load_result:
        print("")
        print(f"  Images loaded: {load_result['total_images']}")
        print(f"    - Training: {load_result['training_images']}")
        print(f"    - Testing:  {load_result['testing_images']}")
    if not args.show_urls:
        print("")
        print("  Tip: Use --show-urls to display Chaise URLs for each dataset")
    print("=" * 60 + "\n")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with attributes:
        - hostname: Deriva server hostname
        - catalog_id: Existing catalog ID (or None)
        - create_catalog: New catalog name (or None)
        - domain_schema: Domain schema name (or None for auto-detect)
        - batch_size: Images per batch
        - dry_run: Skip image download/upload
        - num_images: Max images to upload (or None for all)
        - show_urls: Display Chaise URLs
    """
    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 dataset into a DerivaML catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a new catalog and load CIFAR-10
    load-cifar10 --hostname localhost --create-catalog cifar10_demo

    # Load into an existing catalog
    load-cifar10 --hostname ml.derivacloud.org --catalog-id 99

    # Dry run (create schema/datasets only, no image download)
    load-cifar10 --hostname localhost --create-catalog test --dry-run

    # Load only 100 images (50 training + 50 testing) for testing
    load-cifar10 --hostname localhost --create-catalog test --num-images 100

    # Show Chaise URLs for created datasets
    load-cifar10 --hostname localhost --create-catalog demo --show-urls

For more information, see:
    https://github.com/informatics-isi-edu/deriva-ml
        """,
    )

    parser.add_argument(
        "--hostname",
        required=True,
        help="Deriva server hostname (e.g., localhost, ml.derivacloud.org)",
    )

    catalog_group = parser.add_mutually_exclusive_group(required=True)
    catalog_group.add_argument(
        "--catalog-id",
        help="Catalog ID to connect to (for existing catalogs)",
    )
    catalog_group.add_argument(
        "--create-catalog",
        metavar="PROJECT_NAME",
        help="Create a new catalog with this project/schema name",
    )

    parser.add_argument(
        "--domain-schema",
        help="Domain schema name (auto-detected if not provided)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of images to process per batch (default: 500)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up schema and datasets without downloading/uploading images",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of images to upload (default: all ~60,000)",
    )

    parser.add_argument(
        "--show-urls",
        action="store_true",
        help="Show Chaise web interface URLs for datasets in the summary",
    )

    return parser.parse_args()


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    # Verify Kaggle credentials (skip for dry run)
    if not args.dry_run and not verify_kaggle_credentials():
        sys.exit(1)

    sys.exit(main(args))
