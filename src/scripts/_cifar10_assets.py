"""CIFAR-10 Stage 2: upload images, add classification features.

This module is the assets layer. It downloads + extracts the
CIFAR-10 archive (via ``_cifar10_source``), uploads each image
as an ``Image`` asset inside one Execution, then adds an
``Image_Classification`` feature row for each image inside a
separate Execution.

Stage 2 is fully self-contained: it does not depend on
in-memory state from any earlier step. The feature-labeling
sub-stage recovers each image's class from its filename
(format: ``train_<class>_<id>.png`` or ``test_<class>_<id>.png``)
by reading the catalog, so it can be re-run against any
catalog where stage 1 is complete and some images exist.

Public API:
    - ``upload_images(ml, archive_path=None, max_images=None,
        batch_size=500)`` — Stage 2a.
    - ``add_classification_features(ml)`` — Stage 2b. Reads
      back uploaded Image rows from the catalog.
    - ``class_from_filename(filename)`` — pure helper that
      decodes the class from a CIFAR-10 image filename.
    - ``run_assets_phase(ml, max_images=None, batch_size=500)``
      — orchestrator that runs 2a then 2b.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deriva_ml import DerivaML
from deriva_ml.core.ermrest import UploadProgress
from deriva_ml.execution import ExecutionConfiguration

from scripts._cifar10_source import download_cifar10_archive, extract_cifar10_to_png

logger = logging.getLogger(__name__)

CIFAR10_CLASSES_FROZEN = frozenset(
    {
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    }
)


def class_from_filename(filename: str) -> str | None:
    """Decode the CIFAR-10 class from an image filename.

    Image filenames produced by Stage 2a have the shape
    ``train_<class>_<id>.png`` or ``test_<class>_<id>.png``,
    where ``<class>`` is one of the ten CIFAR-10 class names.
    This helper extracts the class name; returns ``None`` if
    the filename doesn't follow the expected pattern or the
    decoded class isn't a known CIFAR-10 class.

    Args:
        filename: Image filename (with or without leading path).

    Returns:
        The class name if the filename decodes cleanly,
        otherwise ``None``.

    Example:
        >>> class_from_filename("train_frog_42.png")
        'frog'
        >>> class_from_filename("test_cat_19.png")
        'cat'
        >>> class_from_filename("random.png") is None
        True
    """
    stem = Path(filename).name
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    if parts[0] not in ("train", "test"):
        return None
    candidate = parts[1]
    if candidate not in CIFAR10_CLASSES_FROZEN:
        return None
    return candidate


def _create_upload_progress_callback(
    total_files: int,
) -> tuple[Callable[[UploadProgress], None], dict[str, Any]]:
    """Create a progress callback for upload monitoring.

    Lifted from the previous load_cifar10.py with no behavior change.

    Args:
        total_files: Total number of files to be uploaded.

    Returns:
        A tuple of (callback_function, state_dict). The callback
        logs upload progress at configurable percentage intervals.

    Example:
        >>> callback, state = _create_upload_progress_callback(1000)
        >>> callable(callback)
        True
    """
    state = {"last_reported_percent": -1, "started": False, "callback_count": 0}

    if total_files < 20:
        report_every_percent = max(1, 100 // total_files) if total_files > 0 else 10
    elif total_files <= 100:
        report_every_percent = 10
    else:
        report_every_percent = 5

    def progress_callback(progress: UploadProgress) -> None:
        state["callback_count"] += 1
        if not state["started"]:
            state["started"] = True
            logger.info(
                f"  [Upload] Starting upload (reporting every ~{report_every_percent}%)..."
            )
        match = re.search(r"Uploading file (\d+) of (\d+)", progress.message)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            percent = progress.percent_complete
            report_percent = int(percent // report_every_percent) * report_every_percent
            if report_percent > state["last_reported_percent"]:
                state["last_reported_percent"] = report_percent
                logger.info(f"  [Upload] {percent:.0f}% ({current}/{total} files)")

    return progress_callback, state


def upload_images(
    ml: DerivaML,
    archive_path: Path | None = None,
    max_images: int | None = None,
    batch_size: int = 500,  # currently unused but reserved for future batching
) -> dict[str, Any]:
    """Stage 2a — upload CIFAR-10 images as Image assets.

    Downloads the CIFAR-10 archive (cached after first call),
    extracts to a temporary directory, and registers + uploads
    every PNG as an ``Image`` asset inside one Execution. Train
    images get filenames ``train_<class>_<id>.png``; test images
    get ``test_<class>_<id>.png``. The class is encoded in the
    filename so Stage 2b can recover it.

    Args:
        ml: Connected DerivaML instance with the schema set up.
        archive_path: Optional path to a pre-downloaded archive.
            If ``None``, ``download_cifar10_archive()`` is called.
        max_images: Optional total cap (split evenly between train
            and test). ``None`` means upload everything (~60K).
        batch_size: Reserved for future use; currently unused.

    Returns:
        Stats dict with keys ``total_images``, ``training_images``,
        ``testing_images``, ``execution_rid``.

    Example:
        >>> ml = DerivaML(hostname="localhost", catalog_id="42")
        >>> stats = upload_images(ml, max_images=100)
        >>> stats["total_images"]
        100
    """
    if archive_path is None:
        archive_path = download_cifar10_archive()

    workflow = ml.create_workflow(
        name="CIFAR-10 Asset Upload",
        workflow_type="CIFAR_Data_Load",
        description="Upload CIFAR-10 images to the Image asset table",
    )
    config = ExecutionConfiguration(workflow=workflow)

    if max_images:
        train_limit = max_images // 2
        test_limit = max_images - train_limit
        logger.info(f"Loading {train_limit} train + {test_limit} test images")
    else:
        train_limit = None
        test_limit = None

    train_count = 0
    test_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        train_dir, test_dir, labels = extract_cifar10_to_png(archive_path, temp_path)
        logger.info(f"Extracted CIFAR-10 to: {temp_path}")

        with ml.create_execution(config) as exe:
            logger.info(f"  Upload execution RID: {exe.execution_rid}")
            execution_rid = exe.execution_rid

            # Clear working dir
            working_dir = exe.working_dir
            if working_dir.exists():
                for item in working_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            # Register train images
            for img_path in sorted(train_dir.glob("*.png")):
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name is None:
                    logger.warning(f"No label for {image_id}, skipping")
                    continue
                if train_limit and train_count >= train_limit:
                    break
                new_filename = f"train_{class_name}_{image_id}.png"
                exe.asset_file_path(
                    asset_name="Image",
                    file_name=str(img_path),
                    asset_types=["Image"],
                    copy_file=True,
                    rename_file=new_filename,
                )
                train_count += 1
                if train_count % 1000 == 0:
                    logger.info(f"  Registered {train_count} train images...")

            # Register test images
            for img_path in sorted(test_dir.glob("*.png")):
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name is None:
                    logger.warning(f"No label for {image_id}, skipping")
                    continue
                if test_limit and test_count >= test_limit:
                    break
                new_filename = f"test_{class_name}_{image_id}.png"
                exe.asset_file_path(
                    asset_name="Image",
                    file_name=str(img_path),
                    asset_types=["Image"],
                    copy_file=True,
                    rename_file=new_filename,
                )
                test_count += 1
                if test_count % 1000 == 0:
                    logger.info(f"  Registered {test_count} test images...")

            logger.info(
                f"  Total: {train_count} train + {test_count} test = {train_count + test_count}"
            )

        # Upload after context exits (matches previous behavior)
        total_count = train_count + test_count
        progress_callback, _ = _create_upload_progress_callback(total_count)
        upload_result = exe.upload_execution_outputs(
            clean_folder=True, progress_callback=progress_callback
        )

        logger.info("  [Upload] 100% complete")
        uploaded_count = sum(len(files) for files in upload_result.values())
        logger.info(f"  Upload complete: {uploaded_count} files uploaded")

    return {
        "total_images": train_count + test_count,
        "training_images": train_count,
        "testing_images": test_count,
        "execution_rid": execution_rid,
    }


def add_classification_features(ml: DerivaML) -> dict[str, Any]:
    """Stage 2b — add Image_Classification feature for every uploaded image.

    Queries the catalog for all ``Image`` asset rows, decodes the
    class from each filename via :func:`class_from_filename`, and
    adds an ``Image_Classification`` feature row inside one
    Execution. Images whose filenames don't decode are logged and
    skipped.

    This sub-stage is fully self-contained — it reads back from
    the catalog rather than depending on any in-memory state from
    Stage 2a. It can be re-run safely against a catalog where the
    schema is set up and some images have already been uploaded.

    Args:
        ml: Connected DerivaML instance.

    Returns:
        Stats dict with keys ``features_added``, ``images_skipped``,
        ``execution_rid``.

    Example:
        >>> stats = add_classification_features(ml)
        >>> stats["features_added"]
        100
    """
    assets = ml.list_assets("Image")
    logger.info(f"Found {len(assets)} Image assets in catalog")

    workflow = ml.create_workflow(
        name="CIFAR-10 Classification Labeling",
        workflow_type="CIFAR_Data_Load",
        description="Add Image_Classification feature for each Image asset",
    )
    config = ExecutionConfiguration(workflow=workflow)

    ImageClassification = ml.feature_record_class("Image", "Image_Classification")

    feature_records = []
    skipped = 0
    for asset in assets:
        class_name = class_from_filename(asset.filename)
        if class_name is None:
            logger.warning(f"Skipping {asset.filename}: cannot decode class")
            skipped += 1
            continue
        feature_records.append(
            ImageClassification(
                Image=asset.asset_rid,
                Image_Class=class_name,
            )
        )

    with ml.create_execution(config) as exe:
        logger.info(f"  Labeling execution RID: {exe.execution_rid}")
        execution_rid = exe.execution_rid
        logger.info(f"  Adding {len(feature_records)} classification labels...")
        exe.add_features(feature_records)

    exe.upload_execution_outputs(clean_folder=True)
    logger.info(f"  Added {len(feature_records)} Image_Classification features")

    return {
        "features_added": len(feature_records),
        "images_skipped": skipped,
        "execution_rid": execution_rid,
    }


def run_assets_phase(
    ml: DerivaML,
    archive_path: Path | None = None,
    max_images: int | None = None,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Stage 2 orchestrator — upload images then add features.

    Args:
        ml: Connected DerivaML instance.
        archive_path: Optional pre-downloaded archive.
        max_images: Optional total image cap.
        batch_size: Reserved.

    Returns:
        Merged stats dict from upload_images + add_classification_features.

    Example:
        >>> stats = run_assets_phase(ml, max_images=100)
        >>> stats["features_added"] == stats["total_images"]
        True
    """
    upload_stats = upload_images(
        ml, archive_path=archive_path, max_images=max_images, batch_size=batch_size
    )
    feature_stats = add_classification_features(ml)
    return {**upload_stats, **feature_stats}
