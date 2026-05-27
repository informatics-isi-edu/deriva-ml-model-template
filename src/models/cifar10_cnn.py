"""CIFAR-10 2-Layer CNN Model.

A small convolutional network for CIFAR-10 classification, used as the
canonical end-to-end example of integrating PyTorch with DerivaML.

Architecture:
- Conv2d(3, 32) -> ReLU -> MaxPool2d
- Conv2d(32, 64) -> ReLU -> MaxPool2d
- Linear(64*8*8, hidden_size) -> ReLU
- Linear(hidden_size, 10)

Expected accuracy: ~60-70% with default parameters.

Data loading uses DerivaML's framework adapter ``DatasetBag.as_torch_dataset``
for both training and testing — the adapter yields
``(sample, target, rid)`` triples, so per-image predictions can be
recorded back to the catalog as ``Image_Classification`` feature values
keyed by the element RID with no hand-rolled bag-iteration code.
"""

from __future__ import annotations

import csv
import random
import warnings
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from deriva_ml import DerivaML, MLAsset, ExecAssetType
from deriva_ml.dataset import DatasetBag
from deriva_ml.execution import Execution

from models.cifar10_classes import CIFAR10_CLASS_NAMES, CIFAR10_CLASS_TO_IDX


# CIFAR-10's 32x32 RGB images normalized to [-1, 1].
_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Role terms — read straight from the catalog's ``Dataset_Type`` vocabulary.
# The runner switches on these (case-insensitively) and routes to the
# matching lane below. Qualifier terms like ``Labeled`` / ``Complete`` are
# orthogonal and are silently ignored here; they don't pick a lane.
#
# To add a new role (e.g., ``Calibration``) the work is two-step: add the
# vocabulary term to ``Dataset_Type`` in the catalog AND register a handler
# in the appropriate dict below. The runner no longer carries a closed
# internal role table — the dispatch keys are the catalog terms themselves.
_ROLE_TRAINING = "training"
_ROLE_TESTING = "testing"
_ROLE_VALIDATION = "validation"
_ROLE_SPLIT = "split"

_KNOWN_ROLES: frozenset[str] = frozenset(
    {_ROLE_TRAINING, _ROLE_TESTING, _ROLE_VALIDATION, _ROLE_SPLIT}
)


def _bag_dataset_types(bag: DatasetBag) -> list[str]:
    """Return the literal ``Dataset_Type`` vocabulary terms on a bag.

    The runner used to translate these into a closed-list internal role
    (``"training"``/``"testing"``/``"unknown"``) and silently drop anything
    unknown. The catalog's vocabulary is the source of truth — this function
    just exposes it so the dispatch can switch on it directly.

    Args:
        bag: A ``DatasetBag`` returned by ``execution.datasets``.

    Returns:
        The bag's ``Dataset_Type`` terms verbatim (case preserved), e.g.
        ``["Training", "Labeled"]`` or ``["Validation", "Labeled"]``.
    """
    return list(bag.dataset_types)


def _classify_bag(bag: DatasetBag) -> tuple[set[str], list[str]]:
    """Split a bag's ``Dataset_Type`` terms into known roles vs other terms.

    A bag can carry multiple types (``["Training", "Labeled"]``). The role
    set is what the dispatch keys off of; the remaining terms are
    qualifiers (``Labeled``, ``Complete``) or unrecognized types and are
    returned separately so callers can warn / log without breaking the
    dispatch.

    Args:
        bag: A ``DatasetBag`` returned by ``execution.datasets``.

    Returns:
        ``(roles, other)`` where ``roles`` is the lower-cased set of role
        terms recognized by the runner (subset of ``_KNOWN_ROLES``) and
        ``other`` is the original-case list of terms that are not roles.
    """
    roles: set[str] = set()
    other: list[str] = []
    for term in _bag_dataset_types(bag):
        if term.lower() in _KNOWN_ROLES:
            roles.add(term.lower())
        else:
            other.append(term)
    return roles, other


def _flatten_bags(bags: list[DatasetBag]) -> list[DatasetBag]:
    """Flatten a list of bags by recursively expanding ``Split`` parents.

    The CIFAR-10 dataset configs ship as either a leaf ``Training``/
    ``Testing``/``Validation`` bag or a ``Split`` parent containing several
    as children. The dispatch cares only about the leaves, so descend into
    any ``Split`` we encounter. Cycles are not possible since the catalog's
    parent/child graph is a DAG.
    """
    flat: list[DatasetBag] = []
    for bag in bags:
        roles, _ = _classify_bag(bag)
        if _ROLE_SPLIT in roles:
            flat.extend(_flatten_bags(bag.list_dataset_children()))
        else:
            flat.append(bag)
    return flat


def _load_image(path: Any, _row: dict[str, Any]) -> PIL.Image.Image:
    """Sample loader for ``as_torch_dataset``: open an image file as RGB PIL."""
    return PIL.Image.open(path).convert("RGB")


def _target_to_class_idx(rec: Any) -> int:
    """Map an ``Image_Classification`` feature record to its class index.

    Used as the ``target_transform`` for ``bag.as_torch_dataset(...)``.
    ``rec`` is a typed ``FeatureRecord`` with an ``Image_Class`` term
    column (or its ``Name`` alias from the underlying vocab term). Missing
    or unrecognized classes raise — the adapter's ``missing="skip"`` arg
    is the right place to filter unlabeled elements before they reach
    here.
    """
    cls = getattr(rec, "Image_Class", None) or rec.Name
    return CIFAR10_CLASS_TO_IDX[cls]


def _rid_collate(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate ``(tensor, int, str)`` triples — strings can't be tensorized."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    rids = [b[2] for b in batch]
    return images, labels, rids


def record_test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    execution: Execution,
    ml_instance: DerivaML,
    device: torch.device,
    source_label: str = "unknown",
) -> int:
    """Record per-image classification predictions to the DerivaML catalog.

    Iterates ``test_loader`` (which yields ``(image, label, rid)`` triples
    from :meth:`DatasetBag.as_torch_dataset`), runs inference, and stages
    one ``Image_Classification`` feature record per image via
    ``execution.add_features(...)``. Also writes a CSV with the full
    probability distribution per image as an ``Execution_Asset`` for
    downstream ROC analysis.

    **Provenance contract (analyst/02 from 2026-05-27 e2e run).** The
    predictions emitted here come from whatever state the model is in
    *at this moment* — typically the final-epoch state in training mode,
    or the loaded-weights state in evaluation mode. To make that
    knowable downstream, this function:

    1. Tags every catalog feature row and CSV row with ``source_label``
       (e.g. ``"epoch_10"`` in training mode, ``"evaluation"`` in eval
       mode). The Analyst can read this column to know which model
       state produced each prediction.
    2. Recomputes accuracy from the exact same logits used to emit
       predictions (when the test loader yields ground-truth labels)
       and logs the result. This number is guaranteed to match what
       the committed CSV would reproduce when joined against ground
       truth, even if the surrounding training-loop log surfaced a
       different epoch-time number. Analysts who download the CSV
       and re-compute will see this number.

    The recomputed accuracy is also returned in the function's printed
    summary so the training log itself flags any divergence between
    "final-epoch test_acc inside the training loop" and "test_acc at
    the moment predictions were emitted."

    Args:
        model: Trained PyTorch model.
        test_loader: Yields ``(image_batch, label_batch, rid_list)``.
        class_names: List of class names in index order.
        execution: DerivaML execution context.
        ml_instance: DerivaML instance for catalog access.
        device: PyTorch device for inference.
        source_label: Provenance tag describing which model state
            these predictions reflect. Recommended values:
            ``"epoch_N"`` (training mode), ``"evaluation"`` (eval
            mode loading saved weights). Stored as a column on every
            CSV row. The catalog ``Image_Classification`` feature
            row does NOT carry this label (would require a schema
            migration); cross-channel consumers should rely on the
            CSV asset for source-label provenance.

    Returns:
        Number of predictions recorded.
    """
    model.eval()
    ImageClassification = ml_instance.feature_record_class(
        "Image", "Image_Classification"
    )

    feature_records: list[Any] = []
    csv_rows: list[dict[str, Any]] = []

    # Track ground-truth-aware accuracy from the same logits we emit.
    # If the test loader has GT labels (label != -1), this number is
    # the accuracy the Analyst will recompute when joining the
    # committed CSV against the ground-truth feature. By printing it
    # here we close the provenance loop: the training log shows the
    # epoch-time accuracy AND the prediction-emission accuracy, so
    # any divergence is visible without leaving the runner output.
    emit_correct = 0
    emit_total_labeled = 0

    with torch.no_grad():
        for inputs, labels, rids in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            probabilities = F.softmax(model(inputs), dim=1)
            confidences, predicted = probabilities.max(1)

            labeled_mask = labels != -1
            emit_total_labeled += int(labeled_mask.sum().item())
            emit_correct += int(
                (predicted.eq(labels) & labeled_mask).sum().item()
            )

            for i, rid in enumerate(rids):
                probs = probabilities[i].cpu().numpy()
                predicted_class = class_names[predicted[i].item()]
                confidence = confidences[i].item()

                feature_records.append(
                    ImageClassification(
                        Image=rid,
                        Image_Class=predicted_class,
                        Confidence=confidence,
                    )
                )

                row = {
                    "Image_RID": rid,
                    "Source_Label": source_label,
                    "Predicted_Class": predicted_class,
                    "Confidence": confidence,
                }
                for j, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = probs[j]
                csv_rows.append(row)

    if feature_records:
        execution.add_features(feature_records)
        msg = (
            f"  Recorded {len(feature_records)} classification predictions "
            f"with confidence scores (source_label={source_label!r})"
        )
        if emit_total_labeled > 0:
            emit_acc = 100.0 * emit_correct / emit_total_labeled
            msg += (
                f"\n    Emission-time accuracy: {emit_acc:.2f}% "
                f"({emit_correct}/{emit_total_labeled}). "
                f"This is what the Analyst will recompute from the "
                f"committed CSV. Compare against the {source_label} "
                f"line of the training log — any divergence means the "
                f"two accuracies were measured on different model state."
            )
        print(msg)
    else:
        print("  WARNING: No predictions recorded (test loader was empty)")

    if csv_rows:
        csv_file = execution.asset_file_path(
            MLAsset.execution_asset,
            "prediction_probabilities.csv",
            description=(
                "Per-image predicted class and probability distributions "
                "over all CIFAR-10 classes. Source_Label column records "
                "the model state these predictions reflect (e.g. "
                "epoch_N for training, evaluation for eval mode)."
            ),
        )
        fieldnames = [
            "Image_RID",
            "Source_Label",
            "Predicted_Class",
            "Confidence",
        ] + [f"prob_{c}" for c in class_names]
        with csv_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Saved probability distributions to: {csv_file}")

    return len(feature_records)


class SimpleCNN(nn.Module):
    """A simple 2-layer CNN for CIFAR-10 classification.

    Architecture:
        - Conv layer 1: 3 -> conv1_channels, 3x3 kernel, padding=1
        - MaxPool 2x2 (32x32 -> 16x16)
        - Conv layer 2: conv1_channels -> conv2_channels, 3x3 kernel, padding=1
        - MaxPool 2x2 (16x16 -> 8x8)
        - Fully connected: conv2_channels * 8 * 8 -> hidden_size
        - Output: hidden_size -> 10 classes

    Args:
        conv1_channels: Number of output channels for first conv layer.
        conv2_channels: Number of output channels for second conv layer.
        hidden_size: Size of the hidden fully connected layer.
        dropout_rate: Dropout probability for regularization.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_size: int = 128,
        dropout_rate: float = 0.0,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # After two 2x2 pooling operations: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(conv2_channels * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def _seed_everything(seed: int) -> None:
    """Seed every RNG that affects CIFAR-10 CNN training.

    Covers Python's ``random`` (used by ``DatasetBag`` adapters), NumPy
    (used by the per-image probability arrays in
    :func:`record_test_predictions` and any downstream sklearn step), and
    PyTorch's CPU + CUDA RNGs. Also calls
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` so any
    nondeterministic op (e.g. some CUDA kernels) issues a warning rather
    than silently violating reproducibility — ``warn_only`` keeps the run
    from hard-erroring on ops that don't have a deterministic
    implementation.

    The training DataLoader gets its own seeded ``torch.Generator`` in
    :func:`load_cifar10_from_execution`; this function only handles the
    global RNG state.

    Args:
        seed: Non-negative integer seed used everywhere.

    Example:
        >>> _seed_everything(42)
        >>> import random
        >>> a = random.random()
        >>> _seed_everything(42)
        >>> b = random.random()
        >>> a == b
        True
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only=True: don't fail on ops without a deterministic kernel
    # (e.g. some pooling/conv backward passes on CUDA), just surface a
    # warning. Full determinism on GPU additionally requires the user to
    # set CUBLAS_WORKSPACE_CONFIG — out of scope for this template.
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_cifar10_from_execution(
    execution: Execution,
    batch_size: int,
    require_training: bool = False,
    seed: int | None = None,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None, list[str]]:
    """Build PyTorch DataLoaders directly from execution dataset bags.

    Uses :meth:`DatasetBag.as_torch_dataset` for the training, testing, and
    validation lanes. The adapter is lazy, label-aware, requires no on-disk
    reorganization, and yields ``(sample, target, rid)`` triples so per-image
    predictions can be linked back to their catalog RIDs when recording
    feature values.

    Bags are dispatched by their catalog ``Dataset_Type`` terms via the
    per-role handlers registered below. Each handler matches on a single
    role term (``Training``/``Testing``/``Validation``) and ignores
    qualifier terms (``Labeled``, ``Complete``) — so e.g. a bag with type
    ``["Training", "Labeled"]`` still dispatches to the training handler.
    ``Split`` parents are flattened by :func:`_flatten_bags` before
    dispatch.

    Bags whose ``Dataset_Type`` contains no recognized role term emit a
    warning and are dropped. If ``require_training`` is set and no training
    bag was found across the entire input, the function raises — this
    closes the silent-failure mode where a Validation-only execution
    looked exactly like a successful training run (catalog-18 F40).

    Args:
        execution: DerivaML execution containing downloaded ``DatasetBag``s.
        batch_size: Batch size for all DataLoaders.
        require_training: If True, raise ``RuntimeError`` when no bag
            dispatches to the training lane. The training entry point
            ``cifar10_cnn`` sets this; ``test_only`` does not.
        seed: If provided, drive the training DataLoader's shuffle order
            from a dedicated ``torch.Generator`` seeded with this value.
            Test and validation loaders use ``shuffle=False`` and don't
            need a generator. ``None`` reproduces the pre-seed-knob
            behavior (PyTorch's default global generator).

    Returns:
        Tuple of ``(train_loader, test_loader, val_loader, class_names)``.
        Any loader may be ``None`` if no bag dispatched to that lane.
        ``class_names`` is the canonical CIFAR-10 list — it matches the
        model's output index ordering and is independent of which labels
        happen to appear in the bags.

    Raises:
        RuntimeError: If ``require_training=True`` and no training bag is
            present after flattening and dispatch.
    """
    train_loader: DataLoader | None = None
    test_loader: DataLoader | None = None
    val_loader: DataLoader | None = None

    def _build_dataset(bag: DatasetBag, missing: str):
        return bag.as_torch_dataset(
            element_type="Image",
            sample_loader=_load_image,
            transform=_TRANSFORM,
            targets=["Image_Classification"],
            target_transform=_target_to_class_idx,
            missing=missing,
        )

    def _handle_training(bag: DatasetBag) -> None:
        nonlocal train_loader
        train_dataset = _build_dataset(bag, missing="skip")
        # Per-loader generator so the shuffle order is reproducible from
        # the run's seed. Falls back to PyTorch's default global generator
        # when seed is None — this preserves the pre-seed-knob behavior
        # for any caller that hasn't opted in.
        loader_generator: torch.Generator | None = None
        if seed is not None:
            loader_generator = torch.Generator()
            loader_generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # macOS fork() + MPS/GPU threads can deadlock
            collate_fn=_rid_collate,
            generator=loader_generator,
        )
        print(f"  Training samples: {len(train_dataset)}")

    def _handle_testing(bag: DatasetBag) -> None:
        # Test bags may contain unlabeled images — keep them with
        # target=-1 so the test loop can still produce predictions
        # for them (only the RID and predicted class matter for
        # downstream feature recording on unlabeled data).
        nonlocal test_loader
        test_dataset = _build_dataset(bag, missing="unknown")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_rid_collate,
        )
        print(f"  Testing samples: {len(test_dataset)}")

    def _handle_validation(bag: DatasetBag) -> None:
        # Validation bags carry ground truth and are evaluated *per epoch*
        # during training to surface a generalization metric in the
        # training log. Unlabeled rows are skipped (the val metric is
        # meaningless without labels), unlike the test lane which keeps
        # unlabeled rows so it can still record predictions.
        nonlocal val_loader
        val_dataset = _build_dataset(bag, missing="skip")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_rid_collate,
        )
        print(f"  Validation samples: {len(val_dataset)}")

    # Dispatch table. Keys are catalog ``Dataset_Type`` terms (case-folded);
    # values are the per-role handlers. ``Split`` is consumed by
    # ``_flatten_bags`` before we get here. Adding a new role means: add the
    # term to the catalog vocabulary AND add an entry here.
    handlers: dict[str, Any] = {
        _ROLE_TRAINING: _handle_training,
        _ROLE_TESTING: _handle_testing,
        _ROLE_VALIDATION: _handle_validation,
    }

    for bag in _flatten_bags(list(execution.datasets)):
        roles, other = _classify_bag(bag)
        # Pick the highest-priority role this bag carries.
        # Priority: training > testing > validation. A bag tagged with
        # more than one role is rare but well-defined; the dispatch is
        # deterministic.
        dispatched = False
        for role in (_ROLE_TRAINING, _ROLE_TESTING, _ROLE_VALIDATION):
            if role in roles:
                handlers[role](bag)
                dispatched = True
                break
        if not dispatched:
            bag_rid = getattr(bag, "dataset_rid", "<unknown>")
            warnings.warn(
                f"Bag {bag_rid} has no recognized role term in its "
                f"Dataset_Type {other!r} — known roles are "
                f"{sorted(_KNOWN_ROLES - {_ROLE_SPLIT})!r}. Skipping.",
                RuntimeWarning,
                stacklevel=2,
            )

    if require_training and train_loader is None:
        # Safety rail (closes catalog-18 F40-style silent failure).
        # The runner must not silently succeed when its primary input is
        # missing. Surface a structured error so the execution fails loudly
        # — Status=Failed, not Status=Uploaded.
        diagnostics = [
            f"  - {getattr(b, 'dataset_rid', '<unknown>')}: "
            f"Dataset_Type={_bag_dataset_types(b)!r}"
            for b in _flatten_bags(list(execution.datasets))
        ]
        diag_block = "\n".join(diagnostics) if diagnostics else "  (no input bags)"
        raise RuntimeError(
            "No bag with Dataset_Type=Training found in execution input. "
            "Cannot train. Input bags after flattening Split parents:\n"
            f"{diag_block}\n"
            "Add a Training-typed dataset to the execution config (see "
            "src/configs/datasets.py)."
        )

    return train_loader, test_loader, val_loader, list(CIFAR10_CLASS_NAMES)


def cifar10_cnn(
    # Model architecture parameters
    conv1_channels: int = 32,
    conv2_channels: int = 64,
    hidden_size: int = 128,
    dropout_rate: float = 0.0,
    # Training parameters
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    seed: int = 42,
    # Test-only mode
    test_only: bool = False,
    weights_filename: str = "cifar10_cnn_weights.pt",
    # DerivaML integration
    ml_instance: DerivaML | None = None,
    execution: Execution | None = None,
) -> None:
    """Train or evaluate a simple 2-layer CNN on CIFAR-10 data.

    This function integrates with DerivaML to:
    - Load data from execution datasets using restructure_assets()
    - Track training progress
    - Save model weights as execution assets
    - Record per-image predictions to the catalog

    The function expects datasets containing Image assets with Image_Classification
    feature values. Images are reorganized into a directory structure by dataset type
    (training/testing) and class label, then loaded using torchvision's ImageFolder.

    Test-only mode:
        When test_only=True, the model loads pre-trained weights from an execution
        asset and runs evaluation on the test set without training. Use this with
        the assets configuration to specify which weights to load.

    Args:
        conv1_channels: Output channels for first conv layer.
        conv2_channels: Output channels for second conv layer.
        hidden_size: Hidden layer size in fully connected layers.
        dropout_rate: Dropout probability (0.0 = no dropout).
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        weight_decay: L2 regularization weight decay.
        seed: RNG seed for byte-reproducible training. Drives weight
            initialization, training shuffle order, and any numpy/random
            sampling reached during training or prediction recording.
            Defaults to ``42`` (the same value used by
            ``cifar10_labeled_split`` in ``_cifar10_datasets.py``). Vary
            this knob — not the dataset partition seed — to estimate
            run-to-run variance across PyTorch random states.
        test_only: If True, skip training and only run evaluation on test data.
        weights_filename: Filename of weights asset to load in test_only mode.
        ml_instance: DerivaML instance for catalog access.
        execution: DerivaML execution context with datasets and assets.
    """
    if ml_instance is None or execution is None:
        raise ValueError("ml_instance and execution are required")

    mode = "Test-only" if test_only else "Training"
    print(f"CIFAR-10 CNN {mode}")
    print(f"  Host: {ml_instance.host_name}, Catalog: {ml_instance.catalog_id}")
    print(
        f"  Architecture: conv1={conv1_channels}, conv2={conv2_channels}, hidden={hidden_size}"
    )
    if not test_only:
        print(
            f"  Training: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}, seed={seed}"
        )

    # Seed RNGs BEFORE building the model so weight init is reproducible.
    # Order matters: SimpleCNN(...) below uses torch.nn's default init,
    # which pulls from the current global PyTorch RNG.
    _seed_everything(seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Create model
    model = SimpleCNN(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    ).to(device)

    # Load data directly from execution dataset bags (no restructuring needed)
    print("\nBuilding DataLoaders from execution datasets...")
    train_loader, test_loader, val_loader, class_names = load_cifar10_from_execution(
        execution, batch_size, require_training=not test_only, seed=seed
    )

    # Test-only mode: load weights and run evaluation
    if test_only:
        if test_loader is None:
            print("ERROR: No test data found in execution datasets.")
            print("  Test-only mode requires a dataset with type 'Testing'.")
            return

        print(f"  Test batches: {len(test_loader)}")

        # Find weights file in execution assets
        # asset_paths is a dict: {table_name: [AssetFilePath, ...]}
        weights_path = None
        all_assets = []
        for table_name, paths in execution.asset_paths.items():
            for asset_path in paths:
                all_assets.append(asset_path)
                if asset_path.name == weights_filename:
                    weights_path = asset_path
                    break
            if weights_path:
                break

        if weights_path is None:
            print(
                f"ERROR: Weights file '{weights_filename}' not found in execution assets."
            )
            print(
                "  Make sure to include the weights asset in your assets configuration."
            )
            print(f"  Available assets: {[p.name for p in all_assets]}")
            return

        print(f"\nLoading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # Load model config from checkpoint if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            print(f"  Checkpoint config: {config}")
            # Recreate model with saved config
            model = SimpleCNN(
                conv1_channels=config.get("conv1_channels", conv1_channels),
                conv2_channels=config.get("conv2_channels", conv2_channels),
                hidden_size=config.get("hidden_size", hidden_size),
                dropout_rate=config.get("dropout_rate", dropout_rate),
            ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Weights loaded successfully")

        # Run evaluation
        print("\nEvaluating on test set...")
        # Run evaluation. Test loader yields (image, label, rid) triples;
        # `label == -1` indicates an unlabeled test image (loss/accuracy
        # cannot be computed for those — only predictions are recorded).
        model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        test_correct = 0
        test_total_labeled = 0
        test_loss_sum = 0.0
        test_loss_batches = 0

        with torch.no_grad():
            for inputs, labels, _rids in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if torch.isfinite(loss):  # skip all-unlabeled batches
                    test_loss_sum += loss.item()
                    test_loss_batches += 1
                _, predicted = outputs.max(1)
                labeled_mask = labels != -1
                test_total_labeled += int(labeled_mask.sum().item())
                test_correct += int((predicted.eq(labels) & labeled_mask).sum().item())

        if test_total_labeled > 0:
            test_acc = 100.0 * test_correct / test_total_labeled
            test_loss = test_loss_sum / max(test_loss_batches, 1)
            print(f"  Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")
        else:
            test_acc = float("nan")
            test_loss = float("nan")
            print("  Test set is unlabeled — skipping accuracy/loss reporting.")

        # Record predictions to catalog. source_label="evaluation"
        # documents that these predictions reflect the model state
        # loaded from the weights file, not a freshly-trained epoch.
        print("\nRecording test predictions to catalog...")
        record_test_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
            source_label="evaluation",
        )

        # Save evaluation results
        results_file = execution.asset_file_path(
            MLAsset.execution_asset,
            "evaluation_results.txt",
            description="Test set evaluation summary: loss, accuracy, and configuration",
        )
        with results_file.open("w") as f:
            f.write("CIFAR-10 CNN Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Weights file: {weights_filename}\n")
            f.write(f"Labeled test samples: {test_total_labeled}\n")
            f.write(f"Test loss: {test_loss:.4f}\n")
            f.write(f"Test accuracy: {test_acc:.2f}%\n")
        print(f"  Saved results to: {results_file}")

        print("\nEvaluation complete!")
        return

    # Training mode: train_loader is guaranteed non-None by the safety rail
    # in load_cifar10_from_execution (require_training=True). No silent
    # fallback — a missing training bag raises RuntimeError there, which
    # bubbles up and fails the execution loudly (Status=Failed, not the
    # silent Status=Uploaded that catalog-18 F40 exhibited).
    print(f"  Training batches: {len(train_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")
    if val_loader:
        print(f"  Validation batches: {len(val_loader)}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Training loop
    print("\nTraining...")
    training_log = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels, _rids) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
        }

        # Evaluate on test set if available. Test loader yields
        # (image, label, rid) triples; label == -1 means unlabeled.
        if test_loader:
            model.eval()
            test_eval_criterion = nn.CrossEntropyLoss(ignore_index=-1)
            test_correct = 0
            test_total_labeled = 0
            test_loss_sum = 0.0
            test_loss_batches = 0

            with torch.no_grad():
                for inputs, labels, _rids in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = test_eval_criterion(outputs, labels)
                    if torch.isfinite(loss):
                        test_loss_sum += loss.item()
                        test_loss_batches += 1
                    _, predicted = outputs.max(1)
                    labeled_mask = labels != -1
                    test_total_labeled += int(labeled_mask.sum().item())
                    test_correct += int(
                        (predicted.eq(labels) & labeled_mask).sum().item()
                    )

            if test_total_labeled > 0:
                test_acc = 100.0 * test_correct / test_total_labeled
                test_loss = test_loss_sum / max(test_loss_batches, 1)
                log_entry["test_loss"] = test_loss
                log_entry["test_acc"] = test_acc
                print(
                    f"  Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%, "
                    f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%"
                )
            else:
                print(
                    f"  Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}% "
                    f"(test set unlabeled — no test metrics)"
                )
        else:
            print(
                f"  Epoch {epoch + 1}/{epochs}: "
                f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%"
            )

        # Validation lane: evaluate the held-out validation bag (if any)
        # at the end of every epoch and surface val_loss/val_acc as a
        # per-epoch metric in the training log. Validation bags carry
        # ground truth (we built the loader with missing="skip"), so
        # unlabeled rows shouldn't reach this loop — but use
        # ignore_index=-1 defensively to mirror the test lane.
        # Early-stopping policy on this metric is intentionally out of
        # scope for this PR; just surface the signal.
        if val_loader is not None:
            model.eval()
            val_criterion = nn.CrossEntropyLoss(ignore_index=-1)
            val_correct = 0
            val_total_labeled = 0
            val_loss_sum = 0.0
            val_loss_batches = 0

            with torch.no_grad():
                for inputs, labels, _rids in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = val_criterion(outputs, labels)
                    if torch.isfinite(loss):
                        val_loss_sum += loss.item()
                        val_loss_batches += 1
                    _, predicted = outputs.max(1)
                    labeled_mask = labels != -1
                    val_total_labeled += int(labeled_mask.sum().item())
                    val_correct += int(
                        (predicted.eq(labels) & labeled_mask).sum().item()
                    )

            if val_total_labeled > 0:
                val_acc = 100.0 * val_correct / val_total_labeled
                val_loss = val_loss_sum / max(val_loss_batches, 1)
                log_entry["val_loss"] = val_loss
                log_entry["val_acc"] = val_acc
                print(
                    f"  Epoch {epoch + 1}/{epochs} validation: "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
                )

        training_log.append(log_entry)

    # Save model weights
    print("\nSaving model...")
    weights_file = execution.asset_file_path(
        MLAsset.execution_asset,
        "cifar10_cnn_weights.pt",
        ExecAssetType.model_file,
        description="Trained CNN model weights, optimizer state, and training log",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "conv1_channels": conv1_channels,
                "conv2_channels": conv2_channels,
                "hidden_size": hidden_size,
                "dropout_rate": dropout_rate,
                "seed": seed,
            },
            "training_log": training_log,
        },
        weights_file,
    )
    print(f"  Saved weights to: {weights_file}")

    # Save training log as text
    log_file = execution.asset_file_path(
        MLAsset.execution_asset,
        "training_log.txt",
        description="Per-epoch training log: loss, accuracy, and architecture details",
    )
    with log_file.open("w") as f:
        f.write("CIFAR-10 CNN Training Log\n")
        f.write("=" * 50 + "\n\n")
        f.write("Architecture:\n")
        f.write(f"  conv1_channels: {conv1_channels}\n")
        f.write(f"  conv2_channels: {conv2_channels}\n")
        f.write(f"  hidden_size: {hidden_size}\n")
        f.write(f"  dropout_rate: {dropout_rate}\n\n")
        f.write("Training Parameters:\n")
        f.write(f"  learning_rate: {learning_rate}\n")
        f.write(f"  epochs: {epochs}\n")
        f.write(f"  batch_size: {batch_size}\n")
        f.write(f"  weight_decay: {weight_decay}\n")
        f.write(f"  seed: {seed}\n\n")
        f.write("Training Progress:\n")
        for entry in training_log:
            line = f"  Epoch {entry['epoch']}: train_loss={entry['train_loss']:.4f}, train_acc={entry['train_acc']:.2f}%"
            if "test_acc" in entry:
                line += f", test_acc={entry['test_acc']:.2f}%"
            if "val_acc" in entry:
                line += f", val_acc={entry['val_acc']:.2f}%"
            f.write(line + "\n")
    print(f"  Saved log to: {log_file}")

    # Record test predictions to catalog if test data is available.
    # source_label="epoch_N" tags every prediction with the epoch
    # whose weights produced it (the final epoch, since the runner
    # does not implement save-best). Analysts reading the committed
    # CSV can correlate against the matching line in training_log.txt.
    if test_loader is not None:
        print("\nRecording test predictions to catalog...")
        record_test_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
            source_label=f"epoch_{epochs}",
        )

    print("\nTraining complete!")
