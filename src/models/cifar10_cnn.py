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
for training (lazy, label-aware, no on-disk reorganization) plus a thin
RID-aware wrapper for the test loop so per-image predictions can be recorded
back to the catalog as ``Image_Classification`` feature values.
"""
from __future__ import annotations

import csv
from typing import Any

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from deriva_ml import DerivaML, MLAsset, ExecAssetType
from deriva_ml.dataset import DatasetBag
from deriva_ml.execution import Execution

from models.cifar10_classes import CIFAR10_CLASS_NAMES, CIFAR10_CLASS_TO_IDX


# CIFAR-10's 32x32 RGB images normalized to [-1, 1].
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _bag_role(bag: DatasetBag) -> str:
    """Return ``"training"``, ``"testing"``, ``"split"``, or ``"unknown"``.

    Reads ``bag.dataset_types`` (e.g., ``["Training", "Labeled"]``) and picks
    the role that matters for loader routing. ``Training``/``Testing`` go
    straight to a loader; ``Split`` means "descend into children".
    """
    types_lower = [t.lower() for t in bag.dataset_types]
    if "training" in types_lower:
        return "training"
    if "testing" in types_lower:
        return "testing"
    if "split" in types_lower:
        return "split"
    return "unknown"


def _flatten_bags(bags: list[DatasetBag]) -> list[DatasetBag]:
    """Flatten a list of bags by recursively expanding ``Split`` parents.

    The CIFAR-10 dataset configs ship as either a leaf ``Training``/
    ``Testing`` bag or a ``Split`` parent containing both as children. The
    train/test loader cares only about the leaves, so descend into any
    ``Split`` we encounter. Cycles are not possible since the catalog's
    parent/child graph is a DAG.
    """
    flat: list[DatasetBag] = []
    for bag in bags:
        if _bag_role(bag) == "split":
            flat.extend(_flatten_bags(bag.list_dataset_children()))
        else:
            flat.append(bag)
    return flat


def _load_image(path: Any, _row: dict[str, Any]) -> PIL.Image.Image:
    """Sample loader for ``as_torch_dataset``: open an image file as RGB PIL."""
    return PIL.Image.open(path).convert("RGB")


class _RidAwareImageDataset(Dataset):
    """A test-time wrapper that yields ``(image_tensor, label_idx, rid)``.

    ``DatasetBag.as_torch_dataset`` is the right tool for training (lazy,
    label-aware, no on-disk reorganization), but its ``__getitem__`` returns
    just ``(sample, target)`` — recording per-image predictions back to the
    catalog requires the element RID alongside each sample.

    This wrapper iterates the bag's ``Image`` rows directly, decodes via the
    same PIL+transform pipeline as training, and resolves each row's
    classification feature value (or returns ``-1`` if the test bag is
    unlabeled, in which case the label is unused — only the RID and predicted
    class matter for downstream feature recording).

    Args:
        bag: The downloaded dataset bag containing ``Image`` rows.
    """

    def __init__(self, bag: DatasetBag) -> None:
        members = bag.list_dataset_members(recurse=True).get("Image", [])
        # Pre-resolve label per-RID via feature_values; missing → -1 sentinel.
        # `feature_values` returns FeatureRecords with a target FK column.
        rid_to_label: dict[str, int] = {}
        for rec in bag.feature_values("Image", "Image_Classification"):
            cls = getattr(rec, "Image_Class", None) or getattr(rec, "Name", None)
            if cls in CIFAR10_CLASS_TO_IDX:
                # `rec.Image` is the FK pointing back at the asset RID.
                rid_to_label[rec.Image] = CIFAR10_CLASS_TO_IDX[cls]

        self._rids: list[str] = []
        self._labels: list[int] = []
        self._paths: list[Any] = []
        for row in members:
            rid = row["RID"]
            self._rids.append(rid)
            self._labels.append(rid_to_label.get(rid, -1))
            # Bag asset path: <bag_path>/data/assets/Image/<rid>/<filename>
            self._paths.append(bag.path / "data" / "assets" / "Image" / rid / row["Filename"])

    def __len__(self) -> int:
        return len(self._rids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img = PIL.Image.open(self._paths[idx]).convert("RGB")
        return _TRANSFORM(img), self._labels[idx], self._rids[idx]


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
) -> int:
    """Record per-image classification predictions to the DerivaML catalog.

    Iterates ``test_loader`` (which yields ``(image, label, rid)`` triples
    from :class:`_RidAwareImageDataset`), runs inference, and stages one
    ``Image_Classification`` feature record per image via
    ``execution.add_features(...)``. Also writes a CSV with the full
    probability distribution per image as an ``Execution_Asset`` for
    downstream ROC analysis.

    Args:
        model: Trained PyTorch model.
        test_loader: Yields ``(image_batch, label_batch, rid_list)``.
        class_names: List of class names in index order.
        execution: DerivaML execution context.
        ml_instance: DerivaML instance for catalog access.
        device: PyTorch device for inference.

    Returns:
        Number of predictions recorded.
    """
    model.eval()
    ImageClassification = ml_instance.feature_record_class("Image", "Image_Classification")

    feature_records: list[Any] = []
    csv_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for inputs, _labels, rids in test_loader:
            inputs = inputs.to(device)
            probabilities = F.softmax(model(inputs), dim=1)
            confidences, predicted = probabilities.max(1)

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
                    "Predicted_Class": predicted_class,
                    "Confidence": confidence,
                }
                for j, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = probs[j]
                csv_rows.append(row)

    if feature_records:
        execution.add_features(feature_records)
        print(f"  Recorded {len(feature_records)} classification predictions with confidence scores")
    else:
        print("  WARNING: No predictions recorded (test loader was empty)")

    if csv_rows:
        csv_file = execution.asset_file_path(
            MLAsset.execution_asset, "prediction_probabilities.csv", ExecAssetType.output_file,
            description="Per-image predicted class and probability distributions over all CIFAR-10 classes",
        )
        fieldnames = ["Image_RID", "Predicted_Class", "Confidence"] + [
            f"prob_{c}" for c in class_names
        ]
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


def load_cifar10_from_execution(
    execution: Execution,
    batch_size: int,
) -> tuple[DataLoader | None, DataLoader | None, list[str]]:
    """Build PyTorch DataLoaders directly from execution dataset bags.

    Uses :meth:`DatasetBag.as_torch_dataset` for training (lazy adapter, no
    on-disk reorganization) and a :class:`_RidAwareImageDataset` wrapper for
    testing (so per-image predictions can be linked back to their RIDs when
    recording features).

    Bags are routed into train vs test by the dataset type tags
    (``"Training"`` / ``"Testing"``). Bags with neither tag are skipped.

    Args:
        execution: DerivaML execution containing downloaded ``DatasetBag``s.
        batch_size: Batch size for both DataLoaders.

    Returns:
        Tuple of ``(train_loader, test_loader, class_names)``. Either loader
        may be ``None`` if the corresponding role isn't present in the
        execution. ``class_names`` is the canonical CIFAR-10 list — this
        matches the model's output index ordering and is independent of which
        labels happen to appear in the bags.
    """
    train_loader: DataLoader | None = None
    test_loader: DataLoader | None = None

    for bag in _flatten_bags(list(execution.datasets)):
        role = _bag_role(bag)
        if role == "training":
            train_dataset = bag.as_torch_dataset(
                element_type="Image",
                sample_loader=_load_image,
                transform=_TRANSFORM,
                targets=["Image_Classification"],
                target_transform=lambda rec: CIFAR10_CLASS_TO_IDX[
                    getattr(rec, "Image_Class", None) or rec.Name
                ],
                missing="skip",  # drop unlabeled training images
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # macOS fork() + MPS/GPU threads can deadlock
            )
            print(f"  Training samples: {len(train_dataset)}")
        elif role == "testing":
            test_dataset = _RidAwareImageDataset(bag)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=_rid_collate,
            )
            print(f"  Testing samples: {len(test_dataset)}")

    return train_loader, test_loader, list(CIFAR10_CLASS_NAMES)


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
    print(f"  Architecture: conv1={conv1_channels}, conv2={conv2_channels}, hidden={hidden_size}")
    if not test_only:
        print(f"  Training: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")

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
    train_loader, test_loader, class_names = load_cifar10_from_execution(
        execution, batch_size
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
            print(f"ERROR: Weights file '{weights_filename}' not found in execution assets.")
            print("  Make sure to include the weights asset in your assets configuration.")
            print(f"  Available assets: {[p.name for p in all_assets]}")
            return

        print(f"\nLoading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # Load model config from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Checkpoint config: {config}")
            # Recreate model with saved config
            model = SimpleCNN(
                conv1_channels=config.get('conv1_channels', conv1_channels),
                conv2_channels=config.get('conv2_channels', conv2_channels),
                hidden_size=config.get('hidden_size', hidden_size),
                dropout_rate=config.get('dropout_rate', dropout_rate),
            ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
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

        # Record predictions to catalog
        print("\nRecording test predictions to catalog...")
        record_test_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
        )

        # Save evaluation results
        results_file = execution.asset_file_path(
            MLAsset.execution_asset, "evaluation_results.txt", ExecAssetType.output_file,
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

    # Training mode: check for training data
    if train_loader is None:
        print("WARNING: No training data found in execution datasets.")
        print("  Make sure your execution configuration includes CIFAR-10 datasets.")
        # Write a status file indicating no data
        status_file = execution.asset_file_path(
            MLAsset.execution_asset, "training_status.txt", ExecAssetType.output_file,
            description="Training status: indicates no training data was available",
        )
        with status_file.open("w") as f:
            f.write("No training data available in execution datasets.\n")
        return

    print(f"  Training batches: {len(train_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Training loop
    print("\nTraining...")
    training_log = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
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
                    test_correct += int((predicted.eq(labels) & labeled_mask).sum().item())

            if test_total_labeled > 0:
                test_acc = 100.0 * test_correct / test_total_labeled
                test_loss = test_loss_sum / max(test_loss_batches, 1)
                log_entry['test_loss'] = test_loss
                log_entry['test_acc'] = test_acc
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%, "
                      f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")
            else:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}% "
                      f"(test set unlabeled — no test metrics)")
        else:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%")

        training_log.append(log_entry)

    # Save model weights
    print("\nSaving model...")
    weights_file = execution.asset_file_path(
        MLAsset.execution_asset, "cifar10_cnn_weights.pt", ExecAssetType.model_file,
        description="Trained CNN model weights, optimizer state, and training log",
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'conv1_channels': conv1_channels,
            'conv2_channels': conv2_channels,
            'hidden_size': hidden_size,
            'dropout_rate': dropout_rate,
        },
        'training_log': training_log,
    }, weights_file)
    print(f"  Saved weights to: {weights_file}")

    # Save training log as text
    log_file = execution.asset_file_path(
        MLAsset.execution_asset, "training_log.txt", ExecAssetType.output_file,
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
        f.write(f"  weight_decay: {weight_decay}\n\n")
        f.write("Training Progress:\n")
        for entry in training_log:
            line = f"  Epoch {entry['epoch']}: train_loss={entry['train_loss']:.4f}, train_acc={entry['train_acc']:.2f}%"
            if 'test_acc' in entry:
                line += f", test_acc={entry['test_acc']:.2f}%"
            f.write(line + "\n")
    print(f"  Saved log to: {log_file}")

    # Record test predictions to catalog if test data is available
    if test_loader is not None:
        print("\nRecording test predictions to catalog...")
        record_test_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
        )

    print("\nTraining complete!")
