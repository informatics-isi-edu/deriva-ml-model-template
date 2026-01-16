"""
CIFAR-10 2-Layer CNN Model.

A simple convolutional neural network for CIFAR-10 classification.
This model demonstrates how to integrate PyTorch training with DerivaML
execution tracking and asset management.

The model architecture:
- Conv2d(3, 32) -> ReLU -> MaxPool2d
- Conv2d(32, 64) -> ReLU -> MaxPool2d
- Linear(64*8*8, hidden_size) -> ReLU
- Linear(hidden_size, 10)

Expected accuracy: ~60-70% with default parameters.

Data Loading:
The model uses DerivaML's restructure_assets() method to reorganize downloaded
images into a directory structure that torchvision's ImageFolder can consume:

    data_dir/
        training/
            airplane/
            automobile/
            ...
        testing/
            airplane/
            automobile/
            ...

Test Evaluation:
After training, the model evaluates on the test set and records per-image
classification predictions to the DerivaML catalog using the Image_Classification
feature. This enables tracking of model predictions and comparison with ground
truth labels.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from deriva_ml import DerivaML, MLAsset, ExecAssetType
from deriva_ml.execution import Execution


def build_filename_to_rid_map(ml_instance: DerivaML) -> dict[str, str]:
    """Build a mapping from filenames to RIDs from the Image asset table.

    Queries all Image assets in the catalog to create a lookup table
    that maps image filenames to their catalog RIDs.

    Args:
        ml_instance: DerivaML instance for catalog access.

    Returns:
        Dictionary mapping filename (without path) to RID.
    """
    assets = ml_instance.list_assets("Image")
    return {a["Filename"]: a["RID"] for a in assets if "Filename" in a and "RID" in a}


def record_test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    filename_to_rid: dict[str, str],
    execution: Execution,
    ml_instance: DerivaML,
    device: torch.device,
) -> int:
    """Record per-image classification predictions to the DerivaML catalog.

    Runs inference on the test set and records each prediction as an
    Image_Classification feature value, linking the predicted class
    to the image RID. Also includes the confidence (probability) of the
    predicted class.

    Additionally saves a CSV file with full probability distributions for
    all classes, enabling detailed ROC curve analysis.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data (from ImageFolder).
        class_names: List of class names in index order.
        filename_to_rid: Mapping from image filename to RID.
        execution: DerivaML execution context.
        ml_instance: DerivaML instance for catalog access.
        device: PyTorch device for inference.

    Returns:
        Number of predictions recorded.
    """
    model.eval()

    # Get the feature record class for Image_Classification
    ImageClassification = ml_instance.feature_record_class("Image", "Image_Classification")

    # Collect all predictions
    feature_records = []

    # Collect data for CSV output (full probability distributions)
    csv_rows = []

    # Get the underlying dataset to access file paths
    test_dataset = test_loader.dataset

    with torch.no_grad():
        sample_idx = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Compute softmax probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Get predicted class and confidence
            confidences, predicted = probabilities.max(1)

            # Process each sample in the batch
            for i in range(inputs.size(0)):
                # Get the file path for this sample
                img_path, _ = test_dataset.samples[sample_idx]
                filename = Path(img_path).name

                # Get probability distribution for this sample
                probs = probabilities[i].cpu().numpy()

                # Look up the RID
                rid = filename_to_rid.get(filename)
                if rid:
                    predicted_class = class_names[predicted[i].item()]
                    confidence = confidences[i].item()

                    # Record to catalog with confidence
                    feature_records.append(
                        ImageClassification(
                            Image=rid,
                            Image_Class=predicted_class,
                            Confidence=confidence,
                        )
                    )

                    # Build CSV row with full probability distribution
                    csv_row = {
                        "Image_RID": rid,
                        "Filename": filename,
                        "Predicted_Class": predicted_class,
                        "Confidence": confidence,
                    }
                    # Add probability for each class
                    for j, class_name in enumerate(class_names):
                        csv_row[f"prob_{class_name}"] = probs[j]
                    csv_rows.append(csv_row)

                sample_idx += 1

    # Bulk add all predictions to the execution
    if feature_records:
        execution.add_features(feature_records)
        print(f"  Recorded {len(feature_records)} classification predictions with confidence scores")
    else:
        print("  WARNING: No predictions could be recorded (no RID matches)")

    # Save full probability distributions to CSV
    if csv_rows:
        csv_file = execution.asset_file_path(
            MLAsset.execution_asset, "prediction_probabilities.csv", ExecAssetType.output_file
        )
        fieldnames = ["Image_RID", "Filename", "Predicted_Class", "Confidence"] + [
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
    """

    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_size: int = 128,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # After two 2x2 pooling operations: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(conv2_channels * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
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
) -> tuple[DataLoader | None, DataLoader | None, list[str], Path]:
    """Load CIFAR-10 data from DerivaML execution datasets.

    Uses DerivaML's restructure_assets() method to organize downloaded images
    into the directory structure expected by torchvision's ImageFolder.
    Images are grouped by the Image_Classification feature value.

        data_dir/
            training/          # From dataset with type "Training"
                airplane/      # From Image_Classification feature value
                automobile/
                ...
            testing/           # From dataset with type "Testing"
                airplane/
                ...

    Args:
        execution: DerivaML execution containing downloaded datasets.
        batch_size: Batch size for DataLoader.

    Returns:
        Tuple of (train_loader, test_loader, class_names, data_dir).
        - train_loader: DataLoader for training data (or None)
        - test_loader: DataLoader for test data with RID tracking (or None)
        - class_names: List of class names in index order
        - data_dir: The restructured directory path
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create output directory for restructured data
    data_dir = execution.working_dir / "cifar10_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Restructure assets from each dataset
    # The type_selector picks which dataset type to use for the directory name
    def type_selector(types: list[str]) -> str:
        """Select dataset type for directory structure."""
        type_lower = [t.lower() for t in types]
        if "training" in type_lower:
            return "training"
        elif "testing" in type_lower:
            return "testing"
        elif types:
            return types[0].lower()
        return "unknown"

    # Process each dataset in the execution
    for dataset in execution.datasets:
        # Restructure images by dataset type and label
        # This creates: data_dir/<dataset_type>/<label>/image.png
        dataset.restructure_assets(
            asset_table="Image",
            output_dir=data_dir,
            group_by=["Image_Classification"],  # Group by feature to create class subdirs
            use_symlinks=True,
            type_selector=type_selector,
        )

    # Create DataLoaders
    train_loader = None
    test_loader = None
    class_names: list[str] = []

    # Find training and testing directories (may be nested under dataset type like "split")
    def find_data_dir(base_dir: Path, target_name: str) -> Path | None:
        """Find a directory with the given name, searching recursively."""
        # First check direct path
        direct_path = base_dir / target_name
        if direct_path.exists() and any(direct_path.iterdir()):
            return direct_path
        # Search recursively for nested datasets
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                nested_path = subdir / target_name
                if nested_path.exists() and any(nested_path.iterdir()):
                    return nested_path
        return None

    train_dir = find_data_dir(data_dir, "training")
    if train_dir:
        train_dataset = ImageFolder(train_dir, transform=transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
        )
        class_names = train_dataset.classes
        print(f"  Training classes: {train_dataset.classes}")
        print(f"  Training samples: {len(train_dataset)}")

    test_dir = find_data_dir(data_dir, "testing")
    if test_dir:
        test_dataset = ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        # Use test classes if no training data
        if not class_names:
            class_names = test_dataset.classes
        print(f"  Testing classes: {test_dataset.classes}")
        print(f"  Testing samples: {len(test_dataset)}")

    return train_loader, test_loader, class_names, data_dir


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
    ml_instance: DerivaML = None,
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

    # Load data from execution datasets
    print("\nLoading and restructuring data from execution datasets...")
    train_loader, test_loader, class_names, data_dir = load_cifar10_from_execution(
        execution, batch_size
    )
    print(f"  Data directory: {data_dir}")

    # Build filename -> RID mapping for recording predictions
    filename_to_rid = build_filename_to_rid_map(ml_instance)
    print(f"  Loaded {len(filename_to_rid)} filename-to-RID mappings")

    # Test-only mode: load weights and run evaluation
    if test_only:
        if test_loader is None:
            print("ERROR: No test data found in execution datasets.")
            print("  Test-only mode requires a dataset with type 'Testing'.")
            return

        print(f"  Test batches: {len(test_loader)}")

        # For test-only mode, use standard CIFAR-10 class names since test data may not have labels
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
        if class_names == ['unknown'] or len(class_names) != 10:
            print(f"  Using standard CIFAR-10 class names (test data has: {class_names})")
            class_names = cifar10_classes

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
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_correct = 0
        test_total = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total
        test_loss = test_loss / len(test_loader)
        print(f"  Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

        # Record predictions to catalog
        if class_names and filename_to_rid:
            print("\nRecording test predictions to catalog...")
            record_test_predictions(
                model=model,
                test_loader=test_loader,
                class_names=class_names,
                filename_to_rid=filename_to_rid,
                execution=execution,
                ml_instance=ml_instance,
                device=device,
            )

        # Save evaluation results
        results_file = execution.asset_file_path(
            MLAsset.execution_asset, "evaluation_results.txt", ExecAssetType.output_file
        )
        with results_file.open("w") as f:
            f.write("CIFAR-10 CNN Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Weights file: {weights_filename}\n")
            f.write(f"Test samples: {test_total}\n")
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
            MLAsset.execution_asset, "training_status.txt", ExecAssetType.output_file
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

        # Evaluate on test set if available
        if test_loader:
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()

            test_acc = 100.0 * test_correct / test_total
            test_loss = test_loss / len(test_loader)
            log_entry['test_loss'] = test_loss
            log_entry['test_acc'] = test_acc

            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%, "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")
        else:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%")

        training_log.append(log_entry)

    # Save model weights
    print("\nSaving model...")
    weights_file = execution.asset_file_path(
        MLAsset.execution_asset, "cifar10_cnn_weights.pt", ExecAssetType.model_file
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
        MLAsset.execution_asset, "training_log.txt", ExecAssetType.output_file
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
    if test_loader and class_names and filename_to_rid:
        print("\nRecording test predictions to catalog...")
        record_test_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            filename_to_rid=filename_to_rid,
            execution=execution,
            ml_instance=ml_instance,
            device=device,
        )

    print("\nTraining complete!")
