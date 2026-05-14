"""Unit tests for src/scripts/_cifar10_source.py."""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts._cifar10_source import (
    download_cifar10_archive,
    extract_cifar10_to_png,
    load_batch,
    CIFAR10_URL,
)


def _fake_batch(num_images: int, label_offset: int = 0) -> dict:
    """Build a fake CIFAR-10 batch matching the upstream pickle format."""
    return {
        b"data": np.zeros((num_images, 3072), dtype=np.uint8),
        b"labels": [(i + label_offset) % 10 for i in range(num_images)],
        b"filenames": [f"img_{i + label_offset}.png".encode() for i in range(num_images)],
        b"batch_label": b"testing batch",
    }


def test_download_uses_cache_when_present(tmp_path):
    cache = tmp_path / "cifar-10-python.tar.gz"
    cache.write_bytes(b"fake-archive-bytes")

    # urlretrieve should NOT be called when the cache exists.
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        result = download_cifar10_archive(cache_path=cache)

    assert result == cache
    mock_retrieve.assert_not_called()
