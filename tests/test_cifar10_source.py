"""Unit tests for src/scripts/_cifar10_source.py."""

from __future__ import annotations

import pickle
from unittest.mock import patch

import numpy as np

from scripts._cifar10_source import (
    download_cifar10_archive,
    load_batch,
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


def test_load_batch_returns_images_labels_filenames(tmp_path):
    batch_path = tmp_path / "data_batch_1"
    with batch_path.open("wb") as fh:
        pickle.dump(_fake_batch(num_images=4), fh)

    images, labels, filenames = load_batch(batch_path)

    assert images.shape == (4, 32, 32, 3)
    assert images.dtype == np.uint8
    assert labels == [0, 1, 2, 3]
    assert filenames == ["img_0.png", "img_1.png", "img_2.png", "img_3.png"]


def test_load_batch_preserves_channel_order(tmp_path):
    """A wrong reshape/transpose axis order would silently pass the shape
    assertion in the other test. This pins channel ordering: CIFAR-10's
    packed layout is [R(1024), G(1024), B(1024)] per row, and the decoded
    image at HWC index (h, w, c) must come from the c-th 1024-block."""
    # Build one image where R=10, G=20, B=30 at every pixel.
    flat = np.empty((1, 3072), dtype=np.uint8)
    flat[0, 0:1024] = 10   # R block
    flat[0, 1024:2048] = 20  # G block
    flat[0, 2048:3072] = 30  # B block

    batch = {
        b"data": flat,
        b"labels": [7],
        b"filenames": [b"channel_test.png"],
        b"batch_label": b"testing batch",
    }
    batch_path = tmp_path / "data_batch_channel"
    with batch_path.open("wb") as fh:
        pickle.dump(batch, fh)

    images, _, _ = load_batch(batch_path)

    assert images.shape == (1, 32, 32, 3)
    # Every pixel should be (R=10, G=20, B=30).
    assert images[0, 0, 0, 0] == 10
    assert images[0, 0, 0, 1] == 20
    assert images[0, 0, 0, 2] == 30
    assert images[0, 15, 27, 0] == 10  # random pixel: still R=10
    assert images[0, 31, 31, 2] == 30  # last pixel: still B=30
