"""CIFAR-10 data source — download from the Toronto open mirror.

This module isolates the data-source layer (network fetch, extract,
batch decode) so it can be unit-tested without touching DerivaML.

The upstream archive is the canonical Python pickle distribution at
``https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz``. It
contains six pickle files (``data_batch_1`` .. ``data_batch_5``
and ``test_batch``) plus a ``batches.meta`` file. Each batch has
labels for every image — the Toronto distribution is fully labeled
on both train and test, unlike the Kaggle competition format.
"""

from __future__ import annotations

import logging
import pickle
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deriva-ml-model-template"


def download_cifar10_archive(cache_path: Path | None = None) -> Path:
    """Download the CIFAR-10 archive, or return the cached copy.

    Args:
        cache_path: Where to store the archive. Defaults to
            ``~/.cache/deriva-ml-model-template/cifar-10-python.tar.gz``.

    Returns:
        Path to the (now-present) archive file.

    Example:
        >>> archive = download_cifar10_archive()
        >>> archive.name
        'cifar-10-python.tar.gz'
    """
    if cache_path is None:
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DEFAULT_CACHE_DIR / "cifar-10-python.tar.gz"

    if cache_path.exists():
        logger.info(f"Using cached CIFAR-10 archive at {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
    urllib.request.urlretrieve(CIFAR10_URL, cache_path)
    logger.info(f"Downloaded to {cache_path}")
    return cache_path


def load_batch(batch_path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    """Load one CIFAR-10 pickle batch into image array + labels.

    Args:
        batch_path: Path to a CIFAR-10 batch pickle (``data_batch_N``
            or ``test_batch``).

    Returns:
        Tuple of ``(images, labels, filenames)``:
          - images: ``np.ndarray`` of shape ``(N, 32, 32, 3)``, ``uint8``,
            HWC, RGB.
          - labels: list of int class indices (0-9).
          - filenames: list of original filenames (str, decoded from bytes).

    Example:
        >>> imgs, labels, names = load_batch(Path("data_batch_1"))
        >>> imgs.shape
        (10000, 32, 32, 3)
    """
    with batch_path.open("rb") as fh:
        batch = pickle.load(fh, encoding="bytes")

    raw = batch[b"data"]
    images = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = list(batch[b"labels"])
    filenames = [fn.decode("utf-8") for fn in batch[b"filenames"]]
    return images, labels, filenames


def extract_cifar10_to_png(*args, **kwargs):
    """Placeholder — implemented in Task A4."""
    raise NotImplementedError("Task A4")
