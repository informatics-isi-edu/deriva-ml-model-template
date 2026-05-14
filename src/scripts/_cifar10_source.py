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
import urllib.request
from pathlib import Path

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


def load_batch(*args, **kwargs):
    """Placeholder — implemented in Task A3."""
    raise NotImplementedError("Task A3")


def extract_cifar10_to_png(*args, **kwargs):
    """Placeholder — implemented in Task A4."""
    raise NotImplementedError("Task A4")
