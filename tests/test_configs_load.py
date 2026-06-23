"""Smoke tests: skeleton config groups register and load cleanly.

These tests verify that every config module in ``src/configs/`` can be
imported without error and that the required named entries appear in the
hydra-zen store.  They are intentionally lightweight — no catalog
connection, no model execution — just config-tree validity.
"""

import pytest


def test_configs_package_imports():
    """Importing the top-level configs package succeeds without error."""
    import configs  # noqa: F401


def test_model_config_imports():
    """``configs.model`` registers the model_config group without error."""
    import configs.model  # noqa: F401


def test_datasets_config_imports():
    """``configs.datasets`` registers the datasets group without error."""
    import configs.datasets  # noqa: F401


def test_experiments_config_imports():
    """``configs.experiments`` registers the experiment group without error."""
    import configs.experiments  # noqa: F401


def test_base_config_imports():
    """``configs.base`` exports ``DerivaModelConfig`` and its companions."""
    from configs.base import BaseConfig, DerivaModelConfig  # noqa: F401

    assert DerivaModelConfig is not None


def test_default_model_registered():
    """The ``model_config`` group must contain a ``default_model`` entry.

    ``src/configs/base.py`` lists ``{"model_config": "default_model"}`` in
    ``_HYDRA_DEFAULTS``; the whole config tree fails to resolve without it.
    """
    import configs.model  # noqa: F401 — triggers registration
    from hydra_zen import store

    entry = store.get_entry("model_config", "default_model")
    assert entry["name"] == "default_model"
    assert entry["group"] == "model_config"


def test_default_dataset_registered():
    """The ``datasets`` group must contain a ``default_dataset`` entry.

    ``src/configs/base.py`` lists ``{"datasets": "default_dataset"}`` in
    ``_HYDRA_DEFAULTS``; a missing entry causes an immediate config error.
    """
    import configs.datasets  # noqa: F401 — triggers registration
    from hydra_zen import store

    entry = store.get_entry("datasets", "default_dataset")
    assert entry["name"] == "default_dataset"
    assert entry["group"] == "datasets"


def test_required_dataset_sentinels_registered():
    """The ``no_datasets`` and ``none`` sentinel configs are present.

    These are used by notebook analysis workflows that consume asset RIDs
    instead of datasets.
    """
    import configs.datasets  # noqa: F401
    from hydra_zen import store

    for name in ("no_datasets", "none"):
        entry = store.get_entry("datasets", name)
        assert entry["name"] == name, f"Sentinel '{name}' missing from datasets group"


def test_default_experiment_registered():
    """The ``experiment`` group must contain a ``default`` entry.

    The template ships a single live experiment (``default``) so that
    ``--list-configs`` always shows at least one entry and ``+experiment=default``
    resolves without error.
    """
    import configs.experiments  # noqa: F401
    from hydra_zen import store

    entry = store.get_entry("experiment", "default")
    assert entry["name"] == "default"
    assert entry["group"] == "experiment"


def test_no_cifar_entries_registered():
    """Verify the CIFAR implementation has been fully stripped.

    After the A3/A4 strip, no CIFAR-named configs should appear in the
    hydra-zen store.  This guards against accidental re-introduction.
    """
    import configs.model  # noqa: F401
    import configs.datasets  # noqa: F401
    import configs.experiments  # noqa: F401
    from hydra_zen import store

    cifar_groups = [
        ("model_config", "cifar10_quick"),
        ("model_config", "cifar10_default"),
        ("model_config", "cifar10_large"),
        ("datasets", "cifar10_split"),
        ("datasets", "cifar10_labeled_split"),
        ("experiment", "cifar10_quick"),
    ]
    for group, name in cifar_groups:
        with pytest.raises(KeyError):
            store.get_entry(group, name)
