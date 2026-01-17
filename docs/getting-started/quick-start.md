# Quick Start Guide

This guide will help you set up and run your first model with DerivaML.

## Step 1: Create Your Repository

1. Go to the [template repository](https://github.com/informatics-isi-edu/deriva-ml-model-template)
2. Click "Use this template" → "Create a new repository"
3. Clone your new repository locally

```bash
git clone https://github.com/your-org/your-model-repo.git
cd your-model-repo
```

## Step 2: Initialize Your Environment

```bash
# Install uv if you haven't already
# See: https://docs.astral.sh/uv/

# Create environment and install dependencies
uv sync

# For notebook support
uv sync --group=jupyter
uv run nbstripout --install
uv run deriva-ml-install-kernel

# For PyTorch support
uv sync --group=pytorch
```

## Step 3: Authenticate with Deriva

```bash
uv run deriva-globus-auth-utils login --host www.eye-ai.org
```

This opens a browser window for Globus authentication. Credentials are cached locally.

## Step 4: Run the Example Model

```bash
# Run with default configuration
uv run src/deriva_run.py

# Dry run (downloads inputs, skips catalog writes)
uv run src/deriva_run.py dry_run=true

# Run with different configuration
uv run src/deriva_run.py model_config=epochs_20
```

## Step 5: Run a Notebook

```bash
# Run notebook and upload to catalog
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb \
  --host www.eye-ai.org \
  --catalog 2 \
  --kernel your-repo-name
```

## Next Steps

- [Create your own model](creating-models.md)
- [Create a new notebook](creating-notebooks.md)
- [Understand the configuration system](../configuration/overview.md)
- [Review coding guidelines](../reference/coding-guidelines.md)
