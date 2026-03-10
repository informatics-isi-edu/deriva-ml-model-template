# Quick Start Guide

This guide will help you set up and run your first model with DerivaML.

!!! note "Catalog Required"
    This template requires access to a DerivaML catalog. The `deriva-ml-run` CLI connects
    to a catalog to track code provenance, configuration, and outputs. If you don't have
    access to an existing Deriva server, you can run one locally using
    [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker).

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

## Step 3: Set Up Claude Code (Optional)

If using [Claude Code](https://claude.ai/code), install the DerivaML MCP server and skills plugin for catalog tools and guided workflows.

**Install the MCP server** (provides catalog tools, dataset management, execution tracking):

```bash
# Using Docker (recommended)
docker pull ghcr.io/informatics-isi-edu/deriva-mcp:latest
```

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "deriva": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -e HOME=$HOME -v $HOME/.deriva:$HOME/.deriva:ro -v $HOME/.bdbag:$HOME/.bdbag -v $HOME/.deriva-ml:$HOME/.deriva-ml ghcr.io/informatics-isi-edu/deriva-mcp:latest"
      ]
    }
  }
}
```

**Install the skills plugin** (provides workflow guidance and best practices):

```
/plugin marketplace add informatics-isi-edu/deriva-mcp
/plugin install deriva
```

See the [deriva-mcp README](https://github.com/informatics-isi-edu/deriva-mcp) for full setup options.

## Step 4: Authenticate with Deriva

```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

This opens a browser window for Globus authentication. Credentials are cached locally.

## Step 5: Set Up the Catalog

Before running the example model, load CIFAR-10 data into your catalog:

```bash
# Load 500 images (quick start)
uv run load-cifar10 --host <hostname> --catalog_id <catalog_id> --num_images 500

# Load more images for full experiments
uv run load-cifar10 --host <hostname> --catalog_id <catalog_id> --num_images 5000
```

This creates datasets, uploads images, and sets up the schema needed by the example model. See the [CIFAR-10 example](../reference/cifar10-example.md) for details on dataset types and configurations.

## Step 6: Run the Example Model

First, update `src/configs/deriva.py` with your catalog's hostname and catalog ID, then discover what configurations are available:

```bash
# Show available configs, experiment presets, and multiruns
uv run deriva-ml-run --info
```

Then run the model:

```bash
# Run with default configuration
uv run deriva-ml-run

# Run with an experiment preset
uv run deriva-ml-run +experiment=cifar10_quick

# Dry run (downloads inputs, skips catalog writes)
uv run deriva-ml-run dry_run=true

# Override host and catalog from the command line
uv run deriva-ml-run --host <hostname> --catalog <catalog_id>

# Override a specific model config
uv run deriva-ml-run model_config=cifar10_quick

# Run a named multirun
uv run deriva-ml-run +multirun=quick_vs_extended
```

## Step 7: Run a Notebook

```bash
# Run notebook and upload results to catalog
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb

# Discover available notebook configs
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --info

# Override config values using Hydra overrides
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb assets=my_assets

# Override host and catalog from the command line
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \
  --host <hostname> --catalog <catalog_id>
```

## Next Steps

- [Create your own model](creating-models.md)
- [Create a new notebook](creating-notebooks.md)
- [Understand the configuration system](../configuration/overview.md)
