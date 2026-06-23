# DerivaML Model Template

This repository provides a template for creating ML models integrated with DerivaML, a Python library for reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs for reproducibility.

## Documentation

**[View Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)**

Quick links:
- [Quick Start Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/quick-start/) - Get up and running in minutes
- [Environment Setup](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/environment-setup/) - Detailed setup instructions
- [Creating a New Model](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-models/) - Step-by-step guide for adding models
- [Creating a New Notebook](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-notebooks/) - Step-by-step guide for adding notebooks
- [Configuration Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/configuration/overview/) - Understanding hydra-zen configuration
- [Coding Guidelines](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/coding-guidelines/) - Best practices and standards

## What's Included

This template ships as a **runnable skeleton** — every config group resolves
and dry-runs end to end, with a no-op placeholder model so you can verify the
plumbing before writing any code. You fill it in:

- A Python-first hydra-zen configuration scaffold (no YAML): one self-documenting
  module per config group in `src/configs/` — `deriva.py` (catalog connection),
  `datasets.py`, `model.py`, `experiments.py`, `multiruns.py`, `assets.py`,
  `workflow.py`, and `analysis.py` (notebook config). Each carries a live default
  plus a single commented example you uncomment and fill in.
- CLI entry points via `deriva-ml-run` (models) and `deriva-ml-run-notebook`
  (analysis notebooks).
- A model interface to implement (`src/models/model_protocol.py`) and a
  placeholder `default_model` to replace (`src/configs/model.py`).
- Experiment presets and named multirun (sweep) configurations.
- GitHub Actions for automated versioning and documentation.

For a complete worked example — a real model with multiple variants, a dataset
loader, and an analysis notebook — see
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example).

## Quick Start

> **Note:** Running models requires access to a DerivaML catalog. If you don't have access to an existing Deriva server, you can run one locally using [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker).

### 1. Create Your Repository

Use this template to create a new repository: [Creating a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

### 2. Enable GitHub Pages

After creating your repository from this template, enable GitHub Pages for automatic documentation deployment:

1. Go to your repository **Settings > Pages**
2. Under "Build and deployment", set **Source** to **"GitHub Actions"**
3. Save

The documentation workflow will automatically deploy on each push to main.

### 3. Initialize Environment

```bash
# Create environment and install dependencies
uv sync

# For notebook support
uv sync --group=jupyter
uv run nbstripout --install
uv run deriva-ml-install-kernel

# For PyTorch (only if your model needs it)
uv sync --group=pytorch
```

### 4. Set Up Claude Code (Optional)

If using [Claude Code](https://claude.ai/code), connect to a DerivaML MCP server and install the two skills plugins (`deriva` for generic catalog operations, `deriva-ml` for ML workflows).

**Connect the MCP server.** The MCP server stack is split into two pieces: [`deriva-mcp-core`](https://github.com/informatics-isi-edu/deriva-mcp-core) (catalog/schema/vocabulary tools) plus the [`deriva-ml-mcp`](https://github.com/informatics-isi-edu/deriva-ml-mcp) plugin (DerivaML domain tools — datasets, executions, features, assets). When you stand up a [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker) localhost stack, both ship together as the `deriva-mcp-test` service at `https://localhost/mcp` over HTTP with OAuth.

Register the connection with Claude Code:

```bash
claude mcp add -t http dev-localhost https://localhost/mcp \
    --client-id deriva-mcp --callback-port 8080
```

Verify with `claude mcp list` — the entry should show `dev-localhost: https://localhost/mcp (HTTP) - ✓ Connected`. The `deriva-mcp` client-id is pre-registered with the Credenza auth service in the deriva-docker deployment; `--callback-port 8080` is where Claude listens for the OAuth callback.

**Trust the dev-localhost CA.** Claude Code's MCP HTTP transport runs in Node.js, which has its own CA bundle and won't trust the deriva-docker self-signed cert by default. Without this step, the connection fails with a TLS error:

```bash
# Extract the CA from the running container
mkdir -p ~/.config/deriva
docker cp deriva-mcp-test:/usr/local/share/ca-certificates/deriva-dev-ca.crt \
    ~/.config/deriva/deriva-dev-ca.crt
```

Then add to your workspace's `.claude/settings.local.json`:

```json
{
  "env": {
    "NODE_EXTRA_CA_CERTS": "/Users/<you>/.config/deriva/deriva-dev-ca.crt"
  }
}
```

The first MCP call after this opens an OAuth consent page in your browser; approve once and the bearer token is cached.

For non-dockerized setups (native install, production HTTP, or stdio with a local credential), see the [`deriva-mcp-core` deployment guide](https://github.com/informatics-isi-edu/deriva-mcp-core/blob/main/docs/deployment-guide.md).

**Install the skills plugins.** Both plugins share one marketplace:

```
/plugin marketplace add informatics-isi-edu/deriva-plugins
/plugin install deriva
/plugin install deriva-ml
```

`deriva` covers generic Deriva catalog operations (schema, vocabulary, query patterns, Chaise display); `deriva-ml` adds the DerivaML domain layer (dataset lifecycle, executions, features, experiments, Hydra-zen configs, model development). The `deriva-ml` plugin assumes `deriva` is loaded for cross-references — install both.

To pick up new plugin versions automatically, enable `"autoUpdate": true` for the `deriva-plugins` marketplace entry in `~/.claude/settings.json` and restart Claude Code. Otherwise rerun `/plugin install deriva` and `/plugin install deriva-ml` when a release ships.

For checking versions of the underlying components (deriva-py, deriva-mcp-core, deriva-ml, deriva-ml-mcp), the troubleshooting skills cover it:

- `/deriva:troubleshoot-deriva-errors` — versioning for the foundation (deriva-py, deriva-mcp-core, `deriva` plugin)
- `/deriva-ml:troubleshoot-execution` — versioning for the DerivaML layer (deriva-ml, deriva-ml-mcp, `deriva-ml` plugin)

### 5. Authenticate

```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

### 6. Verify the skeleton resolves

Out of the box the template is a runnable skeleton: a no-op placeholder model
on an empty default dataset. Confirm the config tree resolves before you
customize anything.

```bash
# List the config-group menu (deriva_ml, datasets, model_config, ...)
uv run deriva-ml-run --list-configs

# Show the fully resolved config the default run would use (no execution)
uv run deriva-ml-run +experiment=default --cfg job
```

### 7. Customize

Turn the skeleton into your project by editing the config scaffolds in
`src/configs/` — see [Customizing this template](#customizing-this-template)
below for the ordered walkthrough.

### 8. Run

> **Commit before running.** DerivaML records the git commit hash for
> provenance. Uncommitted changes raise a warning and pollute the audit
> trail of any run that uses them. For fast iteration during development,
> prefix a command with `DERIVA_ML_ALLOW_DIRTY=true` to bypass the check.

```bash
# Run your default model with defaults
uv run deriva-ml-run

# Dry run (resolves + validates against the catalog, no catalog writes)
uv run deriva-ml-run dry_run=true

# Use an experiment preset
uv run deriva-ml-run +experiment=<your_experiment>

# Named multirun (sweep)
uv run deriva-ml-run +multirun=<your_sweep>

# Show available configs
uv run deriva-ml-run --list-configs

# Run an analysis notebook
uv run deriva-ml-run-notebook notebooks/<your_notebook>.ipynb

# Override host/catalog from command line
uv run deriva-ml-run --host <hostname> --catalog <id> +experiment=<your_experiment>
```

## Project Layout

```
.
├── pyproject.toml                  # Project metadata and dependencies
├── CLAUDE.md                       # Claude Code project instructions
├── src/
│   ├── configs/                    # Hydra-zen configurations (Python, no YAML)
│   │   ├── base.py                 # Base DerivaModelConfig
│   │   ├── deriva.py               # Catalog connection settings
│   │   ├── datasets.py             # Dataset specifications
│   │   ├── assets.py               # Asset RID configurations
│   │   ├── workflow.py             # Workflow definitions
│   │   ├── model.py                # Model function + hyperparameter configs
│   │   ├── experiments.py          # Experiment presets (model + dataset)
│   │   ├── multiruns.py            # Named multirun (sweep) configurations
│   │   ├── analysis.py             # Analysis notebook config
│   │   └── dev/                    # Alternate per-environment catalog configs
│   ├── models/                     # Model implementations
│   │   └── model_protocol.py       # The interface a model must implement
│   └── scripts/                    # Data loading / generation scripts (package)
├── notebooks/                      # Analysis notebooks (add your own)
└── docs/                           # Documentation (auto-published)
    └── design/                     # Design docs (plan before you build)
        ├── experiment/             #   per-experiment designs (<slug>.md)
        ├── dataset/                #   per-dataset designs
        ├── feature/                #   per-feature designs
        └── model/                  #   per-model designs
```

## Versioning

Create version tags before significant runs (DerivaML records the git commit for provenance):

```bash
uv run bump-version patch   # Bug fixes
uv run bump-version minor   # New features
uv run bump-version major   # Breaking changes
```

## Customizing this template

Turn the skeleton into your project by working through the config scaffolds in
`src/configs/`. Each module ships with a docstring, a live default, and one
commented example you uncomment and fill in. The ordered walkthrough:

1. **Point at your catalog** — set `hostname` and `catalog_id` in
   `src/configs/deriva.py` (or pass `--host`/`--catalog` on the CLI).
2. **Declare your datasets** — uncomment the `datasets_store(...)` example in
   `src/configs/datasets.py` and fill in your dataset RIDs and versions
   (discover them with `ml.find_datasets()`).
3. **Add your model** — implement the `src/models/model_protocol.py` interface,
   then replace `example_model` in `src/configs/model.py` and point
   `default_model` at your config.
4. **Define experiments** — uncomment the `experiment_store(...)` example in
   `src/configs/experiments.py` to pair a model config with a dataset config
   under one name.
5. **(Optional) sweeps / assets / workflows** — `src/configs/multiruns.py`,
   `src/configs/assets.py`, and `src/configs/workflow.py` each carry a commented
   example for parameter sweeps, input asset RID lists, and workflow metadata.
6. **Rename the project** — set `name` and `description` in `pyproject.toml`.

Verify as you go:

```bash
uv run deriva-ml-run --list-configs              # your configs appear in the menu
uv run deriva-ml-run +experiment=<name> --cfg job  # the resolved config looks right
uv run deriva-ml-run +experiment=<name> dry_run=true  # validates against the catalog
```

For the full per-file walkthrough — exactly which block to uncomment, what each
field means, and how to verify each step — see
[docs/customization.md](docs/customization.md).

For a complete worked reference — every step above filled in with a real model,
datasets, and an analysis notebook — see
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example).

## Using Claude Code with DerivaML

With the MCP server connected and the `deriva` + `deriva-ml` skills plugins installed (see step 4), you can interact with catalogs through natural language and get guided workflows for common tasks. Skills auto-trigger based on context, or you can invoke them directly with `/deriva:<skill-name>` for generic catalog operations (e.g. `/deriva:getting-started`, `/deriva:manage-vocabulary`) and `/deriva-ml:<skill-name>` for ML workflows (e.g. `/deriva-ml:dataset-lifecycle`, `/deriva-ml:experiment-lifecycle`, `/deriva-ml:new-model`).

To see what's available, ask Claude *"help with deriva"* or run `/deriva:help` / `/deriva-ml:help` — these list the skills in each plugin organized by task: environment setup, catalog structure, data management, running experiments, and troubleshooting.

## Further Reading

- [Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)
- [DerivaML Library](https://informatics-isi-edu.github.io/deriva-ml/) - Core library documentation
- [DerivaML User Guide](https://deriva-ml.readthedocs.io/) - Tutorials, concepts, and API reference
- [Hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) - Configuration framework
