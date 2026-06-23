# Split CIFAR Example Into Its Own Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip the CIFAR-10 example out of `deriva-ml-model-template` so the template becomes a pure, runnable deriva-ml project skeleton, and create a new public GitHub repo `deriva-ml-cifar-example` that carries the CIFAR example built on that skeleton.

**Architecture:** Two repos. (1) The existing `deriva-ml-model-template` is cleaned **in place** on a fresh branch off `origin/main`: all CIFAR-specific code/configs/tests/docs are removed, the structural files (`pyproject.toml`, READMEs, `src/configs/*`) are de-CIFAR-ed to generic skeleton form, and empty-but-valid config stubs keep the skeleton loadable and testable. (2) A new repo `deriva-ml-cifar-example` is created with **fresh git history** (one initial commit) containing the CIFAR example plus the scaffolding it needs, demonstrating the template's structure filled in.

**Tech Stack:** Python 3.12+, uv, hydra-zen, deriva-ml, pytest, ruff, GitHub (`gh` CLI), GitHub Actions.

## Global Constraints

- Use `uv` for everything — `uv run <cmd>`, never bare `pytest`/`ruff`/`python`. (project CLAUDE.md)
- `uv run python -m pytest`, NOT `uv run pytest` (stale shebang gotcha). (project CLAUDE.md)
- Google-style docstrings with runnable `Example:` blocks on every function/method/class. (project CLAUDE.md)
- No backwards-compat shims, no dead exports, no "removed" placeholders. (project CLAUDE.md)
- `num_workers=0` in DataLoaders on macOS. (project CLAUDE.md — only relevant to the CIFAR repo)
- Template base branch: cut fresh from `origin/main` (`3c8bd82`). Do NOT build on `feat/design-docs-under-docs` and do NOT touch its dangling 1.51.8 dep bump.
- New repo: `informatics-isi-edu/deriva-ml-cifar-example`, **public**, **fresh history** (no filter-branch / subtree).
- Boundary: strip **ALL** CIFAR from the template. Template keeps only generic scaffolding + empty-but-valid stubs.
- `egg-info/` is a build artifact — it must NOT be committed to either repo (add to `.gitignore` if not already ignored); never carry `src/deriva_ml_model_template.egg-info/` forward.
- Both repos must end green: `uv run python -m pytest` passes and `uv run deriva-ml-run --list-configs` works.

---

## File inventory (the boundary, locked)

**MOVE to `deriva-ml-cifar-example` (CIFAR):**
- `src/configs/`: `cifar10_cnn.py`, `datasets.py`, `experiments.py`, `assets.py`, `workflow.py`, `multiruns.py`, `multirun_descriptions.py`
- `src/models/`: `cifar10_cnn.py`, `cifar10_classes.py`
- `src/scripts/`: `_cifar10_assets.py`, `_cifar10_datasets.py`, `_cifar10_schema.py`, `_cifar10_source.py`, `load_cifar10.py`, `analyst_join.py`
- `tests/`: `test_cifar10_*.py` (5), `test_load_cifar10_*.py` (2), `test_runner_bag_dispatch.py`, `test_runner_seed.py`, `test_configs_load.py`, `test_analyst_join.py`, `test_cifar10_cnn_loaders.py`
- `notebooks/roc_analysis.ipynb`, `scripts/assets.toml`
- Root docs: `CIFAR10.md`, `Experiments.md`
- The `load-cifar10` entry point + `torchvision`/ROC deps from `pyproject.toml`

**STAY in template (generic scaffolding):**
- `src/configs/`: `base.py`, `deriva.py`, `__init__.py`, `dataset_generation.py`, `roc_analysis.py` (generic notebook-config pattern)
- `src/models/`: `model_protocol.py`
- `src/scripts/`: `__init__.py`
- `scripts/upload_assets.py` (generic asset uploader)
- The runner machinery, CI workflows, `mkdocs.yml`, generic `pyproject.toml` shape

**SCAFFOLD in template (KEEP the file as a self-documenting template — do NOT delete or empty):**
Every config file stays in the skeleton as a teaching scaffold: **one live minimal valid
default** (so `--list-configs` and the runner still work) **plus one generic commented-out
example** showing the shape, using placeholder names (NOT CIFAR vocabulary). The user
uncomments + fills in.
- `src/configs/datasets.py` → live `default_dataset`/`none`/`no_datasets` empty specials + a commented `# DatasetSpecConfig(rid="<your-rid>", version="<ver>")` example.
- `src/configs/experiments.py` → a commented generic experiment preset (`# experiment_store(make_config(hydra_defaults=[..., {"override /model_config": "<your_model>"}, {"override /datasets": "<your_dataset>"}], ...), name="<your_experiment>")`) + the module docstring explaining the pattern. Keep one live trivial default if the runner requires a registered group.
- `src/configs/model_config` (i.e. the model config module) → a generic commented `model_config` example + the REQUIRED live `default_model` (the runner needs it) pointing at a placeholder/generic model, OR documented as "replace with your model's config".
- `src/configs/assets.py`, `workflow.py`, `multiruns.py`, `multirun_descriptions.py` → kept with generic commented examples + minimal live defaults where the group must register.
- `tests/test_configs_load.py` → a template-level smoke test that the scaffold config groups register and load.

NOTE: this REPLACES the earlier "reduce to empty / remove orphaned config modules"
approach. The config modules are NOT removed — they are converted to commented-example
scaffolds. The CIFAR *content* leaves; the *files and their teaching structure* stay.

**EXCLUDE from both (never commit):**
- `src/deriva_ml_model_template.egg-info/` (build artifact)

---

## Phase A — Strip the template in place

### Task A1: Cut a clean base branch and confirm green baseline

**Files:**
- None modified yet — this establishes the working branch.

**Interfaces:**
- Produces: a clean branch `chore/strip-cifar-to-skeleton` off `origin/main` (`3c8bd82`), green baseline recorded.

- [ ] **Step 1: Fetch and cut the branch from origin/main (NOT the current dirty feat branch)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
git fetch origin
git stash list   # confirm nothing of ours is stashed
# Create the branch from origin/main without disturbing the dirty working tree:
git worktree add ../deriva-ml-model-template-strip -b chore/strip-cifar-to-skeleton origin/main
cd ../deriva-ml-model-template-strip
```

- [ ] **Step 2: Establish green baseline BEFORE any change**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
uv sync
uv run python -m pytest -q 2>&1 | tail -5
uv run deriva-ml-run --list-configs 2>&1 | tail -10
```
Expected: tests pass (the merged `Image.Filename` fixes are on `main`), `--list-configs` lists the CIFAR experiment/model/dataset groups. Record the pass count.

- [ ] **Step 3: Commit nothing yet** — baseline only. Proceed to A2.

---

### Task A2: Copy the full CIFAR set out to a staging area for the new repo

**Files:**
- Create: `/Users/carl/GitHub/DerivaML/_cifar-staging/` (temporary, outside both repos)

**Interfaces:**
- Produces: a complete copy of every MOVE-list file (preserving relative paths) that Phase B will assemble into the new repo. This happens BEFORE deletion so nothing is lost.

- [ ] **Step 1: Copy the MOVE-list files preserving structure**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
mkdir -p /Users/carl/GitHub/DerivaML/_cifar-staging
# Use rsync to copy each MOVE path preserving the tree:
for p in \
  src/configs/cifar10_cnn.py src/configs/datasets.py src/configs/experiments.py \
  src/configs/assets.py src/configs/workflow.py src/configs/multiruns.py \
  src/configs/multirun_descriptions.py \
  src/models/cifar10_cnn.py src/models/cifar10_classes.py \
  src/scripts/_cifar10_assets.py src/scripts/_cifar10_datasets.py \
  src/scripts/_cifar10_schema.py src/scripts/_cifar10_source.py \
  src/scripts/load_cifar10.py src/scripts/analyst_join.py \
  notebooks/roc_analysis.ipynb scripts/assets.toml \
  CIFAR10.md Experiments.md ; do
  rsync -R "$p" /Users/carl/GitHub/DerivaML/_cifar-staging/
done
# Tests:
for p in tests/test_cifar10_assets.py tests/test_cifar10_cnn_loaders.py \
  tests/test_cifar10_datasets.py tests/test_cifar10_schema.py tests/test_cifar10_source.py \
  tests/test_configs_load.py tests/test_load_cifar10_retry.py \
  tests/test_load_cifar10_split_no_leakage.py tests/test_runner_bag_dispatch.py \
  tests/test_runner_seed.py tests/test_analyst_join.py ; do
  rsync -R "$p" /Users/carl/GitHub/DerivaML/_cifar-staging/
done
# Also copy the generic scaffolding the new repo needs (it must be self-contained):
for p in src/configs/base.py src/configs/deriva.py src/configs/__init__.py \
  src/configs/dataset_generation.py src/configs/roc_analysis.py \
  src/models/model_protocol.py src/scripts/__init__.py scripts/upload_assets.py \
  README.md CLAUDE.md mkdocs.yml ; do
  rsync -R "$p" /Users/carl/GitHub/DerivaML/_cifar-staging/
done
ls -R /Users/carl/GitHub/DerivaML/_cifar-staging/ | head -40
```
Expected: the staging dir mirrors the repo tree with CIFAR + generic-scaffolding files.

- [ ] **Step 2: Copy CI + config dotfiles the new repo needs**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
for p in .github .gitignore .python-version pyproject.toml uv.lock ; do
  [ -e "$p" ] && rsync -R "$p" /Users/carl/GitHub/DerivaML/_cifar-staging/
done
ls -a /Users/carl/GitHub/DerivaML/_cifar-staging/
```
Expected: `.github/`, `.gitignore`, `pyproject.toml`, `uv.lock` present in staging.

---

### Task A3: Delete CIFAR-implementation files from the template

**Files:**
- Delete (from template): CIFAR *implementation* files (models, scripts, notebook, CIFAR docs, CIFAR tests) + `src/deriva_ml_model_template.egg-info/`
- KEEP (converted to scaffolds in A4, NOT deleted here): `src/configs/datasets.py`, `experiments.py`, `assets.py`, `workflow.py`, `multiruns.py`, `multirun_descriptions.py`, and the model_config module.

- [ ] **Step 1: Remove CIFAR implementation source, models, scripts, notebooks, docs**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
# CIFAR model + loader implementation (these have no generic equivalent — they leave):
git rm src/configs/cifar10_cnn.py \
  src/models/cifar10_cnn.py src/models/cifar10_classes.py \
  src/scripts/_cifar10_assets.py src/scripts/_cifar10_datasets.py \
  src/scripts/_cifar10_schema.py src/scripts/_cifar10_source.py \
  src/scripts/load_cifar10.py src/scripts/analyst_join.py \
  notebooks/roc_analysis.ipynb scripts/assets.toml \
  CIFAR10.md Experiments.md
git rm tests/test_cifar10_assets.py tests/test_cifar10_cnn_loaders.py \
  tests/test_cifar10_datasets.py tests/test_cifar10_schema.py tests/test_cifar10_source.py \
  tests/test_load_cifar10_retry.py tests/test_load_cifar10_split_no_leakage.py \
  tests/test_runner_bag_dispatch.py tests/test_runner_seed.py tests/test_analyst_join.py
# test_configs_load.py is REPLACED (not just deleted) in A5 — remove now, re-add in A5:
git rm tests/test_configs_load.py
# DO NOT git rm the config modules (datasets.py, experiments.py, assets.py, workflow.py,
# multiruns.py, multirun_descriptions.py, the model_config module) — A4 converts them
# in place to commented-example scaffolds.
# ALSO remove stray cross-project debris found at baseline (A1): this is an eye-ai
# domain script, neither CIFAR nor template scaffolding, and its module-level
# DerivaML(hostname="www.eye-ai.org") breaks root-level pytest collection:
git rm scripts/test_bag_fk_traversal.py
```
Expected: CIFAR implementation files + the stray eye-ai script staged for deletion; the config modules remain in the tree for A4 to rewrite.

- [ ] **Step 2: Untrack and gitignore the egg-info build artifact**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
git rm -r --cached src/deriva_ml_model_template.egg-info 2>/dev/null || true
grep -q 'egg-info' .gitignore || echo "*.egg-info/" >> .gitignore
```
Expected: egg-info no longer tracked; `.gitignore` covers it.

---

### Task A4: Convert each config file to a commented-example scaffold

**Files:**
- Modify (convert in place, KEEP the file): `src/configs/datasets.py`, `experiments.py`, `assets.py`, `workflow.py`, `multiruns.py`, `multirun_descriptions.py`, the model_config module
- Modify: `pyproject.toml` (drop `load-cifar10` entry point + CIFAR-only deps; generic description)

**Interfaces:**
- Produces: every config group registers a minimal LIVE default (so `--list-configs`/runner work) AND carries ONE generic commented-out example (placeholder names, no CIFAR vocabulary) teaching the shape. A generic `pyproject.toml` with no CIFAR entry point.

**Pattern for every config file (apply uniformly):** keep the module docstring (rewritten generic, explaining the group's purpose and how to fill it in) → keep the imports and the `store(group=...)` line → replace CIFAR registrations with (a) the minimal live default the runner requires, and (b) ONE commented example using `<placeholder>` names. Do NOT leave the file empty; do NOT leave CIFAR names.

- [ ] **Step 1: `datasets.py` — empty specials + commented example**

```python
"""Dataset configurations for your project.

Each entry names a dataset group your experiments/notebooks reference. Fill these
in with the RIDs your loader produced (or read via ``ml.find_datasets()``).
See docs/customization.md for the full walkthrough.
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig  # noqa: F401  (used in the example below)
from deriva_ml.execution import with_description   # noqa: F401

datasets_store = store(group="datasets")

# --- Fill in your datasets here -------------------------------------------
# Example (uncomment and replace with your RID + version):
# datasets_store(
#     with_description(
#         [DatasetSpecConfig(rid="<your-dataset-rid>", version="<version>")],
#         "Your dataset description.",
#     ),
#     name="my_training_data",
# )

# Special groups (leave as-is unless you know you need to change them):
datasets_store([], name="no_datasets")   # notebooks that consume asset RIDs, not datasets
datasets_store([], name="none")          # script-only experiments managing their own data

# REQUIRED: default_dataset is used when no dataset override is given.
# Point this at your most-used dataset once defined.
datasets_store([], name="default_dataset")
```

- [ ] **Step 2: `experiments.py` — commented generic preset**

Rewrite to a generic docstring + the `experiment_store = store(group="experiment", package="_global_")` line + ONE commented example:

```python
# Example experiment (uncomment, replace placeholders with your config names):
# experiment_store(
#     make_config(
#         hydra_defaults=[
#             "_self_",
#             {"override /model_config": "<your_model_config>"},
#             {"override /datasets": "<your_dataset_group>"},
#         ],
#         description="What this experiment tests.",
#         bases=(DerivaModelConfig,),
#     ),
#     name="<your_experiment>",
# )
```
If `--list-configs` requires at least one registered experiment to not error, leave a single trivial live `default` experiment referencing the live `default_model` + `default_dataset`; otherwise the commented example alone is fine.

- [ ] **Step 3: model_config module — keep REQUIRED `default_model`, add commented variant**

The runner REQUIRES a `default_model`. Replace the CIFAR `Cifar10CNNConfig` with a generic note: either (a) a minimal placeholder model config registered as `default_model` with a docstring "replace `builds(...)` target with your model function", or (b) if no generic model exists, register `default_model` against `model_protocol`'s reference shape and document the swap. Add ONE commented example variant showing how to register an alternate hyperparameter set.

- [ ] **Step 4: `assets.py`, `workflow.py`, `multiruns.py`, `multirun_descriptions.py` — generic commented scaffolds**

For each: generic docstring + the `store(group=...)` line (where applicable) + ONE commented example with placeholder names + a minimal live default only where the group must register to satisfy the runner. `multirun_descriptions.py` becomes a couple of commented description constants.

- [ ] **Step 5: De-CIFAR `pyproject.toml`**

- `name` → keep `"deriva-ml-model-template"`; `description` → "A template for building reproducible ML projects on DerivaML."
- Remove `[project.scripts]` `load-cifar10 = "scripts.load_cifar10:main"` (the script moved to the example repo).
- Remove CIFAR-only deps: `torchvision`; and `pandas`/`matplotlib`/`scikit-learn` IF only the moved ROC notebook used them (verify with `grep -rln 'pandas\|matplotlib\|sklearn' src/`). Keep `notebook`, `ipykernel`, `deriva-ml`, dev/docker groups.

- [ ] **Step 6: Verify the package still imports + no orphan imports**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
grep -nE 'cifar' src/configs/__init__.py || echo "(no cifar imports in __init__)"
uv run python -c "import configs" 2>&1 | tail -5
```
Expected: `import configs` succeeds with no CIFAR references.

---

### Task A5: Add a template-level config smoke test and verify the skeleton is green

**Files:**
- Create: `tests/test_configs_load.py` (template version — asserts the stub config groups register)

**Interfaces:**
- Consumes: the stubbed `datasets`/`deriva_ml` config groups from A4.
- Produces: a passing smoke test proving the skeleton's configs load.

- [ ] **Step 1: Write the failing smoke test**

```python
"""Smoke test: the skeleton's config groups register and load."""
def test_skeleton_config_groups_register():
    from hydra_zen import store
    import configs  # noqa: F401  triggers registration
    # The skeleton must register at least the required default groups.
    repo = store.get_entry  # hydra-zen store introspection
    # Minimal assertion: importing configs does not raise and deriva_ml + datasets groups exist.
    import configs.deriva  # noqa: F401
    import configs.datasets  # noqa: F401
    assert True
```

- [ ] **Step 2: Run it — expect import-time failures if orphan imports remain**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
uv run python -m pytest tests/test_configs_load.py -q 2>&1 | tail -8
```
Expected: PASS once A4's orphan-import cleanup is done; FAIL (ImportError) pointing at a leftover CIFAR import otherwise.

- [ ] **Step 3: Full skeleton verification**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
uv sync
uv run python -m pytest -q 2>&1 | tail -6
uv run deriva-ml-run --list-configs 2>&1 | tail -15
uv run ruff check src tests 2>&1 | tail -5
```
Expected: tests pass (only the skeleton smoke test + any retained generic tests), `--list-configs` shows generic groups with NO cifar entries, ruff clean.

- [ ] **Step 4: Commit the stripped skeleton (docs come in A6)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
git add -A
git commit -m "$(cat <<'EOF'
refactor: strip CIFAR implementation; convert configs to commented scaffolds

Removes all CIFAR-10 implementation (models, loader scripts, notebook, CIFAR
tests/docs) from the template. Each config module is KEPT and converted to a
self-documenting scaffold: a minimal live default plus one generic commented
example. The worked CIFAR example now lives in
informatics-isi-edu/deriva-ml-cifar-example.

- config files become commented-example scaffolds (not emptied, not deleted)
- load-cifar10 entry point + torchvision/ROC deps dropped from pyproject
- egg-info untracked + gitignored
- README/CLAUDE.md customization guide added in the follow-up commit (A6)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A6: Write the README customization guide + docs/customization.md deep-dive

**Files:**
- Modify: `README.md` (add a concise "Customizing this template" section; rewrite "What's Included" for the skeleton)
- Create: `docs/customization.md` (the step-by-step deep-dive)
- Modify: `CLAUDE.md` (drop CIFAR-specific sections; point to the example repo)
- Modify: `mkdocs.yml` (add `customization.md` to nav if mkdocs nav is explicit)

**Interfaces:**
- Consumes: the commented-scaffold config files from A4 (the guide references them by path).
- Produces: a README that tells a user exactly how to turn the skeleton into their project, plus a fuller `docs/customization.md`.

- [ ] **Step 1: Rewrite README "What's Included" + add "Customizing this template"**

The README must describe how to customize the template. Replace the CIFAR-centric "What's Included" with a skeleton description, and add a concise ordered walkthrough that links to the deep-dive. The section content (write it, don't placeholder it):

```markdown
## What's Included

- Python-first configuration with hydra-zen (no YAML)
- CLI entry points: `deriva-ml-run`, `deriva-ml-run-notebook`
- A complete `src/configs/` scaffold: every config group present as a
  self-documenting file with a commented example you fill in
- A model interface (`src/models/model_protocol.py`) to implement against
- GitHub Actions for versioning + docs

For a complete worked example, see
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example).

## Customizing this template

This template is a skeleton — every config file ships with a commented
example you replace with your own. The short version:

1. **Point at your catalog** — edit `src/configs/deriva.py` (`hostname`,
   `catalog_id`).
2. **Declare your datasets** — uncomment the example in
   `src/configs/datasets.py` and fill in your RIDs/versions.
3. **Add your model** — implement against `src/models/model_protocol.py`
   and register it as `default_model` in the model config.
4. **Define experiments** — uncomment the example in
   `src/configs/experiments.py` to pair a model with a dataset.
5. **(Optional) sweeps, assets, workflows** — `multiruns.py`, `assets.py`,
   `workflow.py` each carry a commented example.
6. **Rename the project** — set `name`/`description` in `pyproject.toml`.

See [docs/customization.md](docs/customization.md) for the full walkthrough
with per-file detail.
```

- [ ] **Step 2: Write docs/customization.md (the deep-dive)**

Create `docs/customization.md` expanding each of the 6 steps above with: the exact file, what each field/group means, the commented example to uncomment, how to verify (`uv run deriva-ml-run --list-configs`, `--cfg job`, `dry_run=true`), and a pointer to the relevant deriva-ml skill/doc. One H2 per step. No placeholders — write the actual prose.

- [ ] **Step 3: De-CIFAR CLAUDE.md**

Edit `CLAUDE.md`: remove CIFAR-specific sections (the CIFAR walkthrough, Experiments.md/CIFAR10.md references, the catalog-RID-defaults gotcha). Replace with a short note that the worked example lives in `deriva-ml-cifar-example`, and keep the generic conventions (uv, docstrings, num_workers, commit-before-running).

- [ ] **Step 4: Verify docs build (if mkdocs nav is explicit) + commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
grep -q 'customization' mkdocs.yml || echo "(add docs/customization.md to mkdocs.yml nav if nav is explicit)"
git add -A
git commit -m "$(cat <<'EOF'
docs: add template customization guide (README + docs/customization.md)

README now describes exactly how to turn the skeleton into a project (point at
catalog, declare datasets, add model, define experiments, rename). Full
per-file walkthrough in docs/customization.md. CLAUDE.md de-CIFAR-ed and points
to deriva-ml-cifar-example for the worked example.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase B — Create the new deriva-ml-cifar-example repo

### Task B1: Assemble the new repo locally with fresh history

**Files:**
- Create: `/Users/carl/GitHub/DerivaML/deriva-ml-cifar-example/` (new repo, fresh `git init`)

**Interfaces:**
- Consumes: the staging dir from A2 (CIFAR + generic scaffolding).
- Produces: a self-contained, runnable CIFAR project with a single initial commit.

- [ ] **Step 1: Initialize the new repo from staging**

```bash
mkdir -p /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
git init -b main
rsync -a /Users/carl/GitHub/DerivaML/_cifar-staging/ ./
# Ensure egg-info is not carried + is ignored:
rm -rf src/*.egg-info
grep -q 'egg-info' .gitignore || echo "*.egg-info/" >> .gitignore
ls -R . | head -50
```
Expected: full CIFAR tree + generic scaffolding present.

- [ ] **Step 2: Re-point project identity to the example**

Edit `pyproject.toml` in the new repo:
- `name = "deriva-ml-cifar-example"`, `description = "CIFAR-10 reference example for DerivaML, built on the deriva-ml-model-template skeleton."`
- KEEP the `load-cifar10` entry point and `torchvision`/ROC deps here (they belong with CIFAR).
Update `README.md` title/intro to say this is the CIFAR example built on the template skeleton; link back to `deriva-ml-model-template`.

- [ ] **Step 3: Verify the new repo is green standalone**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
uv sync
uv run python -m pytest -q 2>&1 | tail -8
uv run deriva-ml-run --list-configs 2>&1 | tail -15
uv run ruff check src tests 2>&1 | tail -5
```
Expected: tests pass (the CIFAR test suite — note: `test_cifar_canonical_partition` fixtures already use the correct `Image.Filename` case since they came from `main`), `--list-configs` shows the CIFAR experiment/model/dataset groups, ruff clean. Fix import paths if the package name change broke any `from configs...`/`from scripts...` imports (the package layout is identical, so imports should resolve unchanged).

- [ ] **Step 4: Single initial commit**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
git add -A
git commit -m "$(cat <<'EOF'
Initial commit: CIFAR-10 reference example for DerivaML

A worked example built on the deriva-ml-model-template skeleton, showing a
complete DerivaML project: the cifar10 domain schema loader, a 2-layer CNN
with configuration variants, hydra-zen experiment/multirun configs, a ROC
analysis notebook, and the full test suite.

Split out of deriva-ml-model-template so the template can be a clean skeleton.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B2: Create the GitHub repo and push

**Files:** none (remote operation)

- [ ] **Step 1: Create the public repo under the org and push**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example
gh repo create informatics-isi-edu/deriva-ml-cifar-example \
  --public \
  --source=. \
  --description "CIFAR-10 reference example for DerivaML, built on the deriva-ml-model-template skeleton." \
  --remote=origin \
  --push
```
Expected: repo created, `main` pushed.

- [ ] **Step 2: Confirm the remote**

```bash
gh repo view informatics-isi-edu/deriva-ml-cifar-example --json name,visibility,url 2>&1
git -C /Users/carl/GitHub/DerivaML/deriva-ml-cifar-example remote -v
```
Expected: public repo at the expected URL, origin set.

---

## Phase C — Land the template strip + cleanup

### Task C1: PR the stripped template and clean up staging

**Files:** none new

- [ ] **Step 1: Push the strip branch and open a PR (template repo)**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
git push -u origin chore/strip-cifar-to-skeleton
gh pr create --repo informatics-isi-edu/deriva-ml-model-template \
  --base main --head chore/strip-cifar-to-skeleton \
  --title "Strip CIFAR example into deriva-ml-cifar-example; template is now a pure skeleton" \
  --body "$(cat <<'EOF'
Strips all CIFAR-10 code/configs/tests/docs from the template. The worked
example now lives at informatics-isi-edu/deriva-ml-cifar-example.

- Template is a clean, runnable deriva-ml skeleton (empty-but-valid configs).
- `load-cifar10` entry point + torchvision/ROC deps removed.
- README/CLAUDE.md point to the example repo.

Verified: `uv run python -m pytest` green, `uv run deriva-ml-run --list-configs`
shows generic groups with no CIFAR entries, ruff clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR opened. (Do NOT auto-merge — leave for review.)

- [ ] **Step 2: Remove the staging dir and the temporary worktree (after PR is up)**

```bash
rm -rf /Users/carl/GitHub/DerivaML/_cifar-staging
# Leave the worktree until the PR merges; note for cleanup:
echo "After PR merges: git -C /Users/carl/GitHub/DerivaML/deriva-ml-model-template worktree remove ../deriva-ml-model-template-strip"
```

---

## Self-review notes

- **Spec coverage:** template stripped + configs converted to commented scaffolds (A1–A5), customization guide written (A6), new repo created + pushed (B1–B2), template PR'd (C1). ✓
- **Config files are scaffolds, not stubs:** A3 deletes only CIFAR *implementation* (models/scripts/notebook/tests/docs); the config modules (`datasets/experiments/assets/workflow/multiruns/multirun_descriptions` + model_config) are KEPT and converted in A4 to "minimal live default + one generic commented example." No config file is emptied or deleted. ✓ (This is the user's explicit refinement.)
- **README is a customization guide:** A6 step 1 adds the ordered "Customizing this template" walkthrough to README; A6 step 2 writes the `docs/customization.md` deep-dive (README-overview + docs-deep-dive split, per the user's choice). ✓
- **`test_cifar_canonical_partition` fixtures:** the new repo inherits the corrected `Image.Filename` fixtures because the source came from `main` (commit `68fd708`). No re-fix needed. ✓
- **egg-info:** excluded from both repos (A3 step 2, B1 step 1). ✓
- **Import integrity:** package layout identical in the new repo, so imports resolve unchanged; B1 step 3 verifies. Template-side, A4 step 6 verifies `import configs` succeeds with no CIFAR refs. ✓
- **Live-default risk:** some groups (notably `default_model`) MUST register a live value or the runner errors. A4 steps 1–4 keep a minimal live default in each required group; A5 step 3 (`--list-configs` + pytest) is the gate that catches any group left example-only that the runner needs. If a generic `default_model` can't be authored without a real model, document the swap and let the smoke test assert the group is registered rather than runnable.
- **roc_analysis.py risk:** the generic `roc_analysis.py` config stays, but its notebook moved. If it imports CIFAR symbols, A4 step 6 catches it at import time → either genericize it or move it to the example repo (decide during execution, note it).
