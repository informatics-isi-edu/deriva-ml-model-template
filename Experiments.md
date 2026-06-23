# Experiments

Canonical registry of all defined experiments and multiruns. Keep this file
in sync with `src/configs/experiments.py` and `src/configs/multiruns.py`.

## Quick Reference

| Experiment | Model Config | Dataset | Description |
|------------|-------------|---------|-------------|
| _(your experiment)_ | _(model config group)_ | _(dataset group)_ | _(one-line summary)_ |

## Multiruns

| Multirun | Overrides | Description |
|----------|----------|-------------|
| _(your multirun)_ | _(`+experiment=...`, parameter ranges)_ | _(what it sweeps / compares)_ |

---

## Experiment Details

### `<your_experiment>`

- **Config group overrides**: `model_config=<your_model_config>`, `datasets=<your_dataset_group>`
- **Parameters**: _(epochs, architecture, batch size, learning rate, ...)_
- **Purpose**: _(what this experiment tests)_

---

## Adding New Experiments

1. Define the experiment in `src/configs/experiments.py`
2. Document it in this file
3. Test with `uv run deriva-ml-run +experiment=<name> dry_run=true`
4. Commit both the code and this file together
