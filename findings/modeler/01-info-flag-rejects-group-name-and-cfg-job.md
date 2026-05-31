# `deriva-ml-run --info <group>` and `--cfg job` don't work as a Hydra user expects

**Persona:** Modeler
**Phase:** Pre-flight config inspection (listing experiments, dumping a resolved job config)

## What happened

While orienting before the first run, I wanted to (a) list just the
`experiment` config group and (b) dump the fully-resolved job config for
one experiment — both standard Hydra inspection moves.

**Attempted #1:** `uv run deriva-ml-run --info experiment`

**Expected:** Hydra's `--info experiment` behavior — print the
`experiment` config group's options (Hydra's native `--info` accepts an
optional argument naming what to show: `config`, `defaults`, `groups`, or
a specific group name).

**Actual:** the CLI wrapper treats the bare word `experiment` as a
positional Hydra override and errors:

```
Error: 'experiment' looks like a positional argument, but deriva-ml-run
expects Hydra overrides in key=value form.
Did you mean:
  deriva-ml-run ... assets=experiment
or
  deriva-ml-run ... +experiment=experiment
?
```

Both suggestions are wrong for the intent (I wasn't trying to *select* an
experiment named "experiment"; I was trying to *inspect* the experiment
group). The guard fires before `--info`'s own argument parsing, so the
Hydra-native `--info <group>` form is unreachable.

**Attempted #2:** `uv run deriva-ml-run +experiment=cifar10_quick --cfg job`

**Expected:** Hydra's `--cfg job` — print the resolved job config as YAML
without running.

**Actual:** `deriva-ml-run: error: unrecognized arguments: --cfg job`.
The wrapper's argparse layer doesn't pass `--cfg` through to Hydra.

## Workaround (what actually works)

- To list groups/options: `uv run deriva-ml-run --info` with **no
  argument** prints every group and its options in one block (this works
  fine — it's the *argument* to `--info` that's rejected).
- To inspect a resolved experiment without running: `uv run deriva-ml-run
  +experiment=<name> dry_run=true` runs the full config resolution +
  bag-download path and stops at `Dry run mode: skipping model execution`.
  It validates RIDs/versions against the live catalog, which `--cfg job`
  would not — so it's arguably a *better* pre-flight gate, just much
  heavier (it downloads the dataset bag) and not a pure config dump.

Neither workaround is a blocker; both cost a couple of confused attempts
and a re-read of the help text.

## Reproduction

Against this worktree (catalog 168 wired in):

```
uv run deriva-ml-run --info experiment          # -> positional-arg error
uv run deriva-ml-run +experiment=cifar10_quick --cfg job   # -> unrecognized --cfg
uv run deriva-ml-run --info                     # -> works (lists all groups)
```

## Notes

- The positional-argument guard is a good feature in general (it catches
  the common `deriva-ml-run cifar10_quick` typo). The rough edge is that
  it shadows Hydra's legitimate `--info <thing>` argument and that the
  "Did you mean" suggestions assume a *selection* intent when the user may
  have an *inspection* intent.
- I did not touch the CLI wrapper (out of scope for this arc — route
  around, don't fix). Recorded here so the evaluator can decide whether
  the `--info <group>` / `--cfg` passthrough is worth restoring or whether
  the help text / error suggestions should mention the
  `--info` (no-arg) and `dry_run=true` paths.
```