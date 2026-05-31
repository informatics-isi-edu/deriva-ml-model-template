# `deriva-ml-run` positional-arg guard shadows Hydra's `--info <group>` and `--cfg`

**Persona:** Evaluator (upholding `findings/modeler/01`)
**Severity:** Medium
**Category:** Polish (DX / CLI ergonomics)

## What happened

The Modeler (findings/modeler/01) reported, and I confirm as legitimate,
that two standard Hydra config-inspection moves are unreachable through
the `deriva-ml-run` wrapper:

1. `uv run deriva-ml-run --info experiment` — the positional-argument
   guard treats the bare word `experiment` as a mis-typed Hydra override
   and errors with "looks like a positional argument," before Hydra's
   own `--info <thing>` parsing runs. Hydra natively accepts an optional
   argument to `--info` (`config`, `defaults`, `groups`, or a group
   name); that form is shadowed.
2. `uv run deriva-ml-run +experiment=cifar10_quick --cfg job` — the
   wrapper's argparse layer rejects `--cfg` ("unrecognized arguments"),
   so the resolved-config dump is unavailable.

The "Did you mean" suggestions assume a *selection* intent
(`assets=experiment`, `+experiment=experiment`) when the user has an
*inspection* intent, which compounds the confusion.

## Why this is Medium / Polish, not higher

- A workaround exists and is documented in the Modeler finding:
  `--info` with **no** argument lists every group + options;
  `+experiment=<name> dry_run=true` resolves the full config (and
  additionally validates RIDs against the live catalog, so it is
  arguably a *better* preflight — just much heavier, since it downloads
  the dataset bag).
- It cost the Modeler "a couple of confused attempts and a re-read of the
  help text," not a blocked deliverable. Nothing in the run was
  compromised by it.

The positional-arg guard itself is a good feature (it catches the common
`deriva-ml-run cifar10_quick` typo). The rough edge is purely that it
fires too early to let the legitimate `--info <group>` / `--cfg` forms
through, and that the suggestion text doesn't point at the working
inspection paths.

## Reproduction

Against this worktree (catalog 168 wired in):

```
uv run deriva-ml-run --info experiment                      # -> positional-arg error
uv run deriva-ml-run +experiment=cifar10_quick --cfg job    # -> unrecognized --cfg
uv run deriva-ml-run --info                                 # -> works (lists all groups)
uv run deriva-ml-run +experiment=cifar10_quick dry_run=true # -> resolves + validates, stops before training
```

## Suggested disposition

GitHub issue against the model-template `deriva-ml-run` wrapper. Either
(a) pass `--info <group>` and `--cfg` through to Hydra after the guard,
or (b) cheaper: when the guard fires on a word that is a known
config-group name, add a suggestion line pointing at `--info` (no-arg)
and `dry_run=true`.
