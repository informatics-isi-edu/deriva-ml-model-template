# Finding: free-text `description=` override rejected by Hydra grammar (parens/commas)

- **Persona:** Modeler
- **Date:** 2026-06-01
- **Catalog:** localhost / catalog 2 / schema `e2e-test-20260601`
- **Severity:** Low (workflow friction; clean workaround exists; no data impact)
- **Category:** CLI / Hydra override-grammar ergonomics

## What I expected

Running an ad-hoc training variation without an `+experiment=` preset, I wanted
to give the execution a human-readable name so it would be scannable in
`ml.list_executions()` / the Analyst's handoff. I passed a descriptive string as
a Hydra override:

```bash
uv run deriva-ml-run model_config=cifar10_regularized \
  datasets=cifar10_small_labeled_split \
  description="Modeler e2e Run A: regularized (20ep, dropout 0.25, wd 1e-4) on small labeled family PJM"
```

I expected the string to be stored verbatim as the `Execution.description`.

## What actually happened

The run failed **before any catalog write or model execution** with:

```
mismatched input ' (' expecting <EOF>
See https://hydra.cc/docs/1.2/advanced/override_grammar/basic for details
```

`deriva-ml-run` forwards positional `key=value` args to Hydra's override parser,
which treats `(`, `)`, and `,` as grammar metacharacters even inside an override
*value*. A free-text description containing those characters is therefore a
**parse error**, not a runtime error — the process never starts. (Shell quoting
doesn't help: the quotes are consumed by the shell, and Hydra still sees the raw
parens/commas in the value.)

## Repro

```bash
cd <worktree>
uv run deriva-ml-run model_config=cifar10_quick datasets=cifar10_small_labeled_split \
  'description=test (with parens, and commas)'   # -> mismatched input ' (' expecting <EOF>
```

A description free of `(`, `)`, `,` parses fine:

```bash
uv run deriva-ml-run model_config=cifar10_quick datasets=cifar10_small_labeled_split \
  'description=test without grammar metacharacters'   # -> OK
```

## Impact

- A common, reasonable instinct — annotate an ad-hoc run with a readable
  description — fails with a grammar error that does not mention `description` or
  point at the offending characters. The Hydra link is generic; a user has to
  know that parens/commas are the trigger.
- Only affects ad-hoc `model_config=` / `datasets=` runs. `+experiment=<name>`
  presets are unaffected because deriva-ml auto-composes the description from the
  preset text + resolved overrides (verified: QK8's description came out as
  `"Quick CIFAR-10 training: ... [overrides: +experiment=cifar10_quick]"`).

## Workaround applied

Sanitized the description to drop grammar metacharacters and ran successfully
(executions QWA, R5C). For runs that need a rich description, the idiomatic path
is to define a one-line experiment preset rather than pass `description=` on the
CLI.

## Suggested direction (NOT done — out of scope for this arc)

Either (a) document in the `deriva-ml-run` CLI reference that free-text
overrides must avoid Hydra grammar metacharacters (and point users at experiment
presets for rich descriptions), or (b) have `deriva-ml-run` expose a dedicated
non-Hydra flag for the execution description (e.g. `--description`) that bypasses
the override grammar. This is a known Hydra-passthrough behavior — the template's
own CLAUDE.md already notes a related "Hydra-flag-passthrough" caveat — so option
(a) is the lower-risk fix.
