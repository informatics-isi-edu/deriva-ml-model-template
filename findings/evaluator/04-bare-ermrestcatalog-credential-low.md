# Bare `ErmrestCatalog(...)` needs an explicit credential — confirmed Low, friction not defect

**Severity:** Low
**Category:** Doc gap / Polish
**Component:** `deriva-py` low-level API used directly from a template script
**Persona finding this consolidates:** `findings/curator/02`

## What the evaluator found

The evaluator independently reproduced the Curator's experience: every
direct-Python verification in this evaluation used
`ErmrestCatalog("https","localhost","69", credentials=get_credential("localhost"))`,
and that worked cleanly. A bare `ErmrestCatalog("https","localhost","69")`
401s on `select` against the private `e2e-test-20260605` schema because the
constructor does not auto-load the on-disk credential.

The evaluator confirms the Curator's own classification: **this is
friction/discoverability, not a platform defect.** The high-level
`DerivaML(...)` client loads credentials automatically; the trap is
specifically reaching for the low-level `deriva-py` `ErmrestCatalog`
directly — which the inheritance/steering rule says to avoid anyway in
favour of the deriva-ml surface.

## Severity rationale

Low: a one-line fix (`credentials=get_credential(host)`), correct platform
behavior (auth is required on a private catalog), and the recommended path
(`DerivaML(...)`) never hits it. Worth a doc note, not a code change.

## Suggested action

Defer or dismiss. If anything: a future template example that reads a
private localhost catalog via the low-level API should model the
`get_credential` line (the existing `scripts/test_bag_fk_traversal.py`
points at a public eye-ai catalog and so never needed it). No platform
change warranted.
