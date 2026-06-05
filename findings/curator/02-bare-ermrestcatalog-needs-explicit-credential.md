# Bare `ErmrestCatalog(...)` needs an explicit credential to read catalog 69

**Persona:** Curator
**Phase:** Writing a read-only split-integrity verification script
(`scripts/curator_verify_splits.py`)

## What happened

To prove the split relationships with set arithmetic I wrote a small
read-only script using `deriva.core.ErmrestCatalog` + `getPathBuilder()`
(the same low-level API the template's existing
`scripts/test_bag_fk_traversal.py` reaches for). First attempt:

```python
from deriva.core import ErmrestCatalog
catalog = ErmrestCatalog("https", "localhost", "69")
catalog.getPathBuilder().schemas["e2e-test-20260605"] \
    .tables["Dataset_Image"].filter(...).attributes(...)
```

failed with:

```
deriva.core.datapath.DataPathException:
401 Client Error: Unauthorized for url:
  https://localhost/ermrest/catalog/69/attribute/.../Dataset_Image/Dataset=H8M/Image
Details: Access requires authentication.
  Detail: select access on :e2e-test-20260605:Dataset_Image:Dataset
```

Catalog 69 requires authentication even for **select** on the domain
schema. The bare `ErmrestCatalog(...)` constructor does not pick up any
ambient credential, so every read 401s.

## Resolution (the fix that worked)

Pass the deriva credential explicitly:

```python
from deriva.core import ErmrestCatalog, get_credential
credential = get_credential("localhost")
catalog = ErmrestCatalog("https", "localhost", "69", credentials=credential)
```

`get_credential(hostname)` reads the token deposited by
`deriva-globus-auth-utils`/`mcp-localhost` auth, and the same reads then
succeed. The script now runs clean and all 19 split-integrity checks
pass.

## Reproduction

1. Be authenticated to localhost (MCP server works, so a token exists on
   disk).
2. `ErmrestCatalog("https","localhost","69").getPathBuilder()...` any
   read on the `e2e-test-20260605` schema -> 401.
3. Add `credentials=get_credential("localhost")` -> reads succeed.

## Notes

- This is a **friction / discoverability** observation, not necessarily
  a platform defect — the credential is available; the bare constructor
  just doesn't auto-load it. The high-level `DerivaML(...)` client *does*
  load credentials automatically, so a script written against the
  deriva-ml domain API would not have hit this. The trap is specifically
  using the low-level `deriva-py` `ErmrestCatalog` directly (which the
  inheritance rule says to avoid anyway — prefer the deriva-ml surface).
- The existing template script `scripts/test_bag_fk_traversal.py` points
  at a *public* eye-ai catalog, so it never needed the credential and
  doesn't model the authenticated-localhost case. A future template
  example that reads a private localhost catalog via `ErmrestCatalog`
  should show the `get_credential` line, or use `DerivaML(...)` instead.
- Routed around it (added the credential) rather than fixing the
  platform, per the no-fix-mid-arc rule.
