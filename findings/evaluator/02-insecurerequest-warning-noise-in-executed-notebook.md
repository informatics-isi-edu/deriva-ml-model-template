# evaluator/02 — InsecureRequestWarning floods the executed notebook output (1152 lines)

**Severity:** Low
**Category:** Polish
**Component:** `deriva-py` HTTPS calls / notebook runner output capture
**Filed by:** Evaluator (2026-05-27d run)

## Summary

The Modeler flagged this as cosmetic in tk-003 ("Default Hydra mode
emits a screen of `InsecureRequestWarning`s from every HTTPS call to
the self-signed localhost cert"). The Analyst hit the same thing — and
when papermill captured the executed notebook back into the catalog
(asset YCW, `roc_analysis.md`, 910 KB), the rendered markdown contains
**1152 lines of `InsecureRequestWarning` chatter** interleaved with the
actual analysis output.

```
$ grep -c "InsecureRequestWarning" analysis-scratch/y90_outputs/roc_analysis.md
1152
```

For comparison, the entire executed notebook is 2300 lines. Roughly
**50% of the catalog-stored markdown export is HTTPS-warning noise**.

## Why this is a finding (and not just "ignore it")

The `roc_analysis.md` is a primary deliverable — the analyst's
report cites it as the catalog-side artifact a downstream reader would
open to see the analysis. A reader who opens YCW today is going to scroll
through pages of identical warnings to find the actual content. This is
the kind of polish issue that erodes confidence in the catalog as a
durable archive even when the underlying machinery is working
correctly.

This is a recurring observation across multiple e2e runs — the prior
2026-05-27c Modeler arc's training_log assets have the same shape, and
the 2026-05-27 (catalog 93) run flagged it too. It hasn't been filed
as its own thread because it's always been treated as a known cosmetic
issue, but having both personas independently surface it in a single
run is the signal to escalate it from "known" to "fix it."

## Two reasonable fixes

1. **At the deriva-py side:** suppress `InsecureRequestWarning` by
   default when `verify=False` is set deliberately (the localhost
   self-signed dev case). This is the cleanest fix — the warning was
   never news, the user opted in.
2. **At the notebook runner / model-template side:** `warnings.filterwarnings("ignore",
   category=urllib3.exceptions.InsecureRequestWarning)` in the
   `run_notebook()` setup path. Narrower scope but addresses the
   catalog-stored markdown export issue specifically.

(1) is the right fix; (2) is the fast fix. Either would land the
executed-notebook export at ~half its current size and let a downstream
reader actually read it.

## Suggested disposition

- **Fix inline** at the `deriva-py` level (option 1). Trivial change,
  one filter call gated on `verify=False`.
- Or **GitHub issue** if it needs design discussion (e.g. whether
  there's a contour where the warning still matters). The Modeler's
  tk-003 already documents the workaround for impatient users.
