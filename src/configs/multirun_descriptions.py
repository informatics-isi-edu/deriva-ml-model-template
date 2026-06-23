"""Multirun descriptions.

Rich markdown descriptions for the *parent* execution recorded when you run a
multirun sweep (see ``configs/multiruns.py``). Keeping the prose here, as plain
Python string constants, lets a sweep document why its runs are being compared
and what outcomes are expected — that text lands on the parent execution in the
catalog.

Each constant defined here is imported by ``multiruns.py`` and passed as the
``description=`` of a ``multirun_config(...)`` call.

Usage:
    # Run a registered multirun (its parent execution gets the description)
    uv run deriva-ml-run +multirun=<name>
"""

# Description attached to the example sweep registered in ``multiruns.py``.
# Replace this with prose describing your own sweep.
EXAMPLE_SWEEP_DESCRIPTION = """## Example Sweep

**Objective:** Describe what this sweep compares and why.

### Runs

| Run | What differs |
|-----|--------------|
| run A | <e.g. lower learning rate> |
| run B | <e.g. higher learning rate> |

### Expected Outcomes

- Summarize what you expect to learn from the comparison.
- Note any caveats (small dataset, short training, etc.).
"""

# -----------------------------------------------------------------------------
# Add a description constant per sweep. Example:
# -----------------------------------------------------------------------------
#
#   MY_SWEEP_DESCRIPTION = \"\"\"## <Sweep Title>
#
#   **Objective:** <what this sweep is for>
#
#   ### Parameter Grid
#
#   | Parameter | Values |
#   |-----------|--------|
#   | <param>   | <a, b, c> |
#   \"\"\"
