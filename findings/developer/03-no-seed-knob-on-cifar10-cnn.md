# `cifar10_cnn` runner has no `seed` parameter — variance-across-seeds story is blocked

**Persona:** Developer
**Phase:** Training arc — wanted to run the same config twice with seed=42 and seed=123 to estimate run-to-run variance, 2026-05-26
**Severity:** Low (already tracked as pending task **D02**; this finding records the workflow gap observed mid-arc)
**Component:** `deriva-ml-model-template/src/models/cifar10_cnn.py` (the `cifar10_cnn()` function signature)

## What happened

The Curator's tk-003 explicitly suggests using both `C86` (seed=42 split)
and `CS0` (seed=123 split) for "variance estimation across seeds." But
"variance across seeds" requires varying the *training* random seed
(PyTorch's RNG state for weight init, batch order, etc.), not just varying
the *dataset partition* seed (which is what C86 vs CS0 differ by).

Inspecting `src/models/cifar10_cnn.py`, the `cifar10_cnn()` function takes:

```python
def cifar10_cnn(
    conv1_channels=32, conv2_channels=64, hidden_size=128, dropout_rate=0.0,
    learning_rate=1e-3, epochs=10, batch_size=64, weight_decay=0.0,
    test_only=False, weights_filename="cifar10_cnn_weights.pt",
    ml_instance=None, execution=None,
) -> None:
```

No `seed` parameter, no `torch.manual_seed(...)` call inside, no
`Cifar10CNNConfig` seed field in `src/configs/cifar10_cnn.py`. Each run
uses whatever PyTorch's default global RNG state happens to be. The
two consecutive runs of `cifar10_quick` (DYC, then re-running) would
*not* produce the same weights — and there's no way to ask for a
reproducible seed via Hydra override.

The Curator's intent (using seed=42 and seed=123 split for variance) is
*almost* the right thing, but it conflates two distinct sources of
randomness: (a) which images get into the train pool, and (b) the model's
training stochasticity. To get a true seed-variance estimate, you'd want
to fix the dataset partition (one of C86 or CS0) and vary the *training*
seed.

## Reproduction

```
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run +experiment=cifar10_quick model_config.seed=42
```

Result: Hydra error — `model_config.seed` is not a known field.

```
grep -n "seed" src/models/cifar10_cnn.py
```

Result: zero hits. No `torch.manual_seed`, no `np.random.seed`, no
seed knob.

## Impact on the persona's work

Routed around. The Developer ran `lr_sweep` to give the Analyst a sweep
to compare and skipped the seed-variance story for this arc. The
multipersona test plan §2.2 calls for "two distinct training runs" and
"at least one multirun" — both met without needing seed variance, so the
arc's success criteria are unblocked.

But the *reason* the Curator picked two splits (C86 + CS0) — to enable
variance-across-seeds work — is unrealizable until D02 lands. The current
"variance" interpretation collapses to "what happens when you train on
two slightly-different 161/200-overlapping subsets" which is a less
clean variance signal than "what happens when you train the same data
twice with different RNG seeds."

## Suggested classification

Missing feature (pending task D02 already tracks it). This finding is
the mid-arc observation that confirms the gap is real.

## Notes for the fix-pass

- Add a `seed: int | None = None` parameter to `cifar10_cnn()` (line
  320-337 of `src/models/cifar10_cnn.py`).
- Inside, if `seed is not None`: `torch.manual_seed(seed)` and
  `np.random.seed(seed)` (also `torch.use_deterministic_algorithms(True)`
  if reproducibility is the goal — note that the `DataLoader` with
  `num_workers=0` and `shuffle=True` will pull from the seeded
  generator).
- Add `seed=None` to `Cifar10CNNConfig` in `src/configs/cifar10_cnn.py`
  so `model_config.seed=42` works as a Hydra override.
- Record the resolved seed in the training_log.txt and the saved
  `checkpoint["config"]` dict (it's a load-bearing piece of provenance
  for any reproducibility question).
- Optional: have the executor pull a seed from the execution RID hash if
  no explicit seed is given, so that even unconfigured runs are
  deterministic-given-the-execution-RID. That would close the gap
  *automatically* for any future arc.
