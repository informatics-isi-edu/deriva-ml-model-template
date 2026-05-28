# Analyst Summary — 2026-05-28 e2e run

**Persona:** Analyst | **Catalog:** `localhost` id 27 (`e2e-test-20260528`) |
**Worktree branch:** `e2e-test/2026-05-28`

## How I ranked the runs

By **accuracy** on the full 550-image M1G test set: **YHP (41.1 %) >
XCE (37.8 %) > W76 (24.0 %)**. By **calibration**: XCE > YHP > W76.
The accuracy ranking and the calibration ranking disagree on the
top two, which is the most important thing this analysis surfaced.

## Top 3 things the analysis says

1. **YHP is accurate but confidently wrong about 1 image in 3** —
   171 of 550 test images get a confidence ≥ 0.8 prediction that's
   actually wrong (vs 23 for XCE, 1 for W76). The validation lane
   already exists in the training loop (Modeler [tk-006](../../tacit-knowledge.md#tk-006));
   wiring save-best-by-val-acc would address this. XCE is the
   better pick for any workflow where "the model is uncertain"
   should mean "the human should look."
2. **Cat is the bottleneck class for all three models.** W76 0.018,
   XCE 0.345, YHP 0.273 — and YHP's extra capacity actually
   *hurt* cat recall vs XCE. This matches the well-known CIFAR-10
   pattern, but it's load-bearing for this catalog: a future
   project that needs >50 % cat recall isn't going to get it by
   training longer on the Toronto pair; it needs more cats, more
   capacity, or better augmentation.
3. **YHP's biggest pairwise confusion is bird ↔ deer (21 mix-ups)**,
   which is not a domain-intuitive pair. The model has learned a
   "small subject with background detail" feature that fires on
   both. Cat ↔ dog and automobile ↔ truck are present too (the
   expected confusions), but bird ↔ deer is the surprise.

## Where the picture matches or fails domain intuition

**Matches:** automobile ↔ truck, airplane ↔ ship, and cat ↔ dog all
show up in the top confusions for both XCE and YHP, which is what a
human looking at 32×32 thumbnails would also confuse. Vehicles are
easier than animals overall; cat and bird are the hardest classes.

**Fails:** bird ↔ deer (above) is not how a domain expert would
predict a 41 %-accuracy model to fail. It's a real signal that the
model isn't using silhouette structure the way a human reader does.

## What the team got right, and what it missed

**Right:** Curator's dataset audit caught the train/test leakage
in `cifar10_labeled_split` *before* the Modeler picked a dataset,
so the Modeler chose the clean Toronto pair without rediscovering
the problem. Three-run hyperparameter spread (3 → 10 → 20 epochs)
gave clearly differentiated results.

**Missed:** the shipped `roc_analysis.ipynb` template uses a
`Confidence IS NULL` heuristic to pick the ground-truth execution,
and on this catalog that picks the partial 500-row attempt `854`
instead of the canonical 1100-row `HSR` — Curator [tk-001](../../tacit-knowledge.md#tk-001)
predicted the trap. The Analyst routed around it with
`scripts/build_joined_wide_table.py` (filters to HSR explicitly,
produces the full n=550 joined table); the one-line notebook fix is
in [`findings/analyst/02`](../../findings/analyst/02-roc-notebook-picks-wrong-gt-execution.md).

## Deliverables for the next reader

- **[`docs/reports/joined-wide-table.csv`](joined-wide-table.csv)** —
  550 rows × 38 cols, one row per test image, every number in the
  analysis re-derivable in two pandas lines.
- **[`docs/reports/2026-05-28-analysis.md`](2026-05-28-analysis.md)**
  — the full domain-readable report (ranking, confusion patterns,
  intuition checks, caveats).
- **Analysis execution [1012](https://localhost/id/27/1012@355-RZWY-9B9R)**
  — the joined table + summary + per-class recall + confusion-long
  CSVs are attached as Execution_Assets, so the catalog has the
  analysis with provenance to W96/XEE/YKP after the worktree goes
  away.
- **Notebook execution [ZW0](https://localhost/id/27/ZW0@355-RZWY-9B9R)**
  — ROC curves + confusion matrices on the n=250 subset (with the
  GT-heuristic caveat from finding 02); still useful as a visual
  cross-check.
- Two new tacit-knowledge entries
  ([tk-007](../../tacit-knowledge.md#tk-007),
  [tk-008](../../tacit-knowledge.md#tk-008)) capturing the
  interpretive judgments — overconfidence-vs-accuracy ranking and
  the bird ↔ deer surprise.
- Two analyst findings under `findings/analyst/`.
