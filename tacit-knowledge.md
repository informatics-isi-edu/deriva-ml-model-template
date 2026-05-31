# Tacit Knowledge

This file records **tacit knowledge** — the *why*, the *intent*, and the
*background* behind decisions made about this project's models and data.

The **catalog** is the source of record for everything else: data contents,
RIDs, dataset versions, workflow URLs and checksums, executions, lineage.
Don't replicate catalog-stored facts here. Don't ask this file what's in
the catalog — query the catalog directly (resources first, tools next).
When this file *needs* to reference a catalog entity, link to it
(`deriva://catalog/{host}/{cat}/ml/...`) instead of inlining its contents.

Each entry captures a decision: what was chosen, what alternatives were
considered, what was rejected and why, and any background context a future
reader would need to evaluate whether the decision still holds.

---

<a id="tk-001"></a>
### tk-001 — Curator substrate audit: all splits perfectly class-balanced, no within-pair leakage ([dataset F28](https://localhost/id/168/F28))
**When:** 2026-05-30T00:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

Audited the freshly-bootstrapped CIFAR-10 substrate (catalog 168) before
the Modeler/Analyst arcs. Question: do the 13 datasets actually represent
what their names and types imply, and is the ground truth sound? Joined
every split's Image members to the `Image_Classification` ground-truth
feature ([feature on Image](https://localhost/id/168/F28)) and tabulated
per-class counts.

Findings (all confirmed, read-only):
- **Ground truth is complete and unique.** 1100 GT rows, 1100 distinct
  images, exactly one label per image, no missing labels.
- **Every split is *exactly* class-balanced** — min == max per class in
  all nine member-bearing datasets: complete 110/class, canonical
  train/test ([F2T](https://localhost/id/168/F2T) /
  [F34](https://localhost/id/168/F34)) 55/class, small split 50/class,
  labeled split ([NE8](https://localhost/id/168/NE8) /
  [NEJ](https://localhost/id/168/NEJ)) 44 & 11/class, small-labeled
  ([PHT](https://localhost/id/168/PHT) /
  [PJ4](https://localhost/id/168/PJ4)) 40 & 10/class. The canonical
  F2T/F34 split is balanced even though its description doesn't claim
  stratification.
- **No leakage *within* any train/test pair**: F2T∩F34, F3W∩F46,
  NE8∩NEJ, PHT∩PJ4 are all 0 shared images.

Implications for collaborators (Modeler/Analyst): the substrate needs no
balance correction. Any of the pairs is fair to train+evaluate on *as a
pair*. The cross-pair traps are recorded separately in
[tk-002](#tk-002) — read that before mixing datasets from different
families.

**Weighed alternatives:** *(none — this was characterization, not a
choice. Considered creating a balanced curated subset and concluded none
was needed: the substrate is already uniform.)*

<a id="tk-002"></a>
### tk-002 — Trap: the "labeled_testing" splits are carved from training, not held out ([dataset NEJ](https://localhost/id/168/NEJ))
**When:** 2026-05-30T00:05:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-001](#tk-001) (the balance/leakage audit this trap was found during)

The labeled-split family is 100% drawn from the canonical *training*
partition [F2T](https://localhost/id/168/F2T), not from the held-out test
partition [F34](https://localhost/id/168/F34). Verified by set
intersection: every member of NE8, [NEJ](https://localhost/id/168/NEJ),
PHT, [PJ4](https://localhost/id/168/PJ4) is in F2T and *none* is in F34.
NE8∪NEJ exactly reconstructs F2T's 550 images; PHT∪PJ4 is a 500-image
subset of F2T.

The trap is a signal mismatch. The catalog `Dataset_Type` on NEJ/PJ4 is
`Testing`+`Labeled`, and their descriptions read "Testing subset" — both
of which a consumer dispatching on `Dataset_Type` (the authoritative
"what is this for" signal) reads as *held-out evaluation set*. But these
are **internal validation splits of the training pool** — useful only
when paired with their sibling training set (`cifar10_labeled_training`
→ NE8, `cifar10_small_labeled_training` → PHT). The `src/configs/datasets.py`
header comments (lines ~70–78) already describe this family correctly as
"cross-validation workflows ... where the test_batch must stay unseen for
final evaluation," so the *config author's* intent matches reality — it's
the catalog-side `Dataset_Type`/description that overstate these as test
sets.

Concrete leakage facts (verified):
- Train F2T → eval NEJ: **110/110 NEJ images are in F2T** — total leakage.
- Train F2T → eval PJ4: **100/100 PJ4 images are in F2T** — total leakage.
- Train NE8 → eval F34: 0 shared — SAFE.
- Train PHT → eval F34: 0 shared — SAFE.
- Train F2T → eval F34: 0 shared — SAFE.

Implications for collaborators:
- For a **held-out** evaluation number, evaluate against
  [F34](https://localhost/id/168/F34) (`cifar10_testing`), regardless of
  which training set was used. Do NOT report NEJ/PJ4 accuracy as a
  held-out metric for an F2T-trained model.
- NEJ/PJ4 are the right choice only as the *validation* half of their own
  sibling pair (hyperparameter selection, early stopping), never as the
  final test set for a model trained on the full F2T.
- The small split ([F3W](https://localhost/id/168/F3W) /
  [F46](https://localhost/id/168/F46)) is an **independent** 1000-image
  draw from the full 1100 pool — F3W overlaps F2T (500 shared) and F46
  overlaps F34 (500 shared). So F3W/F46 and F2T/F34 are not
  interchangeable: training on F3W and evaluating on F34 would leak.
  Keep each family internally consistent.

**Weighed alternatives:** considered "fixing" the catalog by changing
NEJ/PJ4's `Dataset_Type` from `Testing` to `Validation` (the
`Validation` term exists in the `Dataset_Type` vocab). Did **not** do so
in this arc — a `Dataset_Type` change is a catalog mutation that flips the
dataset to a new dev version and would muddy the e2e provenance baseline;
it's also arguably a defensible naming if one reads "testing" as "the
held-out half of *this* split." Recorded the trap here instead so the
decision stays with the user. See finding
`findings/curator/01-labeled-split-is-training-derived.md`.

<a id="tk-003"></a>
### tk-003 — Split hierarchy: labeled splits are NOT registered as catalog children of F2T ([dataset F2T](https://localhost/id/168/F2T))
**When:** 2026-05-30T00:08:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-002](#tk-002) (established the F2T-derivation that the hierarchy fails to record)

The dataset nesting hierarchy records only two split roots:
[F2J](https://localhost/id/168/F2J) → {F2T, F34} and
[F3M](https://localhost/id/168/F3M) → {F3W, F46}. The two labeled-split
roots, [NE0](https://localhost/id/168/NE0) → {NE8, NEJ} and
[PHJ](https://localhost/id/168/PHJ) → {PHT, PJ4}, are **standalone** —
they have no parent and are not children of
[F2T](https://localhost/id/168/F2T), even though their members are 100%
derived from F2T (see [tk-002](#tk-002)). The only catalog-side record of
that derivation is the free-text dataset description ("...subset of F2T,
stratified by Image_Class.Name, seed=42"), which is advisory prose, not a
walkable lineage edge.

Implications for collaborators: a `deriva_ml_list_dataset_relations` or
`deriva_ml_get_lineage` walk from F2T will **not** surface NE8/NEJ/PHT/PJ4
as descendants — the provenance link lives in prose only. If you need to
know "what was derived from the training partition," you can't get it from
the hierarchy; you have to read descriptions or re-derive by set
intersection (as this audit did). This is a gap in the bootstrap loader's
split-registration, recorded in
`findings/curator/02-labeled-split-not-registered-as-child.md`.

<a id="tk-004"></a>
### tk-004 — Modeler chose the canonical Toronto F2T/F34 pair (cifar10_split) for the held-out comparison, not the labeled-split family ([dataset F2J](https://localhost/id/168/F2J))
**When:** 2026-05-30T22:55:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-002](#tk-002) (the leakage trap this choice avoids)

Decision: for the differentiated training runs whose held-out numbers
the Analyst will trust, train on the F2T Training partition and evaluate
on the F34 Testing partition, fed to the runner as the
[F2J](https://localhost/id/168/F2J) `Split` parent (config
`cifar10_split`), which the model harness flattens to its F2T + F34
children. Chose this family specifically *because* [tk-002](#tk-002)
showed the labeled-split "testing" partitions
([NEJ](https://localhost/id/168/NEJ) /
[PJ4](https://localhost/id/168/PJ4)) are carved from the F2T training
pool — training on F2T then "evaluating" on NEJ/PJ4 is total leakage. F2T
and [F34](https://localhost/id/168/F34) are drawn from disjoint Toronto
source batches (F2T ∩ F34 = 0, confirmed in [tk-001](#tk-001)), so the
harness's final-epoch predictions on the F34 `Testing` bag are a genuine
held-out metric.

Mechanism worth knowing (term-of-art for a domain reader): the CIFAR-10
model harness dispatches each input dataset to a *lane* by its
`Dataset_Type` — a bag typed `Training` trains the model; a bag typed
`Testing` is held out and only used to score per-epoch metrics and to
record final-epoch predictions; a `Split` parent is expanded to its
children first. So which dataset becomes "the held-out eval set" is
decided entirely by catalog `Dataset_Type`, not by any flag the Modeler
passes. This is why the leakage trap in [tk-002](#tk-002) matters
operationally: a consumer who fed NEJ as the `Testing` bag would get a
leaked number with no warning from the pipeline.

To make the clean pairing a first-class, re-runnable choice I added two
experiment presets — `cifar10_quick_toronto` and `cifar10_large_toronto`
(in `src/configs/experiments.py`) — both overriding `datasets` to
`cifar10_split`. The pre-existing `cifar10_quick` / `cifar10_extended`
presets point at the labeled-split family and are kept as-is, but a
header comment now flags them as the leaky-for-F2T-eval family.

**Weighed alternatives:** considered using `cifar10_small_split`
([F3M](https://localhost/id/168/F3M) → F3W/F46) for the headline
comparison too — rejected for the *comparison* runs because, although
F3W/F46 are internally disjoint, [tk-002](#tk-002) notes F3W overlaps F2T
and F46 overlaps F34, so mixing F3M-trained and F2J-trained models in one
analysis would cross dataset families. Used F3M only for the throwaway
smoke run (pipeline shakeout), never for a reported number.

<a id="tk-005"></a>
### tk-005 — Two differentiated held-out runs: capacity+duration lifts F34 accuracy from 27.6% to 37.6% (and overfits) ([execution SSE](https://localhost/id/168/SSE))
**When:** 2026-05-30T22:58:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-004](#tk-004) (the clean F2T/F34 pairing both runs used)

Hypothesis: does more model capacity + more training time produce a
measurably higher *held-out* F34 accuracy than a low-capacity baseline,
both trained on F2T with zero leakage? Ran two executions to find out:

- [RM8](https://localhost/id/168/RM8) — `cifar10_quick_toronto`
  (3 epochs, 32→64 ch, 128 hidden, batch 128). Final-epoch held-out F34
  accuracy **27.64%**.
- [SSE](https://localhost/id/168/SSE) — `cifar10_large_toronto`
  (20 epochs, 64→128 ch, 256 hidden, batch 64). Final-epoch held-out F34
  accuracy **37.64%**, peaking ~39.8% mid-training.

Both used seed=42 and trained on 550 F2T images, evaluated on 550 F34
images. The runs differentiate clearly — a ~10-point held-out gap — so
the pipeline reflects hyperparameter variation in its outputs rather than
producing identical results (one of the things this arc set out to
stress-test). The headline numbers are low in absolute terms because the
substrate is tiny (550 train images, ~55/class) — these are
pipeline-and-comparison validation numbers, **not** a model-capability
claim for CIFAR-10.

The large run also exhibits a textbook overfit a domain reader should
note: train accuracy climbs to 100% by epoch 16 while F34 test_loss rises
monotonically from 1.76 (epoch 6) to 3.40 (epoch 20) — the network is
memorizing 550 images. The per-epoch `training_log.txt` asset captures
the full curve, and the held-out F34 metric (not a leaky NEJ/PJ4 number)
is what makes the overfit visible. Confirms the platform surfaces
generalization signal correctly when the eval set is genuinely held out.

For the Analyst: compare RM8 vs SSE on **F34**. Both wrote 550
`Image_Classification` feature rows (queryable via
`deriva_ml_list_feature_values(table="Image",
feature_name="Image_Classification", execution_rids=["RM8"])` etc.) AND a
wide per-image prediction CSV with per-class probabilities. The two CSVs
are wired into `src/configs/assets.py` as the group
`roc_quick_vs_large_toronto` (RIDs
[RP6](https://localhost/id/168/RP6) for RM8,
[SVC](https://localhost/id/168/SVC) for SSE). Join either against the
loader-written ground-truth rows on F34 (see [tk-006](#tk-006)).

<a id="tk-006"></a>
### tk-006 — Convention — Image_Classification is dual-purpose: loader writes ground truth, training runs write predictions ([feature on Image](https://localhost/id/168/F28))
**When:** 2026-05-30T23:00:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-005](#tk-005) (the prediction rows that established the second purpose)

The `Image_Classification` feature on `Image` is written by two distinct
kinds of execution and the rows are **not** distinguishable by table
membership alone. The bootstrap loader execution writes the *ground
truth* — one labeled row per image, `Confidence` unset. Each training
execution (e.g. [RM8](https://localhost/id/168/RM8),
[SSE](https://localhost/id/168/SSE)) writes *predictions* — one row per
evaluated image, `Confidence` populated with the softmax max. After the
two runs in [tk-005](#tk-005), an F34 image therefore carries three
`Image_Classification` rows: one GT + one per training run.

Implications for collaborators (the Analyst especially): when reading
this feature, scope by the producing execution. To get *predictions* for
one model, filter `execution_rids=[<that run's RID>]`; to get *ground
truth*, filter to the loader execution's RID (or, equivalently, the
`Confidence IS NULL` rows). An unfiltered read returns GT + every
recorded prediction interleaved, which is almost never what an analysis
wants — and the `newest` selector is **not** a safe substitute for "ground
truth," since "newest" is whichever execution last wrote (a prediction),
not the loader's GT. This is why the prediction CSV assets
([RP6](https://localhost/id/168/RP6),
[SVC](https://localhost/id/168/SVC)) are handy as a parallel surface:
each CSV is already scoped to exactly one run's predictions, with a
`Source_Label` column recording the model state (e.g. `epoch_20`).

<a id="tk-007"></a>
### tk-007 — Analyst verdict: SSE beats RM8 on F34, but the win is non-uniform and SSE is overconfident ([execution TYR](https://localhost/id/168/TYR))
**When:** 2026-05-30T23:12:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-005](#tk-005) (the two runs being compared), [tk-006](#tk-006) (the GT-vs-prediction scoping the join relied on)

Closed the team's story by scoring [RM8](https://localhost/id/168/RM8)
and [SSE](https://localhost/id/168/SSE) on the held-out F34 set. The
headline confirms the Modeler's numbers (RM8 27.6%, SSE 37.6%), but the
domain reading is richer than "bigger model wins," and three judgments
are worth carrying forward:

- **The win is concentrated in the *hard* classes, not uniform.** SSE
  rescues exactly the classes RM8 abandoned to near-random — automobile
  13%→55%, deer 5%→38%, cat 4%→27% — which is where the whole 10-point
  gain lives. On the ROC/AUC measure SSE dominates at every operating
  point (micro-AUC 0.81 vs 0.73), so for *ranking* quality there is no
  ambiguity. But RM8 nominally "wins" frog (73% vs 40%), which is an
  artifact, not a strength (next point).
- **RM8's failure mode is mode collapse.** `[inferred from pattern]` It
  funnels uncertain inputs into two classes: 171/550 of its predictions
  are "frog" and 115 are "truck" (together 52%), while it predicts cat /
  deer / automobile ~10 times each. That is why it catches most real
  frogs — it calls everything froggish "frog" — so RM8's frog number must
  not be cherry-picked as evidence the small model is good at anything.
  The under-trained-low-capacity model getting "lucky" on one class this
  way is a textbook signature, hence the pattern marker.
- **SSE is confidently wrong a lot — its softmax is miscalibrated by the
  overfit.** Mean confidence 0.81 (vs RM8's honest-but-useless 0.26), yet
  among its >0.70-confidence predictions, 223 are wrong vs 170 right —
  i.e. *when SSE says it's sure, it's wrong more than half the time.*
  This is the same overfit [tk-005](#tk-005) saw as rising F34 test_loss;
  it surfaces here as overconfidence. Practical guidance for any
  downstream consumer: **trust SSE's ordering (ROC), do not trust its
  confidence magnitudes.** Its mistakes are at least sensible (animals↔
  animals: bird→deer, dog→deer; vehicles↔vehicles: ship→truck,
  automobile→truck), which is a quality signal beyond accuracy.

All numbers are pipeline-validation numbers on a deliberately tiny
substrate (550 train / 55-per-class test), not a CIFAR-10 capability
claim — see [tk-005](#tk-005). Full writeup, the re-derivable wide table,
and the figures are in `docs/reports/2026-05-30-analysis.md`; the
analysis is captured with provenance as execution
[TYR](https://localhost/id/168/TYR).

**Weighed alternatives:** ranked the runs by top-1 accuracy *and* by AUC
rather than accuracy alone, specifically because the per-class table
showed accuracy and ranking-quality could in principle disagree; here
they agreed (SSE wins both), but the AUC view is what makes the
"dominates at every operating point" claim, and the per-class view is
what exposes the frog artifact that the single accuracy number hides.

<a id="tk-008"></a>
### tk-008 — Convention — the ROC notebook now emits a joined wide table; runner's execution description shows the static config text, not the override ([execution TYR](https://localhost/id/168/TYR))
**When:** 2026-05-30T23:13:00-07:00
**By:** Carl Kesselman (carl@isi.edu)
**Supported by:** [tk-006](#tk-006) (the dual-purpose feature the wide-table join had to scope around)

Two reusable facts from running the analysis pipeline on this catalog:

- **`notebooks/roc_analysis.ipynb` now materializes a single joined wide
  table** (`prediction_wide_table.csv`, one row per held-out image: GT +
  every model's predicted class, confidence, and full per-class
  probability vector, columns prefixed `<model_config>__<asset_rid>__`)
  and commits it as an `Execution_Asset`. The pattern for a multi-model
  comparison on this template is therefore: point the `assets=` group at
  the prediction CSVs, and the notebook produces the wide table + ROC +
  confusion matrices + `roc_metrics.csv` in one provenance-tracked run.
  The wide table is the re-derivable artifact behind the report — a
  consumer never has to re-touch the dual-purpose feature
  ([tk-006](#tk-006)) because the join was done once, correctly, here. A
  built-in assertion fails the run if the wide-table accuracy disagrees
  with the per-experiment merge-cell accuracy, so a lossy/duplicating
  join can't ship silently. The reusable named config is
  `roc_quick_vs_large_toronto` (`src/configs/roc_analysis.py`), pointing
  at the same-named `assets` group (RP6 + SVC).
- **Gotcha worth knowing:** the catalog `Execution.Description` for a
  notebook run shows the *static* `notebook_config` description ("ROC
  curve analysis (default: quick vs extended training)") even when the
  actual asset group was overridden on the CLI. The real choice is
  recorded separately in the `[overrides: assets=...]` suffix the runner
  appends and in the resolved Hydra config asset — so to know what a
  notebook execution *actually* analyzed, read the overrides suffix /
  `config_choices`, not the prose description. `[observed]` This is
  cosmetic (provenance is intact in the override record), but a reader
  skimming execution descriptions could be misled.

**Weighed alternatives:** considered editing the notebook's
`run_notebook(...)` call to name `roc_quick_vs_large_toronto` directly so
the description prose would match; kept the notebook's config name
unbound and selected the asset group via a positional `assets=` override
instead, per the run-notebook skill's guidance (keep the notebook string
stable, select the target with positional overrides) — the description
mismatch is the price of that (correct) choice, recorded above so it
isn't mistaken for a bug.
