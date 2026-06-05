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
### tk-001 — Bootstrap of e2e-test-20260605 (catalog 69) ([dataset H8M](https://localhost/id/69/H8M@357-14PS-8W2T))
**When:** 2026-06-05T10:30:00-07:00
**By:** Carl Kesselman (carl@isi.edu)

Phase-0 bootstrap of the 2026-06-05 multipersona e2e run. Created a
fresh localhost catalog (id 69, alias `e2e-test-20260605`) and
populated it with `load-cifar10` at `--num-images 1100` (550 train /
550 test — the floor is `>1000` so the small-Toronto-split family
stays a strict subset; see the loader's `SmallVariantDegenerateError`
guard). The full CIFAR-10 dataset hierarchy hangs off the `Complete`
dataset [H8M](https://localhost/id/69/H8M@357-14PS-8W2T): canonical
Training [KE0](https://localhost/id/69/KE0@357-14PS-8W2T) / Testing
[KEA](https://localhost/id/69/KEA@357-14PS-8W2T), the stratified
labeled splits, and the `Subsample` small variants. `default_dataset`
in `src/configs/datasets.py` points at the small labeled split
[RQP](https://localhost/id/69/RQP@357-14PS-8W2T) — labeled on both
partitions for evaluation, small for fast iteration.

Sibling versions for this run: deriva-ml `4d56677d` (one docs-only
commit past the v1.45.0 tag), deriva-mcp-core / deriva-ml-mcp-plugin
at their respective `main` HEADs; the MCP test container was rebuilt
against these. The e2e env's transitive deps are pinned on the
`e2e-test/2026-06-05` branch (an `[E2E-DROP]` commit).

Bootstrap was not clean: the `--phase datasets` step initially aborted
on a template bug — `cifar_canonical_partition` read the denormalized
column `Image.filename` (lowercase) but deriva-ml's denormalization
produces `Image.Filename` (catalog column case). Fixed in
`src/scripts/_cifar10_datasets.py` (a genuine template fix, committed
to the branch for cherry-pick to `main`, not an `[E2E-DROP]`); see
`findings/phase0/01`. The partial first run also left an **orphan**
`Complete` dataset (RID `F2J`, 0 children, unreferenced) that the
idempotent retry did not reuse; left in place as cosmetic, see
`findings/phase0/02`.

Implications for collaborators: the catalog passed the Phase-0
fail-fast gate — every labeled partition has `Image_Classification`
ground truth on 100% of its members, and the class distribution is
exactly uniform across all 10 CIFAR-10 classes (the loader did not
regress into a skewed distribution). Two `Complete` datasets exist
(H8M live, F2J orphan); a Curator browsing the catalog will see both —
work from H8M. `Image_Classification` is a dual-purpose feature table:
the loader writes ground-truth rows, and training executions will
later write prediction rows into the same table, so once any model
run has happened, read ground truth by filtering on the loader
execution rather than taking the whole table.

---

<a id="tk-002"></a>
### tk-002 — Curator verification: canonical splits are leak-free and class-balanced ([dataset KE0](https://localhost/id/69/KE0@357-14PS-8W2T))
**When:** 2026-06-05T17:30:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/372da2af-ee50-42b0-ada8-7c9eba10493c)
**Supported by:** [tk-001](#tk-001) (bootstrapped the split hierarchy this audit characterizes)

Question the audit answered: do the split datasets actually mean what
their names and descriptions imply, and is the substrate safe to hand to
a Modeler/Analyst? The skeptical worry is *test leakage* — a training
split that secretly overlaps the test partition silently inflates every
downstream accuracy number. Verified the whole family with set
arithmetic on the actual `Image` member RIDs (read from `Dataset_Image`),
not by trusting the descriptions. Reproducible via
`scripts/curator_verify_splits.py` (read-only; 19 checks, all PASS).

Durable guarantees established (these survive normal catalog writes —
they're properties of the seeded membership, which downstream personas
don't mutate):
- Canonical Training [KE0](https://localhost/id/69/KE0@357-14PS-8W2T)
  and Testing [KEA](https://localhost/id/69/KEA@357-14PS-8W2T) are
  disjoint and partition the Complete set exactly (550 + 550 = 1100).
- The training-derived labeled splits (QMA/QMM and the small
  RQW/RR6) draw **only** from KE0 and have **zero** intersection with
  KEA. So an Analyst can evaluate on KEA against a model trained on any
  labeled-split training partition without test-set contamination — the
  test partition is genuinely held out. This is the load-bearing
  property for honest evaluation.
- Every partition checked (KE0, KEA, RQW, RR6) is exactly class-balanced
  across all 10 classes (KE0/KEA: 55/class; RQW: 40/class; RR6:
  10/class). The stratification the descriptions claim is real.

A term-of-art for the domain reader: "stratified split" means the
class proportions are preserved in each partition — here, exactly equal
counts per class rather than merely *approximately* preserved, because
1100 / 10 divides evenly at every split size.

Implications for collaborators: the Modeler can train on KE0 (or any
`*_labeled_training` partition) and the Analyst can evaluate on KEA (or
the matching `*_labeled_testing` holdout) with confidence that no image
crosses the train/test boundary. The small labeled split
[RQP](https://localhost/id/69/RQP@357-14PS-8W2T) (RQW train / RR6
holdout) is the right default for fast iteration and ROC work — labeled
on both sides, leak-free, and stratified.

---

<a id="tk-003"></a>
### tk-003 — Convention — Image_Classification is dual-purpose; and which Complete dataset to use ([feature row CWP](https://localhost/id/69/CWP@357-14PS-8W2T))
**When:** 2026-06-05T17:35:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/372da2af-ee50-42b0-ada8-7c9eba10493c)
**Supported by:** [tk-001](#tk-001) (noted the dual-purpose table and the F2J orphan in passing)

Two durable conventions a future reader needs, both surfaced by the
Curator audit.

**(1) `Image_Classification` is a dual-purpose feature table.** Rows are
written by two distinct kinds of execution and are not distinguishable
by table membership alone: the loader execution writes ground-truth rows
with `Confidence IS NULL`; training executions write prediction rows with
`Confidence` populated (0–1). At audit time the table held exactly 1100
rows (one ground-truth label per image, perfectly class-balanced at
110/class) — but that count is a *snapshot* that grows the moment any
training run records predictions. When reading this feature as ground
truth, filter by the loader execution RID **or** by `Confidence IS NULL`;
an unfiltered read returns ground truth + every recorded prediction
interleaved. The `newest` selector is **not** a safe substitute —
"newest" is whichever execution last wrote, not "ground truth." (My
verification script uses the `Confidence IS NULL` filter for exactly this
reason, so it stays correct after the Modeler runs.)

**(2) Use Complete dataset [H8M](https://localhost/id/69/H8M@357-14PS-8W2T),
never F2J.** The catalog has two `Complete,Labeled` datasets with
byte-identical descriptions and identical 1100-image membership (F2J is a
*full duplicate*, not the empty 0-children husk the bootstrap notes
implied — see `findings/curator/01`). What distinguishes them is
lineage: the entire split family derives from H8M (via execution H7M);
**nothing** derives from F2J. Pinning F2J would silently sever provenance
for anything built on it. The config name → RID mapping (which the
catalog does not itself store) is therefore: `cifar10_complete` → H8M,
and F2J should be treated as do-not-use. Not deleted — deleting a
dataset is a destructive op reserved for explicit user authorization.

---

<a id="tk-004"></a>
### tk-004 — Modeler: capacity sweep on the small labeled split, 3 differentiated runs for the Analyst ([execution TAC](https://localhost/id/69/TAC@357-1GX4-RGMT))
**When:** 2026-06-05T11:55:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/372da2af-ee50-42b0-ada8-7c9eba10493c)
**Supported by:** [tk-002](#tk-002) (proved RQW/RR6 are leak-free and class-balanced — the precondition for trusting these test numbers), [tk-003](#tk-003) (dual-purpose feature convention these prediction rows obey)

Hypothesis under test: does the training pipeline run end-to-end against
a real catalog dataset, and do varied hyperparameters produce *different*
outputs (vs all runs collapsing to the same number)? Ran a deliberate
**capacity sweep** — three runs, identical dataset and seed, increasing
only model capacity + training duration — so any accuracy difference is
attributable to the model, not to the data. All three trained on
[dataset RQP v0.1.0.post1.dev1](https://localhost/id/69/RQP@357-1GX4-RGMT)
(small labeled split: RQW 400-image train / RR6 100-image held-out test,
seed=42) and recorded predictions on the RR6 partition.

Term-of-art for the platform reader: a "capacity sweep" holds the data
fixed and scales the model up; rising train accuracy with flat-or-falling
*test* accuracy is the signature of **overfitting** (the model memorizing
training images rather than learning generalizable features).

The three runs (headline = final-epoch test accuracy on RR6, the number
that became the recorded predictions; random-guess baseline is 10% across
10 balanced classes):

| Run | Execution | Config | Epochs / Arch | final train_acc | recorded test_acc | best test_acc |
|---|---|---|---|---|---|---|
| 1 | [SR8](https://localhost/id/69/SR8@357-1GX4-RGMT) | `cifar10_quick` | 3 / 32→64, 128h | 28.75% | **20%** | 27% (ep1) |
| 2 | [T1A](https://localhost/id/69/T1A@357-1GX4-RGMT) | `cifar10_small_default` | 10 / 32→64, 128h | 63.50% | **26%** | 29% (ep7) |
| 3 | [TAC](https://localhost/id/69/TAC@357-1GX4-RGMT) | `cifar10_small_large` | 20 / 64→128, 256h | 99.50% | **24%** | 34% (ep5/7) |

Findings: (1) The pipeline works end-to-end against a real catalog —
three clean Uploaded executions, each with weights + training log +
prediction CSV linked, predictions written as Image_Classification
feature rows. (2) The runs differentiate sharply: train accuracy spans
29%→99.5%, a clear learning signal, not noise. (3) The high-capacity run
(TAC) is a textbook overfit — train hits 99.5% while test peaks at 34%
(epoch 5/7) then *decays* to 24% by epoch 20 as test loss climbs
2.25→3.54. On only 400 training images, more capacity bought
memorization, not generalization. The *final-epoch* prediction the model
recorded (24%) is therefore worse than its mid-training peak (34%) —
a consequence of the template recording final-epoch predictions, not
best-epoch (see [tk-005](#tk-005)).

Implications for the Analyst: these are pipeline-validation + capacity-
characterization runs, **not** performance baselines — don't cite 20/26/24%
as model-capability claims; the test partition is 100 images, so each
point is ±~3 images of noise. The interesting comparison is the *shape*
(quick underfits, large overfits, default is the least-bad generalizer),
not the absolute ranking. All three share dataset and seed, so they are
directly comparable. Predictions live in the
[Image_Classification feature](https://localhost/id/69/CWP@357-1GX4-RGMT)
filtered by `Execution=SR8|T1A|TAC` (100 rows each, `Confidence`
populated); ground truth is the same table filtered `Confidence IS NULL`
per [tk-003](#tk-003).

---

<a id="tk-005"></a>
### tk-005 — Convention — capacity-comparison experiments hold dataset + seed fixed; template records final-epoch (not best-epoch) predictions
**When:** 2026-06-05T11:58:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/372da2af-ee50-42b0-ada8-7c9eba10493c)
**Supported by:** [tk-004](#tk-004) (the run set that motivated both conventions)

Two durable notes a future modeler/analyst on this project needs.

**(1) Why the new `cifar10_small_*` experiments hold data + seed
constant.** Added two experiment presets — `cifar10_small_default`
(default_model, 10 epochs) and `cifar10_small_large` (cifar10_large,
20 epochs) — both pinned to `cifar10_small_labeled_split`, joining the
pre-existing `cifar10_quick` (3 epochs, same dataset). The point of the
trio is a *controlled* comparison: same train/test partitions, same
seed=42 (the template default, which matches the split's own seed), so
the only independent variable across the three is model capacity ×
training duration. If a future run wants to compare *datasets* instead,
vary the `datasets=` group and hold the model fixed — don't conflate the
two axes in one comparison. These presets bake in no catalog RIDs (they
reference dataset *group names*), so they are reusable template config,
not catalog-69-specific.

**(2) The template records FINAL-epoch predictions, not best-epoch.**
`src/models/cifar10_cnn.py` writes predictions once, after the last
training epoch, tagged `source_label="epoch_N"`. It does **not** track
or restore a best-validation checkpoint. Consequence, made concrete by
the TAC run in [tk-004](#tk-004): when a model overfits, the recorded
predictions reflect the *degraded* final state (24%), not the model's
peak (34% at epoch 5/7). An analyst comparing recorded predictions
across these runs is comparing final-epoch states, which is fair (it's
apples-to-apples) but is **not** a "best each model could do" comparison.
The per-epoch trajectory needed to see the peak lives only in each run's
`training_log.txt` execution asset, not in the feature rows. If
best-epoch predictions are ever wanted, that's a model-code change
(add save-best + restore before the predict step), not a config tweak.

---

<a id="tk-006"></a>
### tk-006 — Analyst ranking + the coarse-vs-fine reading of the capacity sweep ([execution TKM](https://localhost/chaise/record/#69/deriva-ml:Execution/RID=TKM))
**When:** 2026-06-05T12:10:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/372da2af-ee50-42b0-ada8-7c9eba10493c)
**Supported by:** [tk-004](#tk-004) (the three runs + their recorded test_acc this analysis scores), [tk-005](#tk-005) (final-epoch caveat that bounds the ranking claim)

Interpretive judgment a future analyst/modeler on this project should
inherit, from scoring SR8/T1A/TAC against ground truth on RR6 (the joined
wide table, [asset TN0](https://localhost/id/69/TN0); report at
`docs/reports/2026-06-05-analyst-findings.md`).

**Ranking.** By top-1 accuracy on the 100-image RR6 holdout: **default
(T1A) 26% > large (TAC) 24% > quick (SR8) 20%** (random baseline 10%).
The ranking is identical under macro-AUC (0.749 / 0.740 / 0.739), so it's
robust to the metric choice. **If you have to pick one model, pick
default (T1A)**: best top-1, best macro-AUC, and it doesn't pay large's
overfitting tax. BUT — per [tk-005](#tk-005) these are *final-epoch*
predictions; TAC peaked ~34% mid-training and decayed to 24%, so on a
best-epoch basis large could plausibly win. State the ranking as
"final-epoch states," never as "best each model can do." With 100 test
images each point is ±~3 images, so default-vs-large is a near-tie on
top-1; the only clean separation is quick's poor *calibration*
(micro-AUC 0.643 vs ~0.74) — the 3-epoch model is confidently wrong more
often, visible in ROC but invisible in top-1.

**The load-bearing domain insight: coarse >> fine.** All three models
score **~80% on the coarse "animal vs vehicle" question** while scoring
only 20–26% on the fine 10-way label (computed by collapsing the 10
classes into animals {bird,cat,deer,dog,frog,horse} vs vehicles
{airplane,automobile,ship,truck} and checking supergroup agreement).
The signal these models learned is real but at the wrong granularity —
exactly what a 2-layer CNN on 400 images should produce. For a domain
reader this single fact ("they know it's a vehicle, they just can't tell
truck from car") is more informative than any top-1 number. The
confusions confirm it: the dominant error pairs stay *within* supergroup
— `automobile↔truck` (the canonical CIFAR-10 confusion, and the single
most-confused pair here), `ship↔airplane`, and animal↔animal mixing
(cat↔deer/horse/frog). `cat` is the hardest class for every run (0.0/0.2/0.0),
matching CIFAR-10 lore. So the models confuse things that genuinely look
alike — systematic, explainable errors, not noise. A future analyst
should report the *shape* (coarse-learned, sensible confusions, mid-size
wins), not cite 20–26% as a capability claim — these are tiny-dataset
pipeline-validation runs.

---

<a id="tk-007"></a>
### tk-007 — Convention — capture the analysis itself as a provenance execution; recorded test_acc reconciled exactly ([asset TN0](https://localhost/id/69/TN0))
**When:** 2026-06-05T12:12:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/372da2af-ee50-42b0-ada8-7c9eba10493c)
**Supported by:** [tk-003](#tk-003) (the `Confidence IS NULL` ground-truth filter this join depends on), [tk-006](#tk-006) (the ranking these outputs support)

Two durable notes from the Analyst arc.

**(1) Run the evaluation as a DerivaML execution that *consumes the
prediction CSVs as declared inputs*.** I captured the analysis as
execution [TKM](https://localhost/chaise/record/#69/deriva-ml:Execution/RID=TKM)
(workflow type Analysis/Testing) whose `ExecutionConfiguration(assets=[ST6,
T38, TCA])` declares the three runs' prediction CSVs as inputs. The payoff:
`get_lineage(TN0)` (the joined-table asset) walks in one call all the way
back — TN0 ← TKM ← {ST6,T38,TCA} ← {SR8,T1A,TAC} ← dataset RQP
v0.1.0.post1.dev1 ← split exec ← KE0. Every figure in the report is
therefore reproducible *and* traceable to the exact data+code that made it.
The alternative (an ad-hoc script that just reads and plots) loses that
chain. Cost is low: a clean tree before the run (commit the analysis code
first) so the git hash is honest; dev iteration uses
`--dry-run` + `DERIVA_ML_ALLOW_DIRTY=true`. Design split that's worth
copying: pure RID-free join/metric logic in `src/scripts/analyst_join.py`
(unit-tested, reusable template config), catalog RIDs only in the
`[E2E-DROP]` driver `scripts/analyst_analysis.py`.

**(2) The recorded test_acc reconciled *exactly* with independent
recomputation.** Recomputing top-1 from the raw joined per-image rows gave
20/26/24% — byte-identical to the catalog-recorded `test_acc` for
SR8/T1A/TAC. This is the integrity check that matters: the platform's
stored numbers are reproducible from its stored raw data, with no fudge.
Note the two facts that coexist without contradiction: "recorded ==
recomputed" is true, and (per [tk-005](#tk-005)) "recorded != the model's
best epoch" is *also* true — an analyst must carry both. Also verified the
feature-row argmax (`Image_Class`) agrees with the CSV `Predicted_Class`,
so the two prediction surfaces tell the same story. No platform bug
surfaced during the Analyst arc; the one cross-tier friction point
(generic `count_table`/`query_*` need the *physical* feature-table name
`Execution_Image_Image_Classification`, not the logical
`Image_Classification`) was already filed by the Modeler
(`findings/modeler/01`), so I routed around it rather than re-filing.
