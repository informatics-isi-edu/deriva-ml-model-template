# Analyst findings — capacity-sweep evaluation (catalog 69, 2026-06-05)

**Persona:** Analyst (third and last arc of the 2026-06-05 multipersona e2e run)
**Catalog:** `localhost` / catalog `69` (alias `e2e-test-20260605`)
**Analysis execution:** [TKM](https://localhost/chaise/record/#69/deriva-ml:Execution/RID=TKM)
(workflow TKE, status `Uploaded`)
**What I evaluated:** the Modeler's three capacity-sweep training runs —
**SR8** (quick), **T1A** (default), **TAC** (large) — all trained on the same
small labeled split (RQP: 400 train / 100 held-out test, seed=42), scored
against the held-out test partition RR6 (100 images, exactly 10 per class).

---

## TL;DR for a non-ML collaborator

We have three versions of the same image classifier, differing only in size
and how long they trained. I lined up every test image next to the right answer
and what each model guessed, then counted.

- **All three models are bad at naming the exact object** (20–26% correct out of
  10 categories), but **all three are good — about 80% — at the coarse question
  "is this a vehicle or an animal?"** The fine-grained skill just isn't there
  yet; the coarse skill is. That gap is the most useful single fact in here.
- **The mid-size model (default, 10 epochs) is the best of the three** at 26%.
  The biggest model (large, 20 epochs) is *not* the best — it memorised its
  training images and got slightly worse on new ones (24%). The smallest (quick,
  3 epochs) is the weakest (20%). More is not better here.
- **Where the models get confused, they get confused the way a person would
  expect**: cars ↔ trucks, ships ↔ planes, cats ↔ deer/horses. They mix up
  things that genuinely look alike. That's reassuring — it means the models are
  learning *something* real, just not enough of it.
- **The numbers the platform recorded for these runs are trustworthy.** I
  recomputed accuracy from scratch off the raw per-image data and got the exact
  same 20 / 26 / 24% the catalog already had. No discrepancy.

These are **pipeline-validation runs on a deliberately tiny dataset** (400
training images), not a serious model. Don't quote 20–26% as "how good a
CIFAR-10 CNN can be" — quote the *shape* of the result (mid-size wins, coarse >
fine, sensible confusions).

---

## The joined table (the team deliverable)

Everything below is derived from one wide table — **one row per evaluation
image**, carrying the image RID, its ground-truth class, and for each of the
three runs: the predicted class, the model's confidence, a correct/wrong flag,
and the full per-class softmax probability distribution. Any number in this
report can be re-derived from it.

- **In the worktree:** [`assets/analyst_joined_predictions.csv`](assets/analyst_joined_predictions.csv)
  (100 rows × 41 columns).
- **In the catalog (provenance-linked output asset):** RID **TN0**, produced by
  execution TKM.

How the table was built (see `scripts/analyst_analysis.py` +
`src/scripts/analyst_join.py`):

1. The predictions and the ground truth live in the **same** dual-purpose
   `Image_Classification` feature table (tk-003). Ground-truth rows have
   `Confidence` empty (written by the loader execution **CVP**, 1100 rows);
   prediction rows have `Confidence` populated (written by SR8/T1A/TAC, 100 rows
   each). I split on that predicate, then **joined predictions to ground truth
   on the shared `Image` RID**.
2. Per-class softmax probabilities (needed for ROC) came from each run's
   `prediction_probabilities.csv` asset (ST6 / T38 / TCA), merged in on the same
   `Image` key.

Row-count reconciliation: feature table held **1400** rows = 1100 ground truth
(CVP) + 100 × 3 predictions. The joined table is **100 images** — the RR6 test
partition every run scored. All three runs scored the *identical* 100 images
(the join asserts this), so the comparison is apples-to-apples.

---

## How I ranked the runs, and by what measure

Primary measure: **top-1 accuracy** on RR6 (fraction of the 100 test images
whose predicted class equals the true class). I also computed one-vs-rest ROC
AUC (macro and micro) from the softmax distributions, because with a 100-image
test set top-1 is noisy (±~3 images ≈ ±3 points per tk-004) and AUC uses the
full probability ranking rather than just the argmax.

Leaderboard ([`assets/analyst_run_metrics.csv`](assets/analyst_run_metrics.csv),
catalog RID **TN2**):

| Rank | Run | Exec | Config | Correct /100 | **Computed acc** | Recorded acc | Reconciles? | macro-AUC | micro-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | default | T1A | small_default — 10 ep, 32→64, 128h | 26 | **26.0%** | 26.0% | ✅ | 0.749 | 0.743 |
| 2 | large | TAC | small_large — 20 ep, 64→128, 256h | 24 | **24.0%** | 24.0% | ✅ | 0.740 | 0.741 |
| 3 | quick | SR8 | quick — 3 ep, 32→64, 128h | 20 | **20.0%** | 20.0% | ✅ | 0.739 | 0.643 |

Random-guess baseline = 10% (10 balanced classes). All three beat chance by
2–2.6×.

**The ranking is the same whether you sort by top-1 accuracy or by AUC**, and it
matches the Modeler's prediction (tk-004): *quick underfits, large overfits,
default is the least-bad generalizer.* The differences are small in absolute
terms (20 vs 26 = 6 images) — at this test-set size I'd call default and large a
near-tie on top-1, and the macro-AUCs (0.739 / 0.740 / 0.749) are within a hair
of each other. The one place a run clearly separates is **quick's micro-AUC
(0.643)**, well below the other two (0.743 / 0.741): the 3-epoch model's
*probability calibration* is markedly worse even where its argmax sometimes
lands right — it is confidently wrong more often. So if I had to pick one model
to take forward, it's **default (T1A)**: best top-1, best macro-AUC, and not
paying large's overfitting cost.

### Caveat that changes the "which is best" story (tk-005)

The recorded predictions are **final-epoch**, not best-epoch. For the large run
(TAC) that matters: per tk-004 its test accuracy *peaked at 34% around epoch 5–7
and then decayed to 24%* by epoch 20 as it overfit. So the 24% I'm scoring is
TAC's degraded final state, not its best. **If best-epoch checkpoints were
recorded, the large model might actually rank first.** This is a fair
apples-to-apples comparison of final-epoch states, but it is *not* a "best each
model could do" comparison. The per-epoch trajectory lives only in each run's
`training_log.txt` asset, not in the feature rows.

---

## Per-class breakdown

Per-class top-1 accuracy
([`assets/analyst_per_class_accuracy.csv`](assets/analyst_per_class_accuracy.csv),
catalog RID **TN4**). Each cell is accuracy on that class's 10 test images:

| True class | quick | default | large |
|---|---|---|---|
| airplane | **0.7** | 0.3 | 0.3 |
| automobile | 0.2 | 0.1 | 0.3 |
| bird | 0.0 | **0.5** | 0.2 |
| cat | 0.0 | 0.2 | 0.0 |
| deer | 0.0 | 0.0 | **0.4** |
| dog | 0.1 | 0.3 | 0.2 |
| frog | 0.0 | 0.3 | 0.2 |
| horse | **0.6** | 0.1 | 0.1 |
| ship | 0.0 | 0.1 | **0.4** |
| truck | 0.4 | **0.7** | 0.3 |

What this shows:

- **The per-class profiles are wildly different across runs even though the
  headline accuracies are close.** The quick model essentially bets on two
  classes — airplane (0.7) and horse (0.6) — and gets *zero* on five classes
  (bird, cat, deer, frog, ship). That is the signature of an undertrained model
  collapsing onto a couple of easy, distinctive shapes rather than learning all
  ten. Its decent overall 20% is carried by two classes.
- **The default model spreads its competence more evenly** (truck 0.7, bird 0.5,
  dog/frog 0.3) — fewer zeros, no single class doing all the work. That broader
  coverage, not a higher peak, is why it's the better generalizer.
- **`cat` is the hardest class for every run** (0.0 / 0.2 / 0.0) — exactly as
  CIFAR-10 lore predicts; cat is notoriously the lowest-accuracy class because
  it's small, deformable, and visually overlaps dog/deer/frog.

---

## Where the models confuse classes — and whether that matches domain intuition

This is the question I most wanted to answer as a domain expert: *when a model
is wrong, is it wrong in a way that makes sense?* Two cuts of the joined table:

### 1. Coarse semantic structure is learned even when fine labels aren't

I collapsed the 10 classes into two supergroups — **animals**
(bird/cat/deer/dog/frog/horse) and **vehicles**
(airplane/automobile/ship/truck) — and asked: did each model at least get the
*supergroup* right?

| Run | Coarse (animal-vs-vehicle) accuracy | Fine top-1 accuracy |
|---|---|---|
| quick | **79%** | 20% |
| default | **81%** | 26% |
| large | **82%** | 24% |

All three are ~**4× better at the coarse question than the fine one**. The
models genuinely learned the high-level "vehicle vs. living thing" distinction;
they just can't reliably tell a truck from a car or a cat from a deer yet. For a
domain reader this is the single most informative result: there *is* real signal
in these models, it's just at the wrong granularity for the task — exactly what
you'd expect from a 2-layer CNN trained on 400 images.

### 2. The actual confusion pairs are the ones a person would predict

Top confusion pairs (true → predicted, summed across all three runs;
re-derivable from the joined table and visualised in the per-run confusion-matrix
plots, catalog RIDs **TN6** quick / **TN8** default / **TNA** large):

| true → predicted | count (3 runs) | type | domain-sensible? |
|---|---|---|---|
| automobile → truck | 15 | vehicle ↔ vehicle | ✅ both wheeled road vehicles |
| ship → airplane | 11 | vehicle ↔ vehicle | ✅ both large, often sky/water background |
| ship → truck | 9 | vehicle ↔ vehicle | ✅ |
| bird → horse | 8 | animal ↔ animal | ✅ |
| bird → deer | 8 | animal ↔ animal | ✅ |
| horse → dog | 7 | animal ↔ animal | ✅ four-legged mammals |
| cat → deer | 7 | animal ↔ animal | ✅ |
| frog → cat | 7 | animal ↔ animal | ✅ |
| truck → automobile | 7 | vehicle ↔ vehicle | ✅ (the reverse of the #1 pair) |
| deer → truck | 7 | cross-type | ⚠️ the main "wrong-supergroup" leak |

**The dominant errors stay inside their supergroup** — cars↔trucks and
ships↔planes among vehicles; deer/horse/dog/cat/bird/frog mixing among animals.
`automobile ↔ truck` being the single most-confused pair is *the* canonical
CIFAR-10 confusion, and it shows up here loud and clear. The genuinely
cross-type leaks (deer→truck, airplane→deer) are a minority. This is the
behaviour I'd want to see: the models confuse things that actually look alike,
which means their mistakes are systematic and explainable, not random.

### ROC

The micro-averaged one-vs-rest ROC overlay (catalog RID **TNC**,
[`assets/roc_micro_overlay.png`](assets/roc_micro_overlay.png)) shows all three
curves well above the chance diagonal (micro-AUC 0.64–0.74), with quick clearly
the lowest. Consistent with the calibration story above: even when these models
miss the top-1 label, their probability ranking carries real signal — the right
class is often ranked highly even when it's not ranked #1.

---

## Did the recorded metrics reconcile with my own computation?

**Yes, exactly.** I recomputed top-1 accuracy from the raw joined per-image data
(predicted class == true class, counted over 100 images) and got **20 / 26 / 24%
for quick / default / large** — identical to the catalog-recorded `test_acc` for
each run (the `reconciles` column in the leaderboard is `True` for all three).
This is the load-bearing integrity check of the whole exercise: the platform's
stored numbers are reproducible from the stored raw data. No divergence to
explain.

Two reconciliations worth stating explicitly so a skeptical reader can re-walk
them:

- The **feature-table argmax agrees with the CSV `Predicted_Class`** — the
  `Image_Class` written as the feature row is the same class the softmax CSV
  calls the prediction, so the two surfaces tell the same story (expected, since
  the model writes both from the same forward pass, but I verified the join
  rather than assuming it).
- The reconciled 24% for **large** is its *final-epoch* number, which per tk-005
  is below its mid-training peak (~34%). So "recorded = computed" is true, *and*
  "recorded ≠ the model's best" is also true — both facts hold without
  contradiction, and an analyst needs to carry both.

---

## Provenance (every figure traces back to the data)

The analysis was captured as DerivaML execution **TKM** (workflow TKE, type
Analysis/Testing), which **consumed the three prediction CSVs (ST6/T38/TCA) as
declared inputs** and produced all outputs as linked assets. `get_lineage(TN0)`
walks cleanly: joined table TN0 ← TKM ← {ST6, T38, TCA} ← {SR8, T1A, TAC} ←
dataset RQP v0.1.0.post1.dev1 ← split execution ← training subset KE0. The full
chain from this report's headline number back to the source images is one call
away and intact.

**Output assets (catalog RIDs, all under execution TKM):**

| RID | File | Contents |
|---|---|---|
| TN0 | analyst_joined_predictions.csv | the joined wide table (the deliverable) |
| TN2 | analyst_run_metrics.csv | per-run leaderboard + reconciliation |
| TN4 | analyst_per_class_accuracy.csv | per-run per-class accuracy |
| TN6 | confusion_quick.png | confusion matrix, quick run |
| TN8 | confusion_default.png | confusion matrix, default run |
| TNA | confusion_large.png | confusion matrix, large run |
| TNC | roc_micro_overlay.png | micro-averaged ROC, all three runs |

Local copies of all seven are in [`assets/`](assets/) beside this report.

---

## Reproduction

```bash
# Reusable, RID-free join/metric logic (unit-tested, no catalog needed):
uv run python -m pytest tests/test_analyst_join.py -v

# The catalog-facing analysis (clean tree → records a provenance execution):
uv run python scripts/analyst_analysis.py
#   --dry-run            build everything, skip the catalog write
#   (prefix DERIVA_ML_ALLOW_DIRTY=true for dirty-tree dev iteration)
```

The split between `src/scripts/analyst_join.py` (pure functions, no RIDs,
reusable template config) and `scripts/analyst_analysis.py` (catalog-69 RIDs
baked in, `[E2E-DROP]`) is deliberate — the logic is portable, only the wiring
is run-specific.
