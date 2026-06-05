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
