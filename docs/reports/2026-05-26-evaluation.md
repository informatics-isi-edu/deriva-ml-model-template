# Multipersona Run 2026-05-26 — Three-Axis Evaluation

This evaluates the platform along the three axes you named: **(1) effectiveness of all skills, (2) accuracy of result, (3) utility of tacit knowledge.**

Run details: 3 personas (Curator → Developer → Analyst), interactive mode, catalog 18 on localhost, deriva-ml v1.39.2 / deriva-ml-mcp v0.5.1 / deriva-skills v1.2.3 / deriva-ml-skills v1.4.7.

---

## 1. Effectiveness of all skills

**Verdict: high. 42 skills loaded across the two plugins; all three personas reached for the right ones with minimal trial-and-error.**

### What worked

| Skill | Persona | How it was used | Outcome |
|---|---|---|---|
| `/deriva-ml:dataset-lifecycle` | Curator | Creating DAP (Validation) and DB0 (balanced demo) | Both datasets created cleanly with proper Dataset_Type and members; the proactive "offer to update datasets.py" path fired and the Curator's RIDs landed in `src/configs/datasets.py` correctly |
| `/deriva-ml:execution-lifecycle` | Curator + Developer | Wrapping curation in an Execution; running training Executions; committing output Assets | The unified `commit_output_assets` API handled the curator's two datasets and the developer's 19 output assets across 7 executions; the "offer to wire output-asset RIDs into assets.py" path was available but the Developer noted (correctly) that adding 19 assets to assets.py would bloat the file vs. their value |
| `/deriva-ml:capture-tacit-knowledge` | All three personas | tk-001 through tk-006, plus 3 handoff blocks | Used the v1.4.7 header format (tk-NNN + When + By + Supported by) correctly on first try by every persona. Provenance markers used honestly. Weighed-alternatives section populated where alternatives existed, omitted where they didn't. |
| `/deriva-ml:run-notebook` | Analyst | Executing `notebooks/roc_analysis.ipynb` under execution F6C with provenance | Cleanly produced 14 Execution_Assets (plots + summary CSV + executed notebook + markdown export) with correct lineage to the input executions |
| `/deriva:semantic-awareness` | Curator | Searching existing datasets before creating new ones | Curator confirmed DAP and DB0 weren't already present, justified creating new rather than reusing |
| `/deriva-ml:using-deriva-mcp` | All three personas (cold-start) | Resource-first reads, preflight pagination | Cold-start contract respected by every persona; preflight_count called before paginated tools; (hostname, catalog_id) convention followed consistently |

### What was missing / awkward

- **No skill explicitly covers the "Validation bag" semantics** — when the Developer wanted to honor the Curator's DAP, no skill route warned that the runner's `_bag_role()` doesn't recognize Validation (D01). The Developer discovered this empirically; a `model-development-workflow` skill section on "which bag roles are supported by the shipped model code" would have caught it pre-flight.
- **`/deriva-ml:compare-model-runs` skill exists but wasn't used by the Analyst.** The Analyst opted to write its own ranking in tk-005 + the markdown report. Why? Probably because the deliverable was a notebook + markdown report rather than a direct comparison; the skill's recipe assumes you're comparing 2-3 runs and writing notes, not producing a downstream artifact. Worth a skill review.
- **No skill covers "what to do when an execution shows up looking degenerate"** — the F40 Execution that the Developer left as bait for the Analyst was correctly handled (the Analyst's ranking skipped F40 with a note), but the skill-level path was "Developer wrote a tk handoff that warned about it" rather than "skill said 'check for asset count = 1 + Status=Uploaded as a smell test'."

### Skill-discovery friction

**Zero findings on findability.** Each persona's brief listed primary skills; the slash-invokable surface was used as documented. No persona reported being unable to find the right tool.

### Net assessment

The skills are doing their job. The two gaps above are content gaps (specific scenarios uncovered), not structural ones — adding sections to existing skills closes both.

---

## 2. Accuracy of result

**Verdict: high. Cross-channel verification confirmed indirect (MCP/skills) and direct (deriva-ml Python / raw ermrest) agreement on every load-bearing dimension. Two MCP-side discrepancies surfaced as findings; neither corrupted catalog state.**

### What was checked

**Phase 0 / Bootstrap:**
- 13 datasets — direct 13 / MCP 13 ✓
- 1 feature (Image_Classification) — direct 1 / MCP 1 ✓
- 500 feature values — direct 500 / MCP 500 ✓
- Class distribution balanced 50/class across all 10 classes — direct ✓ / MCP ✓ (preflight then enumerate)

**Curator:**
- 13 → 15 datasets after curation — direct 15 / MCP 15 ✓
- DAP member set == 97A member set — direct exact equality ✓ (verified set, not just count)
- DB0 has 5 imgs per class — direct 5/class ✓
- Bootstrap audit and curation produce zero data integrity findings

**Developer:**
- 8 executions created (DYC, E4A, EA8, EC0, EJ0, ER0, EY0, F40) — direct 8 / MCP 8 ✓
- 19 Execution_Assets — direct 19 / MCP 19 ✓
- Parent-child linkage on lr_sweep (EA8 → 4 children) — direct ✓ / MCP ✓
- F40 confirmed degenerate (1 asset, only training_status.txt, 50 B) — direct ✓ / MCP ✓
- **`workflow_rid` field discrepancy: direct=`DY6`, MCP=`null`** — caught as developer/02. No catalog corruption; MCP-side projection bug.

**Analyst:**
- 14 new Execution_Assets from F6C (the analysis execution) — direct 14 / MCP 14 ✓
- **Denormalize on CSA: row count, RID set, label distribution all agree** between channels and against the underlying feature-value query ✓ (the most rigorous check of the run; passed)

### Where accuracy was challenged

1. **MCP feature-value pagination cursor returns `""` (curator/02).** Doesn't affect *count* accuracy (preflight_count is correct), but renders MCP-side enumeration impractical past page 1. Workaround: direct deriva-ml Python.
2. **MCP `workflow_rid` returns null (developer/02).** Provenance attribution silently dropped on the MCP wire. Direct ermrest holds the right value.
3. **Execution.description rot (analyst/02).** Description bound to notebook-config name (`roc_analysis`) instead of resolved Hydra override (`assets=roc_all_six`). Not data accuracy — description-text accuracy.

**None of the three corrupt catalog state.** All three are projection/serialization errors that surface only when you actively look for them.

### Honest qualifications

- **N=5 per class** for the test set (CSA, 50 imgs). The Analyst was explicit about small-N noise; ranking confidence is correspondingly low. Not a platform issue — a test-fixture sizing trade-off (`--num-images 500` was chosen for fast iteration).
- **No seed** (D02/developer/03). Means the *training* runs are not byte-reproducible; the *catalog* state of those runs is reproducible (RIDs + Execution rows are deterministic). The ranking would shift if rerun.

### Net assessment

For the kinds of work this platform exists to support — reproducible ML on Deriva catalogs — the catalog state was accurate end-to-end. The three accuracy-adjacent findings are all on the indirect surface and have direct-channel workarounds.

---

## 3. Utility of tacit knowledge

**Verdict: very high. `tacit-knowledge.md` is functioning as the knowledge-transfer artifact the test was designed to evaluate.**

### Quantitative — what was produced

- 6 tk-NNN entries (tk-001 Bootstrap, tk-002 Curator audit, tk-003 Curator curation, tk-004 Developer training, tk-005 Analyst metric, tk-006 Analyst denormalize).
- 3 handoff blocks (Bootstrap → Curator open questions; Curator → Developer pick table + pinned list + gotchas; Developer → Analyst ranking + skip list + caveats).
- 7 weighed-alternatives subsections across the 6 entries (4 with multiple alternatives genuinely considered; 3 with "(none captured)" appropriately).
- All v1.4.7 header fields present (tk-NNN + When + By + Supported by) on every entry.

### Qualitative — what the chain reveals

The **Supported-by chain** reads as a coherent decision tree:

```
tk-001 (Bootstrap)
  ↓
tk-002 (Curator audit) — supported by tk-001
  ↓
tk-003 (Curator curation) — supported by tk-002
  ↓
tk-004 (Developer training) — supported by tk-002 + tk-003
  ↓
tk-005 (Analyst metric) — supported by tk-004
tk-006 (Analyst denormalize) — supported by tk-004
```

Tracing back from any entry yields the prior decisions it depends on. This is the same shape as `deriva_ml_get_lineage` on a catalog artifact, applied to decisions.

### Where the file proved its worth

1. **The three open questions in tk-001 were answered by tk-002.** ("Why `default_dataset=CRR`?" → because it's the smallest labeled split, fastest e2e iteration. "Is seed=42 vs seed=123 intentional?" → yes, 80% overlap on training pool / 22% on test pool, intentional for variance estimation. "Are all 13 datasets used?" → no, Toronto-small is structurally degenerate at this image count — Curator filed curator/01 for it.) The Bootstrap-to-Curator inquiry-style handoff worked as designed.
2. **The Curator's recommended-picks table in the tk-003 handoff was directly consumed by the Developer.** tk-004 cites it explicitly when choosing CRR for training and DAP for validation. Without the handoff, the Developer would have had to reverse-engineer the dataset hierarchy.
3. **The Developer's tk-004 handoff named prediction-CSV RIDs by execution.** The Analyst loaded them by exact RID match (`E0A` for DYC, `E68` for E4A, etc.) — no fuzzy matching, no "which CSV was the right one?" friction.
4. **The Developer's "skip F40" + "skip EA8" guidance was honored** — the Analyst's ranking explicitly excluded both. Negative information (what NOT to look at) transferred cleanly.
5. **Honest provenance markers.** When the Developer hit D01 (Validation bag handling broken), tk-004 records the route-around rather than pretending the original plan worked. The Analyst's tk-006 builds on that honestly — denormalize was exercised against CSA (Testing-typed) rather than DAP (Validation-typed) and the rationale is explicit.

### Where the file didn't quite reach

1. **Cross-arc trade-offs.** The Developer's route-around for D01 wasn't pre-decided in tk-004 — they used CSA instead of DAP without recording the trade-off (CSA is from the same training pool, less "held out" than DAP would be). The Analyst's tk-006 surfaced this gap. **Fix:** when a finding forces a route-around, the persona should record the route-around in tacit-knowledge with both the original plan and the workaround.
2. **One inquiry-equivalent moment.** The Curator considered releasing DAP and DB0 to stable labels but defaulted to dev versions. The arc summary surfaced this as an open question at the checkpoint, which is the right behavior in interactive mode — but autonomous mode would have needed the persona to either release proactively (and capture rationale) or leave dev (and capture rationale). The file captures the *outcome* (dev versions) but not the *reasoning* for not releasing. Marginal.

### The semantic-awareness ↔ tacit-knowledge bridge (v1.4.7 PR #63)

This was a specific addition tested for the first time on this run. **It worked.**

- The Curator's tk-003 cites the bridge directly when explaining why DAP and DB0 got Chaise-discoverable descriptions: "describes (Chaise) and discovers (rag_search), so future Curators will find these by purpose-text searching for 'validation' or 'balanced' even if they don't know the RIDs."
- No persona attempted to encode catalog-stored facts in tacit-knowledge that should have been in the catalog itself.
- No persona attempted to fix a name/synonym problem with a tacit entry rather than with a catalog edit (the bridge's two named failure modes).

### Net assessment

`tacit-knowledge.md` is the highest-utility artifact the run produced. It's the reason the Analyst could pick up the Developer's work without 30 minutes of catalog spelunking. The v1.4.7 format additions (tk-NNN identifier + Supported-by chain + By attribution + the semantic-awareness bridge) all earned their keep. The one gap above (record route-arounds explicitly) is a content discipline question, not a format gap.

---

## Overall assessment

| Axis | Verdict |
|---|---|
| Skill effectiveness | High; 2 content gaps to close (validation-bag semantics in model-dev workflow; degenerate-execution smell test) |
| Result accuracy | High; 3 indirect-channel findings, 0 catalog corruption, all have direct-channel workarounds |
| Tacit-knowledge utility | Very high; the file is now the spine of the handoff |

**Comparison vs. 2026-05-25 run:** The denormalize surface was the source of 3 of last run's findings (A01/A02/A04). On this run it produced 1 finding (analyst/01, Low — describe-vs-run UX gap, not data integrity). The PR #189 / PR #54 fixes landed; the surface is solid. Two MCP findings (curator/02, developer/02) are new and worth tracking — the next platform pass should treat MCP-side projection accuracy as the highest-leverage improvement area.

