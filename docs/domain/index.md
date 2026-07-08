# Domain Background

Semantic, refined-in-place background about the target domain — facts,
confounds, and methodological conventions a cross-disciplinary newcomer needs.
One `type: Concept` doc per subject (e.g. `staining-variance.md`), refined in
place over time. Distinct from the episodic tacit-knowledge Log and from
`docs/design/` up-front plans. Link catalog vocabulary-term descriptions by RID
rather than restating them. A tacit-knowledge Log entry may *anchor* to a subject
here (Family C of the anchor taxonomy).

This file is the bundle's OKF `index.md`: it lists the Concept docs below, each
as `* [title](file.md) - description`, with the description taken from the
Concept doc's own frontmatter. It carries no frontmatter of its own. Give each
Concept doc a discriminating `tags:` line (e.g. `[domain, site-effect, imaging]`
vs. `[domain, cohort, sampling-bias]`) so a human can tell siblings apart —
descriptive only; the LLM reaches a Concept doc via its Family-C anchor, not tags.

## Subjects

_none yet — add one bullet per Concept doc, e.g._
_`* [Staining variance](staining-variance.md) - stain differences across sites`_
