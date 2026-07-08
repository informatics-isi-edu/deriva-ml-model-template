---
type: Vocabulary
title: Tacit Knowledge — topic controlled vocabulary
description: >
  Repo-local controlled vocabulary the LLM classifies tacit-knowledge
  entries against. Human-gated: new terms are proposed into the index's
  candidate-terms list and confirmed here. Cross-links catalog CV terms by RID.
tags: [tacit-knowledge, vocabulary, deriva-ml]
---

# Tacit Knowledge — Topic Vocabulary

Each entry in `tacit-knowledge.md` is classified under one or more of these
terms. Reuse an existing term via synonym-aware lookup before proposing a new
one; new terms are human-gated (see the index's `candidate-terms` list).

## entity-anchored

- **dataset-construction** — how a dataset was assembled, split, or subsampled
- **dataset-versioning** — why a dataset version was cut or pinned
- **feature-design** — why a feature exists and how it is shaped
- **model-configuration** — hyperparameter and architecture choices for a model
- **workflow-typing** — why a workflow was classified as it was
- **execution-provenance** — what an execution consumed, produced, or established

## entity-free

- **process-convention** — a recurring 'whenever we do X we also do Y' pattern
- **domain-background** — target-domain facts, confounds, and conventions
- **tooling-gotcha** — a non-obvious behavior of the toolchain or platform
- **team-ownership** — role/process facts about who owns or decides what
- **dead-end** — an approach that was tried and abandoned, and why
