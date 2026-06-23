# Split CIFAR from template — progress ledger

Plan: docs/superpowers/plans/2026-06-05-split-cifar-from-template.md
Mode: subagent-driven. Checkpoints: STOP before B2 (gh repo create --public) and C1 (open PR).

- [x] A1: complete (base 3c8bd82, worktree ../deriva-ml-model-template-strip, baseline 68 passed/2 skipped; concern: stray eye-ai scripts/test_bag_fk_traversal.py breaks root pytest collection -> delete in A3)
- [x] A2: complete (47 files staged to ../_cifar-staging, 0 lost; .python-version absent in source)
- [x] A3: complete (25 files removed incl eye-ai script; CONCERNS for A4: (1) model_config group now unregistered - need generic default_model; (2) workflow.py refs deleted Cifar10CNNWorkflow; (3) tests/test_notebook_examples.py remains - check it)
- [x] A4: complete + REVIEWED CLEAN (spec PASS, quality Approved, 0 Critical/0 Important). Configs are scaffolds; generic default_model added (configs/model.py:example_model); roc_analysis.py->analysis.py; dead load_all_configs shim removed; pyproject de-CIFAR-ed. A6 MUST fix doc-side roc_analysis/CIFAR refs: README.md:201,224,231; CLAUDE.md:40; docs/index.md:45,52; docs/configuration/notebooks.md; docs/getting-started/*; docs/workflow/experiments.md; mkdocs.yml:35 (CIFAR-10 Example nav).
- [x] A5: complete (commit 6f4e7ae, 10 passed, ruff clean, strip committed)
- [x] A6: complete + REVIEWED CLEAN (commit 16e4b82, spec PASS, quality Approved, guide accurate vs real scaffolds, 0 broken refs). Minor: upstream deriva-ml CLI footer prints cifar10_quick (separate lib issue).
- [x] B1: complete (local repo deriva-ml-cifar-example, commit 0e3aed5, 66 passed/2 skipped, ruff clean, NO remote yet, 47 files; kept load_all_configs as proper def for CIFAR test)
- [x] B2: complete (user created repo; I added origin + pushed 0e3aed5 to informatics-isi-edu/deriva-ml-cifar-example, public, was empty)
- [x] C1: complete (PR #64 opened, not auto-merged). Pending: final whole-branch review, then user-merge + cleanup.

## Final whole-branch review: READY TO MERGE (0 Critical, 0 Important, 3 Minor)
Minor fixes to apply before merge:
- M1: README layout diagram lists src/configs/dev/ + notebooks/ (untracked empty dirs) -> .gitkeep or annotate
- M2: deriva.py docstring says dev/deriva.py; others say dev/deriva_<env>.py -> unify
- M3: customization.md:173 refs /deriva-ml:new-model -> verify correct skill name (likely model-development-workflow)

## COMPLETE
- M1+M2 fixed (commit a455e75, pushed to PR #64). M3 was a false positive (/deriva-ml:new-model is a real skill).
- PR #64 open + ready to merge (final review: READY TO MERGE). NOT auto-merged — user merges.
- New repo informatics-isi-edu/deriva-ml-cifar-example pushed (0e3aed5).
