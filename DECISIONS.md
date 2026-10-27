<!-- statutor: plane=log | policy=append_only (insertions only; supersede, never edit) | writer=orchestrator/human -->
# DECISIONS

## D-0001 — Adopt the statutor ledger framework
**Status:** accepted
**Context:** Multi-agent sessions re-litigate settled questions and lose state across context windows.
**Decision:** Four-plane typed ledger, single writer per file, hook-enforced mutation policies.
**Consequences:** HANDOFF.md is overwrite-only and bounded; this file is append-only; CHANGELOG.md is generated from conventional commits, never hand-maintained.

## D-0002 — main is the only long-lived branch
**Status:** accepted
**Context:** feat/multi-gpu-parallel held the finished multi-input and CUDA scheduler work (PR #3) beside main after the code was complete.
**Decision:** Merge that work to main and delete the feature branch. Feature branches die at merge. main is the only long-lived branch.
**Consequences:** Later work starts from main. Vulkan multi-GPU stays deferred. CUDA hardware smoke is T-0001, not a branch.
