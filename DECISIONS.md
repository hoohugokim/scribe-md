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

## D-0003 — The next milestone is a TUI on one shared job
**Status:** accepted
**Context:** v0.3.0 landed multi-input and CUDA `--gpus`. `scribe_md/cli.py` still owns orchestration, Rich output, and `typer.Exit`, and file, url, and live each run their own copy of the pipeline.
**Decision:** The next milestone is a scribe-md TUI. It calls one job interface extracted from the CLI. It does not grow a second pipeline. T-0001, the CUDA smoke, stays open and does not block the TUI.
**Consequences:** Progress and errors have to leave the CLI before widgets exist. Widgets do not call ffmpeg, whisper, or yt-dlp.

## D-0004 — The TUI is a branch experiment, not the milestone
**Status:** accepted
**Supersedes:** D-0003's claim that the TUI is the next milestone
**Context:** D-0003 fixed the TUI as the milestone on main. We want to try that design without making it the committed product path.
**Decision:** Explore the TUI on the short-lived branch `explore/tui`. main has no fixed product milestone. D-0002 still holds: main is the only long-lived branch. The one-job seam in D-0003 still applies inside the experiment.
**Consequences:** T-0002 is not on the main queue. CUDA smoke stays T-0001. Merging or deleting `explore/tui` is a later decision.
