<!-- statutor: plane=state | policy=overwrite_bounded (max 40 lines) | writer=executor | OVERWRITE, NEVER APPEND -->
# HANDOFF

last_verified: 2026-10-27 by `git rev-parse --short HEAD` (b6ac671) and `git status -sb`
last_worker: grok
last_machine: 1befd044f2ccc95033f353b2b7468adc
handoff_id: bf7766637d0547f375f8a9ed93f320e7
supersedes: e27b3164c1b0f68c3d67b1c24c8884b5

## Goal
Keep main free of a fixed product milestone. The TUI is an experiment on explore/tui (D-0004).

## Last verified state
- main is b6ac671, pushed, with the live-chunk timestamp fix. pytest was 240 passed before that push.
- D-0004 supersedes D-0003's milestone claim. T-0002 is off the main queue. T-0001 remains.
- The TUI design lives only on explore/tui. main does not contain it.

## Next action
On explore/tui, read docs/superpowers/specs/2026-10-27-tui-explore.md and try the job seam before any widget. On main, do not start that work.

## Gotchas
- D-0002 still holds: explore/tui is short-lived, not a second long-lived branch.
- The pipeline still prints and raises typer.Exit. Live SIGINT is process-global.
- Do not commit pixi.lock drift, repo-root media, or whisper.cpp pycache.
- bash_guard blocks shell writes to governed files.

## Do not touch
vendor/whisper.cpp/, capture/.build/, output/, plans/archive/, and the untracked media files in the repo root.
