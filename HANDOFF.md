<!-- statutor: plane=state | policy=overwrite_bounded (max 40 lines) | writer=executor | OVERWRITE, NEVER APPEND -->
# HANDOFF

last_verified: 2026-10-27 by `pixi run pytest -q` (240 passed)
last_worker: grok
last_machine: 1befd044f2ccc95033f353b2b7468adc
handoff_id: e27b3164c1b0f68c3d67b1c24c8884b5
supersedes: 0a13255a2dbb27a393fe6a0262be71a2

## Goal
Build the TUI on one shared transcription job (D-0003). Widgets come after that seam.

## Last verified state
- main was v0.3.0 at 212a6ca. This commit keeps live-chunk indexes and sets the TUI milestone.
- Live chunked merge now keeps a silent or failed chunk's index, so later timestamps stay put. Regression: tests/test_live_chunked.py.
- `pixi run pytest -q`: 240 passed. CUDA smoke was not run.

## Next action
T-0002: extract the job from scribe_md/cli.py. file, url, and the parallel prepare/finalize path should call it. The CLI keeps Typer and Rich. Do not start a TUI widget until that call exists.

## Gotchas
- The pipeline still prints and raises typer.Exit. Live SIGINT is process-global.
- Live chunked mode accepts --diarize, then skips it after capture.
- This machine is Apple Silicon. Do not commit pixi.lock drift, repo-root media, or whisper.cpp pycache.
- bash_guard blocks shell writes to governed files.

## Do not touch
vendor/whisper.cpp/, capture/.build/, output/, plans/archive/, and the untracked media files in the repo root.
