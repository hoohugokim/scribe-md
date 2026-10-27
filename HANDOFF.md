<!-- statutor: plane=state | policy=overwrite_bounded (max 40 lines) | writer=executor | OVERWRITE, NEVER APPEND -->
# HANDOFF

last_verified: 2026-10-27 by `pixi run pytest -q` (238 passed on 18efe98) and `gh pr merge 3` (main b1a79e9)
last_worker: grok
last_machine: 1befd044f2ccc95033f353b2b7468adc
handoff_id: 0a13255a2dbb27a393fe6a0262be71a2
supersedes: none, 8a043208a3a3903f74859447cd5564a5

## Goal
Keep main as the only long-lived branch. The next proof is the CUDA --gpus smoke.

## Last verified state
- PR #3 is merged at b1a79e9. feat/multi-gpu-parallel is deleted locally and on origin.
- v0.3.0 is staged in this commit. Unit tests passed (238). CUDA hardware smoke was not run.
- stash@{0} (linux-support pixi.lock) is dropped. The worktree still has local pixi.lock drift, whisper.cpp pycache, and two repo-root media files.

## Next action
On a Linux NVIDIA machine, run T-0001: Task 7 step 5 in docs/superpowers/plans/2026-06-24-multi-gpu-parallel.md (`SCRIBE_MD_WHISPER_ACCEL=cuda`, `pixi run -e cuda`, `--gpus auto`).

## Gotchas
- This machine is Apple Silicon, so --gpus falls back to sequential.
- Do not commit pixi.lock drift, repo-root media, or vendor/whisper.cpp pycache.
- bash_guard blocks shell writes to governed files.
- Vulkan multi-GPU stays deferred (D-0002).

## Do not touch
vendor/whisper.cpp/, capture/.build/, output/, plans/archive/, and the untracked media files in the repo root.
