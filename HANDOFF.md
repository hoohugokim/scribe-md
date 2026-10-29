<!-- statutor: plane=state | policy=overwrite_bounded (max 40 lines) | writer=executor | OVERWRITE, NEVER APPEND -->
# HANDOFF

last_verified: 2026-10-29 by `pixi run pytest -q` (259 passed) on cursor/qwen-nemotron-engines-77b7
last_worker: composer
last_machine: cloud-agent
handoff_id: qwen-nemotron-77b7
supersedes: bf7766637d0547f375f8a9ed93f320e7

## Goal
Land Qwen3-ASR (macOS) and Nemotron 3.5 Transcribe (Linux CUDA default-when-available) engines (D-0005).

## Last verified state
- Branch cursor/qwen-nemotron-engines-77b7: backends qwen/nemotron/segments, --engine, config, docs, D-0005.
- Auto policy: macOS mlx; Linux nemotron if CUDA + /v1/models healthy else whispercpp.
- pytest 259 passed (stubbed; no live vLLM / mlx-audio inference in CI).

## Next action
Review/merge the PR. Optional follow-ups: live Nemotron smoke on H100, mlx-audio install path in pixi, Qwen as optional Mac default.

## Gotchas
- Nemotron is client-only; user must run vLLM + NeMo Speech separately.
- --gpus remains whisper.cpp CUDA-only.
- Do not commit pixi.lock drift, repo-root media, or whisper.cpp pycache.

## Do not touch
vendor/whisper.cpp/, capture/.build/, output/, plans/archive/, and the untracked media files in the repo root.
