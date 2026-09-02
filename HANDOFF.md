<!-- statutor: plane=state | policy=overwrite_bounded (max 40 lines) | writer=executor | OVERWRITE, NEVER APPEND -->
# HANDOFF

last_verified: 2026-09-02 by `statutor-doctor` + `git status` (branch feat/multi-gpu-parallel @ 394e055)

## Goal
Enable multi-agent development on feat/multi-gpu-parallel (batch + --gpus parallel)

## Last verified state
- Ledger initialized: .statutor.yaml + AGENTS.md (28L) + CLAUDE.md import; .opencode corruption removed
- Global plugin statutor@0.1.0 active; `statutor-doctor` passes (no errors)
- Branch clean except pixi.lock + vendor/whisper.cpp untracked content

## Next action
Orchestrator: seed TASKS.md queue, run `pixi run pytest -q` to baseline, then dispatch workers

## Gotchas
- bash_guard: shell writes to governed files blocked — use edit/write tools
- --gpus parallel is CUDA-only; Vulkan/MLX run sequential (scheduler.py/gpu.py)
- vendor/whisper.cpp is submodule — never hand-edit

## Do not touch
- vendor/whisper.cpp/, capture/.build/, output/*.md, plans/archive/*
