# Qwen + Nemotron Engines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qwen3-ASR (macOS) and Nemotron 3.5 Transcribe (Linux CUDA default-when-available) as first-class scribe-md backends beside Whisper.

**Architecture:** Extend `backends.get_backend()` with an engine override + auto policy. New `QwenBackend` (mlx-audio) and `NemotronBackend` (OpenAI HTTP client to local vLLM). Normalize all outputs to the existing segment dict shape.

**Tech Stack:** Python ≥3.12, Typer, pytest, optional `mlx-audio` (macOS), stdlib/`urllib` or `httpx` for Nemotron HTTP (prefer stdlib to avoid new hard deps).

**Spec:** `docs/superpowers/specs/2026-10-29-qwen-nemotron-engines-design.md`

## Global Constraints

- Pixi is the only env manager; optional engines must not break Linux `pixi install`.
- Backend modules stay lazy-imported.
- One engine per job; no ensemble.
- `--gpus` remains whisper.cpp CUDA-only in v1.
- Do not edit `vendor/whisper.cpp`, `HANDOFF.md` without overwrite discipline, or hand-edit `CHANGELOG.md`.

## File map

| File | Responsibility |
|---|---|
| `scribe_md/backends/base.py` | Unchanged protocol |
| `scribe_md/backends/__init__.py` | Selection + auto policy |
| `scribe_md/backends/qwen.py` | MLX Qwen3-ASR adapter |
| `scribe_md/backends/nemotron.py` | vLLM OpenAI client + health check |
| `scribe_md/backends/segments.py` | Shared text/timestamp → segments helpers |
| `scribe_md/config.py` | `[defaults].engine`, `[qwen]`, `[nemotron]` |
| `scribe_md/cli.py` | `--engine`, list-models notes |
| `scribe_md/transcriber.py` | Pass-through only if needed |
| `tests/test_backends.py` | Selection matrix |
| `tests/test_qwen.py` / `tests/test_nemotron.py` | Adapter tests |
| `README.md`, `docs/LINUX.md`, `AGENTS.md` | User-facing docs |

---

### Task 1: Shared segment helpers + selection policy

**Files:**
- Create: `scribe_md/backends/segments.py`
- Modify: `scribe_md/backends/__init__.py`
- Test: `tests/test_backends.py`, `tests/test_segments.py`

- [ ] Write failing tests for `segments_from_text` and engine selection (`qwen`/`nemotron` overrides, auto nemotron when healthy, fallback when not)
- [ ] Implement helpers + `get_backend(engine=None)` selection
- [ ] Run `pixi run pytest tests/test_backends.py tests/test_segments.py -q`
- [ ] Commit

### Task 2: Nemotron HTTP backend

**Files:**
- Create: `scribe_md/backends/nemotron.py`
- Test: `tests/test_nemotron.py`
- Modify: `scribe_md/config.py` for `[nemotron]`

- [ ] Failing tests with mocked `urlopen`
- [ ] Implement health check + chat-completions transcription + segment normalize
- [ ] Wire config fields
- [ ] Run pytest for nemotron + backends
- [ ] Commit

### Task 3: Qwen MLX backend

**Files:**
- Create: `scribe_md/backends/qwen.py`
- Test: `tests/test_qwen.py`
- Modify: `scribe_md/config.py` for `[qwen]`

- [ ] Failing tests with stubbed `mlx_audio`
- [ ] Implement adapter + presets; clear error on non-macOS
- [ ] Run pytest
- [ ] Commit

### Task 4: CLI `--engine` + docs

**Files:**
- Modify: `scribe_md/cli.py`, `README.md`, `docs/LINUX.md`, `AGENTS.md`
- Test: CLI option resolves via existing config merge tests / new cli test

- [ ] Add `--engine` to file/url/live shared options
- [ ] Document engines and Nemotron server setup
- [ ] Full `pixi run pytest -q`
- [ ] Commit
