# Qwen3-ASR + Nemotron 3.5 Transcribe engines for scribe-md

**Date:** 2026-10-29 (Asia/Seoul)
**Branch:** `cursor/qwen-nemotron-engines-77b7`
**Status:** Design — approved direction (Qwen + Nemotron); implementing

## Goal

Add two first-class transcription engines beside the existing Whisper stack:

1. **Qwen3-ASR 1.7B** on macOS via MLX (`mlx-audio`) — selectable accuracy engine, especially strong for KO/CJK.
2. **NVIDIA Nemotron 3.5 Transcribe** (`NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`, ~4.3B active) on Linux with NVIDIA CUDA — **auto default when available**.

Whisper remains: MLX Whisper on macOS when engine is unset / `mlx`; whisper.cpp on Linux when Nemotron is unavailable or not selected.

## Decisions already locked

| Decision | Choice |
|---|---|
| Dual-default semantics | Selectable co-defaults; one engine per job |
| Nemotron availability | Linux + NVIDIA CUDA; macOS is a no-go (no CUDA / no MLX port of 30B-A4B) |
| Nemotron default rule | Auto-select on Linux when CUDA is usable **and** a local Nemotron server is reachable |
| Qwen role | First-class Mac engine (`--engine qwen`); Whisper stays the Mac auto-default for language breadth |
| Nemotron runtime | OpenAI-compatible **client** to a local vLLM + NeMo Speech server (not in-process Transformers in v1) |
| Result shape | Every backend still returns mlx-whisper-shaped `{"segments": [...]}` |

## Scope

**In scope:**
- New backends: `scribe_md/backends/qwen.py`, `scribe_md/backends/nemotron.py`
- Engine selection: `--engine`, `[defaults] engine`, `SCRIBE_MD_BACKEND` (aliases: `mlx`, `whispercpp`, `qwen`, `nemotron`)
- Auto policy (when engine unset):
  - macOS → `mlx` (Whisper)
  - Linux + CUDA present + Nemotron health check OK → `nemotron`
  - else Linux → `whispercpp`
- Config for Nemotron: base URL, model id, optional prompt preset, timeout
- Config for Qwen: default HF / mlx-community model id (1.7B-8bit)
- Map Qwen / Nemotron outputs into segment dicts (`start`, `end`, `text`, `no_speech_prob`)
- `list-models` / help / README / `docs/LINUX.md` updates
- Unit tests with stubs (no real GPU / no live vLLM required in CI)

**Out of scope (deferred):**
- In-process Nemotron Transformers / GGUF NeMo-Speech.cpp (GGUF for Transcribe still maturing)
- Making Qwen the macOS auto-default
- Ensemble / dual-run merge
- Replacing pyannote with Nemotron’s built-in speaker attribution (can follow later via prompts)
- Shipping or managing the vLLM + NeMo Speech server process inside pixi (document how to run it; client-only in scribe-md)
- Windows

## Architecture

```
CLI / config / SCRIBE_MD_BACKEND
        │
        ▼
 backends.get_backend()   ← auto policy + explicit override
        │
        ├── MLXBackend        (mlx-whisper)     macOS default
        ├── QwenBackend       (mlx-audio STT)   macOS, --engine qwen
        ├── WhisperCppBackend (whisper-cli)     Linux fallback
        └── NemotronBackend   (OpenAI HTTP)     Linux CUDA default when server up
```

All paths still go through `transcriber.transcribe_audio` → `extract_segments`. Chunking, merge, Obsidian, diarize, and scheduler stay engine-agnostic except where CUDA `--gpus` already requires whisper.cpp.

### Nemotron client contract

- Default base URL: `http://127.0.0.1:8000/v1`
- Health: `GET {base}/models` (or `/health` if exposed); failure → auto policy falls back to whisper.cpp (unless user explicitly set `--engine nemotron`, which errors with setup hints)
- Request: OpenAI chat completions with audio (base64 WAV) + transcription prompt (`temperature=0`, thinking off)
- Response text → one or more segments. Prefer word/segment timestamps when the server returns them via CTC middleware; otherwise a single segment spanning `[0, duration]` with `no_speech_prob=0.0`
- Multi-GPU `--gpus` stays whisper.cpp-only in v1; Nemotron relies on the vLLM server’s own parallelism

### Qwen client contract

- Optional dependency: `mlx-audio` (macOS), same pattern as optional `mlx-lm` / `pyannote-audio`
- Default model: `mlx-community/Qwen3-ASR-1.7B-8bit`
- Presets: `qwen` / `qwen-1.7b` → 1.7B-8bit; `qwen-0.6b` → 0.6B-8bit
- Load via `mlx_audio.stt` and normalize to segments; if the library returns only full text, emit one segment `[0, duration]`
- On Linux, `--engine qwen` fails with a clear “macOS / Apple Silicon only” message

## CLI / config surface

```toml
[defaults]
engine = ""                 # empty = auto policy; or mlx|qwen|whispercpp|nemotron
model = "large-v3"          # Whisper presets; ignored for Nemotron server model id
# ...

[qwen]
model = "mlx-community/Qwen3-ASR-1.7B-8bit"

[nemotron]
base_url = "http://127.0.0.1:8000/v1"
model = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
prompt = "Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only."
timeout_seconds = 600
```

```bash
scribe-md file talk.wav --engine qwen -l ko
scribe-md file talk.wav --engine nemotron
SCRIBE_MD_BACKEND=nemotron scribe-md file talk.wav
```

`--model` continues to mean Whisper (and Qwen) weight selection. Nemotron’s served model id comes from `[nemotron].model` / env, not Whisper presets.

## Testing

- Extend `tests/test_backends.py` for selection matrix (auto + overrides + fallback when Nemotron unhealthy)
- New `tests/test_qwen.py`, `tests/test_nemotron.py` with stubbed mlx-audio / HTTP
- Segment normalization unit tests (text-only → single segment; timestamp payloads → multi-segment)
- No CI requirement for real CUDA, vLLM, or Apple Silicon inference

## Docs

- README: engines table, `--engine`, Mac Qwen install note, Nemotron server quick-start link
- `docs/LINUX.md`: Nemotron as CUDA default-when-available; how to start vLLM + NeMo Speech; fallback behavior
- AGENTS.md: one-line backend list update

## Risks

- Nemotron BF16 wants ~H100 80GB; users on smaller GPUs need FP8/NVFP4 later — document hardware expectation
- mlx-audio STT API may differ across versions — pin a known API in the adapter and test with stubs
- Auto-default probing a local server adds a small startup latency; keep health check short (≤1s timeout)
