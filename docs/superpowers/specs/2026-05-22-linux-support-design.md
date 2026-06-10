# Linux support for scribe-md (file + url, whisper.cpp + Vulkan/CUDA)

**Date:** 2026-05-22
**Branch:** `linux-support`
**Status:** Design — approved for spec write-up

## Goal

Make scribe-md run on Linux — specifically Pop!\_OS 24.04 LTS (which also covers
Ubuntu 24.04) — for the `file` and `url` commands, from a **single codebase**
that auto-detects the platform. Transcription on Linux uses **whisper.cpp** with
GPU acceleration via **Vulkan by default** (works on both AMD and NVIDIA), with
**CUDA as an opt-in** for NVIDIA peak performance, and CPU as the fallback.

The immediate target machine has an **AMD RX 5700 XT** (Navi 10 / gfx1010 /
RDNA1). Hermes Agent automation is explicitly **out of scope** for this work.

## Scope

**In scope (Linux):**
- `scribe-md file` — transcribe local audio files.
- `scribe-md url` — download + transcribe YouTube videos/playlists.
- whisper.cpp transcription backend with CPU / Vulkan / CUDA build-time accel.
- Single codebase, platform auto-detection (macOS keeps MLX, Linux uses whisper.cpp).
- Packaging so `pixi install` succeeds on Linux (MLX must not be pulled in).
- Graceful, non-crashing messages for macOS-only features when run on Linux.

**Out of scope:**
- `scribe-md live` (system-audio capture) on Linux — would need a whole new
  PipeWire/PulseAudio backend to replace the macOS Swift/ScreenCaptureKit binary.
- `--summarize` on Linux (depends on `mlx-lm`, Apple-only).
- `--diarize` on Linux — pyannote-audio is cross-platform CPU and *should* work
  as-is, but is not actively ported or verified here. Left enabled, marked
  untested-on-Linux.
- Hermes Agent automation.

## Background: why these engine choices

The RX 5700 XT is RDNA1 (`gfx1010`), which AMD's ROCm has never officially
supported for compute. The common `HSA_OVERRIDE_GFX_VERSION=10.3.0` workaround
compiles kernels for `gfx1030` (RDNA2 — a *different* instruction set); on RDNA1
this frequently crashes or returns incorrect output. So ROCm is not a reliable
baseline on this card.

Engine comparison for whisper transcription on this hardware:

| Path | GPU access on RX 5700 XT | Reliability | Notes |
|---|---|---|---|
| whisper.cpp + Vulkan | Mesa RADV | Solid | First-class RDNA1 driver; no ROCm needed. |
| PyTorch + ROCm | ROCm/HIP | Fragile | gfx1010 unsupported; override targets wrong ISA. |
| faster-whisper (CTranslate2) | none (CUDA-only) | n/a on AMD | CPU-only on AMD. |

**Vulkan vs ROCm performance:** if ROCm worked, whisper.cpp's HIP path (shared
CUDA kernels) would likely beat Vulkan by roughly ~1.3–2× on the encoder (RDNA1
lacks the cooperative-matrix Vulkan extensions that accelerate RDNA3+). But that
headroom is largely inaccessible on this card because ROCm doesn't reliably run
on RDNA1. Vulkan via RADV is correct and stable today. Real numbers will be
measured on-hardware via a benchmark step (see Testing).

Because whisper.cpp is driven as a subprocess, the GPU engine (CPU / Vulkan /
CUDA) is purely a **build-time flag** of the same `whisper-cli` binary producing
identical JSON — so all three are one Python backend, and a Vulkan/CUDA crash is
isolated in a separate process (the failure mode previously hit with Metal,
commit 50786d7).

## Architecture

### 1. Backend abstraction

Today `transcriber.py` calls `mlx_whisper.transcribe()` directly. Introduce a
backend layer selected by platform:

```
scribe_md/backends/
  __init__.py      # get_backend() -> Backend; platform detection + env override
  base.py          # Backend protocol: .transcribe(wav, model, language) -> result dict
                   #                    .resolve_model(name) -> backend-specific ref
                   #                    .describe() -> human string for logs (engine + device)
  mlx.py           # MLXBackend — current mlx-whisper logic moves here (macOS/arm64)
  whispercpp.py    # WhisperCppBackend — subprocess whisper-cli; CPU/Vulkan/CUDA
```

- `transcriber.py` becomes a thin facade: it keeps the input validation
  (file-exists / size / >=44-byte checks) and `extract_segments()` **unchanged**,
  then delegates to `get_backend().transcribe(...)`.
- **Both backends return the same dict shape**
  `{"segments": [{"start", "end", "text", "no_speech_prob"}]}` so
  `extract_segments()` and all downstream code (merger, diarize, obsidian) is
  untouched. (whisper.cpp segments lack `no_speech_prob`; default it to `0.0`,
  which is below the 0.6 filter threshold, so no segments are wrongly dropped.)
- **Selection logic:**
  - `darwin` + arm64 → `MLXBackend`
  - `linux` → `WhisperCppBackend`
  - `SCRIBE_MD_BACKEND=mlx|whispercpp` env var overrides for testing.
  - Any other platform → clear NotImplemented-style error.
- **Model presets stay platform-stable.** `MODEL_PRESETS` names
  (`tiny`…`large-v3`, `large-v3-turbo`) remain the user-facing vocabulary; each
  backend's `resolve_model()` maps a name to its own artifact (MLX → HF mlx repo;
  whisper.cpp → GGML `.bin`).

### 2. whisper.cpp + Vulkan/CUDA backend

- **Vendored** as a git submodule at `vendor/whisper.cpp`, pinned to a release tag.
- **Build:** `pixi run build-whisper` runs cmake, producing
  `vendor/whisper.cpp/build/bin/whisper-cli`. Mirrors the existing
  `build-capture` Swift task. `ensure_whisper_binary()` (modeled on
  `ensure_capture_binary()` in `capture.py`) builds on first run and raises a
  clear `WhisperCppError` if the toolchain (cmake / compiler / Vulkan) is missing.
- **Accelerator auto-detection** (at build time):
  - CUDA toolkit (`nvcc`) **and** `nvidia-smi` present → `-DGGML_CUDA=1`
  - else Vulkan available (`libvulkan` / `vulkaninfo`) → `-DGGML_VULKAN=1`
  - else CPU (no accel flag)
  - `SCRIBE_MD_WHISPER_ACCEL=cuda|vulkan|cpu` forces a choice.
- **Invoke:**
  `whisper-cli -m <model.bin> -f <wav> -oj -of <prefix> -l <auto|lang> --no-prints`
  then parse `<prefix>.json` into segment dicts. Input is already 16 kHz mono
  s16 WAV (produced by `audio.convert_to_16k_mono`, `audio.py:32`), which is
  exactly whisper.cpp's required format — no resampling work.
- **Active device reporting:** parse `whisper-cli`'s startup/stderr log (which
  names the backend + device) so `WhisperCppBackend.describe()` can report e.g.
  "whisper.cpp (Vulkan: AMD Radeon RX 5700 XT)" in scribe-md's own logs.
- **Models:** GGML `.bin` files auto-downloaded from
  `huggingface.co/ggerganov/whisper.cpp` to `~/.cache/scribe-md/models/` on first
  use, with a "downloading model…" log line. Already-present files are reused.

### 3. Packaging / dependency restructuring

This is what makes `pixi install` actually succeed on Linux.

- `pixi.toml`: `platforms = ["osx-arm64", "linux-64"]`.
- **MLX is Apple-only — must be gated** or Linux installs fail (no MLX wheels for
  Linux). In `pyproject.toml`:
  - `mlx-whisper; sys_platform == "darwin"`
  - move `mlx-lm` to an Apple-only optional/marker as well.
- `[target.linux-64.dependencies]` (default = Vulkan toolchain, from conda-forge
  where possible): `cmake`, `cxx-compiler`, `shaderc`/`glslang`, Vulkan headers +
  loader. The **runtime GPU driver is the system Mesa RADV ICD** (preinstalled on
  Pop!\_OS) — documented, not vendored.
  - **Implication:** NVIDIA users also get GPU acceleration out-of-the-box via
    Vulkan with **zero CUDA toolkit**.
- **CUDA is an opt-in pixi feature/environment** that pulls the heavy
  `cuda-toolkit` from conda-forge, so AMD users never download it. Default Linux
  environment = Vulkan.
- `[target.linux-64.tasks]`: `build-whisper`.
- ffmpeg stays a cross-platform dependency (already in `[dependencies]`).

### 4. Graceful degradation on Linux

"Compatible" means no confusing crashes when a macOS-only path is invoked:

- `live` → clear message: *"Live system-audio capture is macOS-only for now."*
  (no attempt to build the Swift binary).
- `--summarize` → clear message: *"Summarization (mlx-lm) is macOS-only for now."*
- `--diarize` → left enabled (pyannote is cross-platform CPU), documented as
  untested-on-Linux.
- Platform-aware install hints: `apt install ffmpeg` instead of
  `brew install ffmpeg` in error text (`audio.py` and elsewhere).

### 5. Testing & delivery

- **Branch:** `linux-support`, so the work can be pulled and tested on the
  RX 5700 XT machine.
- **Unit tests (no GPU/binary required):**
  - whisper.cpp JSON → segment-dict parser (fixture JSON; accel-agnostic, so it
    covers CPU/Vulkan/CUDA equally).
  - backend selection logic (mock `sys.platform`, exercise `SCRIBE_MD_BACKEND`).
  - per-backend `resolve_model()` mapping.
  - graceful-degradation messages for `live`/`--summarize` on Linux.
  - existing macOS tests keep passing.
- **Manual on-hardware verification** (documented checklist):
  `pixi install` → `pixi run build-whisper` → `scribe-md file sample.wav`,
  confirming the reported device is the GPU.
- **Benchmark step:** time a fixed clip on Vulkan vs CPU (via `whisper-cli`) to
  produce real Vulkan-vs-CPU numbers on the 5700 XT.
- **Docs:** README gains a Linux section (install, build-whisper, GPU notes,
  CUDA opt-in, known limitations).

## Risks / things to flag

- **Vulkan path is unverifiable here.** Build + invocation will be made correct,
  but real GPU validation happens on the user's hardware via the test checklist.
- **CUDA path is unverifiable here** as well (no NVIDIA hardware in dev); covered
  by the accel-agnostic JSON tests + documented manual check.
- **Diarization on Linux** is reachable but outside scope — left as-is rather
  than verified.
