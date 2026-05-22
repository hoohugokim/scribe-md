---
# Running scribe-md on Linux

Supported on Pop!_OS 24.04 / Ubuntu 24.04 for the `file` and `url` commands.
`live` capture and `--summarize` are macOS-only for now.

## Install

```bash
git clone <repo-url> && cd scribe-md
git submodule update --init vendor/whisper.cpp
pixi install
pixi run build-whisper      # builds whisper.cpp with auto-detected GPU backend
pixi run scribe-md file recording.wav
```

`build-whisper` auto-detects the accelerator: **Vulkan** if available (the
default for both AMD and NVIDIA), **CUDA** if the toolkit is present, else
**CPU**. Force a choice with `SCRIBE_MD_WHISPER_ACCEL=vulkan|cuda|cpu`.

## GPU notes

- **AMD (e.g. RX 5700 XT, RDNA1):** Vulkan via Mesa RADV is the reliable path.
  ROCm is not officially supported on RDNA1 and is not used here.
- **NVIDIA:** Vulkan works out-of-the-box. For peak performance, install the
  CUDA toolkit via the opt-in environment: `pixi install -e cuda` then
  `SCRIBE_MD_WHISPER_ACCEL=cuda pixi run build-whisper`.
- Confirm the device in use: scribe-md logs `Transcribing ... via whisper.cpp (vulkan)`.

## Benchmark (Vulkan vs CPU)

Time a fixed clip on each accelerator to get real numbers on your hardware:

```bash
rm -rf vendor/whisper.cpp/build
SCRIBE_MD_WHISPER_ACCEL=vulkan pixi run build-whisper
time SCRIBE_MD_WHISPER_ACCEL=vulkan pixi run scribe-md file sample.wav -m small

rm -rf vendor/whisper.cpp/build
SCRIBE_MD_WHISPER_ACCEL=cpu pixi run build-whisper
time SCRIBE_MD_WHISPER_ACCEL=cpu pixi run scribe-md file sample.wav -m small
```

## Known limitations on Linux

- `scribe-md live` -> "Live system-audio capture is macOS-only for now."
- `--summarize` -> "Summarization (mlx-lm) is macOS-only for now."
- `--diarize` (pyannote, CPU) is expected to work but is not verified on Linux.
---
