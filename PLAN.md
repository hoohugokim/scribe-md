# scribe-md — Development Plan

## Current State (v0.1.1 — cross-platform CLI)

### What exists
- **Swift CLI** (`capture/`): System-wide audio capture via ScreenCaptureKit → 48kHz stereo WAV, with chunked output, overlap buffer, proper per-channel buffer handling, and reliable signal handling (macOS only)
- **Python CLI** (`scribe_md/`): Typer-based CLI with three subcommands (`live`, `url`, `file`), platform-selected transcription backend, yt-dlp YouTube download, overlap-aware chunk merge, silence detection
- **Transcription backends**: mlx-whisper on macOS (Apple Silicon), whisper.cpp on Linux (Vulkan/CUDA/CPU); selected automatically by platform, overridable via `SCRIBE_MD_BACKEND`
- **Pixi** for all dependencies — portable to Apple Silicon Macs and Linux via `git clone` + `pixi install` (see `docs/LINUX.md` for the Linux build)

### Architecture
```
scribe-md/
  capture/                    # Swift CLI (ScreenCaptureKit, macOS)
    Package.swift
    Sources/main.swift
  vendor/whisper.cpp          # git submodule (Linux backend)
  scribe_md/                  # Python package (typer CLI)
    __init__.py
    cli.py                    # Subcommands: live, url, file, config, list-*
    platform_support.py       # OS detection + platform-aware hints
    transcriber.py            # Transcription facade (validates, delegates)
    backends/                 # base protocol + mlx (macOS) + whispercpp (Linux)
    merger.py                 # Overlap-aware chunk merge + timestamp rendering
    postprocess.py            # Rule-based cleaning + optional mlx-lm summary
    diarize.py                # Optional speaker diarization (pyannote)
    obsidian.py               # Frontmatter / daily-note / vault output
    audio.py                  # ffmpeg helpers, silence detection
    downloader.py             # yt-dlp wrapper
    capture.py                # Swift binary management (macOS)
    config.py                 # Layered TOML config loading
    utils.py                  # Shared utilities
  pyproject.toml              # Python package definition
  pixi.toml                   # Dependency management
```

---

## Phase 1 — Stabilize Core ~~(Bug Fixes + Reliability)~~ DONE

### ~~1.1 Fix audio buffer handling~~ DONE
- ~~Detect interleaved vs non-interleaved from ASBD flags (`kAudioFormatFlagIsNonInterleaved`)~~
- ~~For non-interleaved: split CMBlockBuffer data into separate channel buffers using `UnsafeMutableAudioBufferListPointer`~~
- ~~Add format logging on first sample to confirm actual layout~~
- Frame count now uses `CMSampleBufferGetNumSamples()` instead of `dataLength / bytesPerFrame`

### ~~1.2 Fix Ctrl+C signal handling~~ DONE
- ~~Replace `signal(SIGINT, SIG_IGN)` + DispatchSource with `sigprocmask(SIG_BLOCK)` + `sigwait()` on a dedicated thread~~
- Handles inherited SIG_IGN from parent by resetting to SIG_DFL after blocking

### ~~1.3 Silence detection / hallucination guard~~ DONE
- ~~After ffmpeg conversion, check RMS energy of the 16kHz WAV~~
- Pre-transcription: `audio.is_silent()` uses ffmpeg `volumedetect` (threshold -50 dBFS)
- Post-transcription: `extract_segments()` filters segments with `no_speech_prob > 0.6`
- Applied in all pipelines: single-file, chunked, and live chunked

### ~~1.4 Error handling~~ DONE
- ~~Validate WAV file size > 0 before transcription~~
- ~~Handle ffmpeg conversion failures gracefully~~ — `AudioConversionError` with actionable messages
- ~~Timeout on ScreenCaptureKit permission prompt~~ — `_check_capture_permission()` with 10s timeout
- ~~Handle disk-full scenarios~~ — `DiskFullError` + `_check_disk_space()` pre-write checks

---

## Phase 2 — ~~YouTube / URL Transcription~~ DONE

### ~~2.1 yt-dlp integration~~ DONE
- ~~Add `yt-dlp` to `pixi.toml` dependencies~~
- `scribe-md url <URL>` subcommand
- Flow: yt-dlp download → ffmpeg 16kHz mono → transcribe

### ~~2.2 Long video support~~ DONE
- ~~For videos > 30 min, automatically use chunked transcription (split with ffmpeg `-ss`/`-t`)~~
- Configurable via `--chunk-seconds` (default 1800s)
- Reuses chunk merge infrastructure

### ~~2.3 Playlist / batch support~~ DONE
- Playlist detection and iteration (one `.md` per video)
- Output filename derived from video title

---

## Phase 3 — ~~Per-App Audio Capture~~ DONE

### ~~3.1 App targeting~~ DONE
- ~~Add `--app <name>` and `--bundle-id <id>` flags to Swift CLI~~
- ~~Use `SCShareableContent.excludingDesktopWindows` to enumerate running apps~~
- ~~Create `SCContentFilter` targeting a specific `SCRunningApplication` instead of display-wide capture~~
- ~~`--list-apps` flag to show currently running apps~~
- `scribe-md list-apps` subcommand and `scribe-md live --app <name>` in Python CLI
- App name matching: exact match first, then case-insensitive substring

### ~~3.2 Multi-app capture~~ DONE
- ~~Support `--app "Zoom" --app "Chrome"` to capture from multiple apps simultaneously~~
- `SCContentFilter(display:including:exceptingWindows:[])` with array of matched apps
- `scribe-md live --app Zoom --app Chrome` (repeatable flag)

---

## Phase 4 — Output Quality & Formatting

### ~~4.1 Speaker diarization~~ DONE
- ~~Investigate `pyannote-audio` or similar for speaker identification~~ — pyannote-audio v4.x (CPU on Apple Silicon)
- ~~Add `--diarize` flag: label segments with `Speaker 1:`, `Speaker 2:`, etc.~~ — `--diarize`, `--hf-token`, `--num-speakers`
- Especially valuable for meeting transcription
- Note: pyannote-audio is an optional dependency (`pip install pyannote-audio`), requires HuggingFace token

### ~~4.2 Intelligent formatting~~ DONE
- ~~Paragraph detection: merge segments with short pauses into paragraphs~~ — `--paragraph-gap` (default 2.0s)
- ~~Sentence boundary detection~~ — `_find_sentence_boundary()` prefers sentence-ending punctuation in overlap regions
- ~~Configurable timestamp granularity~~ — `--timestamp-mode segment|paragraph|minute|none`

### ~~4.3 Obsidian integration~~ DONE
- ~~`--vault <path>` flag: write output directly to vault~~
- ~~YAML frontmatter: date, source, duration, language, model, tags~~
- ~~Daily note append mode: `--daily-note` appends `## Transcription (HH:MM)` section~~
- `--frontmatter/--no-frontmatter` flag (default: on when vault is set)

### ~~4.4 Post-processing with LLM (optional)~~ DONE
- ~~`--summarize` flag: pipe transcription through a local LLM (e.g., mlx-lm) for summarization~~
- ~~`--clean` flag: fix obvious Whisper artifacts (repeated phrases, hallucinated text)~~
- Keep this optional — core tool stays offline/local without requiring an LLM

---

## Phase 5 — ~~Unified Python CLI~~ DONE

### ~~5.1 Replace shell orchestrator with Python~~ DONE
- ~~Rewrite as a Python CLI using typer~~
- Three subcommands: `scribe-md live`, `scribe-md url`, `scribe-md file`
- Entry point: `pixi run scribe-md` (registered via pyproject.toml `[project.scripts]`)
- Old `transcribe.sh` and `transcribe.py` deleted

### ~~5.2 Configuration file~~ DONE
- ~~`~/.config/scribe-md/config.toml` or project-local `.scribe-md.toml`~~
- ~~Default language, model, output directory, chunk settings~~
- ~~Override per-invocation with CLI flags~~
- `scribe-md config show|path|init` subcommands

---

## Phase 6 — Performance & Model Options

### ~~6.1 Model management~~ DONE
- ~~`--model` presets: `tiny`, `base`, `small`, `medium`, `large-v3` (map to mlx-community HF repos)~~
- `scribe-md list-models` subcommand shows all presets with default marker
- `resolve_model()` maps short names to full HF repo paths
- Default: `large-v3` (mlx-community/whisper-large-v3-mlx)

### 6.2 Parallel chunk transcription — NOT PURSUED (sequential by design)
- Investigated, then deliberately dropped: mlx-whisper saturates the GPU with a
  single inference and parallel Metal command buffers crash on Apple Silicon, so
  chunks are transcribed **sequentially**. There is no `--workers`/`--no-parallel`
  flag.
- The whisper.cpp (Linux) backend runs as a subprocess and could in principle
  parallelize, but currently inherits the same sequential path. (See the
  performance notes — a potential future improvement.)

### ~~6.3 Incremental output~~ DONE
- ~~In chunked mode, append to `.md` file as each chunk is transcribed~~
- ~~User sees results in real-time (tail -f or Obsidian auto-refresh)~~
- ~~Final merge pass overwrites with clean deduped result~~
- `--incremental/--no-incremental` flag (default: on for live, off for file/url)

---

## Implementation Priority

| Priority | Item | Effort | Status |
|----------|------|--------|--------|
| ~~P0~~ | ~~1.1 Fix audio buffers~~ | ~~Small~~ | DONE |
| ~~P0~~ | ~~1.2 Fix Ctrl+C~~ | ~~Small~~ | DONE |
| ~~P0~~ | ~~2.1 yt-dlp integration~~ | ~~Small~~ | DONE |
| ~~P1~~ | ~~1.3 Silence detection~~ | ~~Small~~ | DONE |
| ~~P1~~ | ~~1.4 Error handling~~ | ~~Small~~ | DONE |
| ~~P1~~ | ~~2.2 Long video chunking~~ | ~~Medium~~ | DONE |
| ~~P1~~ | ~~4.3 Obsidian integration~~ | ~~Small~~ | DONE |
| ~~P2~~ | ~~3.1 Per-app capture~~ | ~~Medium~~ | DONE |
| ~~P2~~ | ~~3.2 Multi-app capture~~ | ~~Small~~ | DONE |
| ~~P2~~ | ~~4.2 Intelligent formatting~~ | ~~Small~~ | DONE |
| ~~P2~~ | ~~5.1 Python CLI rewrite~~ | ~~Medium~~ | DONE |
| ~~P2~~ | ~~5.2 Configuration file~~ | ~~Small~~ | DONE |
| ~~P2~~ | ~~6.1 Model management~~ | ~~Small~~ | DONE |
| ~~P2~~ | ~~6.3 Incremental output~~ | ~~Small~~ | DONE |
| ~~P3~~ | ~~4.1 Speaker diarization~~ | ~~Large~~ | DONE |
| ~~P3~~ | ~~4.4 LLM post-processing~~ | ~~Medium~~ | DONE |
| P3 | 6.2 Parallel transcription | Medium | NOT PURSUED (sequential by design) |

---

## Tech Stack Summary

| Component | Status |
|-----------|--------|
| Audio capture | Swift + ScreenCaptureKit (system-wide, macOS) |
| Audio download | yt-dlp (via Pixi conda-forge) |
| Audio conversion | ffmpeg (via Pixi conda-forge) |
| Transcription | mlx-whisper (macOS) / whisper.cpp (Linux), `large-v3` default |
| Orchestrator | Python CLI (typer) |
| Dependencies | Pixi (conda-forge + PyPI) |
| Output | Markdown with optional timestamps |

---

## Setup

### macOS (Apple Silicon)

```bash
# Prerequisites
xcode-select --install          # Swift toolchain (for live capture)
curl -fsSL https://pixi.sh/install.sh | bash  # pixi

# Install
git clone <repo-url> && cd scribe-md
pixi install

# Use
pixi run scribe-md url "https://youtube.com/watch?v=..."
pixi run scribe-md live -l ko -o meeting.md
pixi run scribe-md file recording.wav -o output.md
```

### Linux

`live` (system-audio capture) is macOS-only; use `file`/`url`. Transcription
runs on whisper.cpp, built on first use with auto-detected GPU acceleration.
See `docs/LINUX.md` for details.

```bash
git clone <repo-url> && cd scribe-md
pixi install
pixi run build-whisper          # one-time: build whisper.cpp (Vulkan/CUDA/CPU)
pixi run scribe-md url "https://youtube.com/watch?v=..."
```
