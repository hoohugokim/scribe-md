# scribe-md — Development Plan

## Current State (v0.1 — Working Prototype)

### What exists
- **Swift CLI** (`capture/`): System-wide audio capture via ScreenCaptureKit → 48kHz stereo WAV, with chunked output and overlap buffer
- **Python script** (`transcribe.py`): mlx-whisper transcription → Markdown, with JSON chunk format and overlap-aware merge
- **Shell orchestrator** (`transcribe.sh`): Single-file and chunked (pipelined) modes, ffmpeg 48kHz→16kHz conversion
- **Pixi** for Python + mlx-whisper dependency management

### Known issues
1. **Ctrl+C unreliable** — `signal(SIGINT, SIG_IGN)` + DispatchSource pattern doesn't fire when the process inherits SIG_IGN from the parent shell. `trap : INT` fix was applied but not fully verified. Enter key works as the primary stop mechanism.
2. **Audio buffer handling fragile** — Non-interleaved stereo PCM is copied with a single `memcpy` into the first audio buffer. Works by accident (contiguous memory layout of AVAudioPCMBuffer). Should use proper per-channel buffer handling.
3. **Whisper hallucination on silence** — Silent or near-silent audio produces "자막제공자" (subtitle provider). No silence detection or filtering.
4. **No per-app capture** — Original vision was per-app audio isolation. Currently captures all system audio.

---

## Phase 1 — Stabilize Core (Bug Fixes + Reliability)

### 1.1 Fix audio buffer handling
- Detect interleaved vs non-interleaved from ASBD flags (`kAudioFormatFlagIsNonInterleaved`)
- For non-interleaved: split CMBlockBuffer data into separate channel buffers using `UnsafeMutableAudioBufferListPointer`
- Add format logging on first sample to confirm actual layout
- Test: capture → afplay → verify clean audio

### 1.2 Fix Ctrl+C signal handling
- Replace `signal(SIGINT, SIG_IGN)` + DispatchSource with a blocking `sigwait()` on a dedicated thread (same pattern as Enter key handler)
- Or: use `sigaction` with a C-level handler that sets an atomic flag, polled by a thread
- Verify shell `trap : INT` allows child to receive SIGINT
- Test: both Enter (graceful stop → transcribe) and Ctrl+C (cancel → no transcribe)

### 1.3 Silence detection / hallucination guard
- After ffmpeg conversion, check RMS energy of the 16kHz WAV
- If below threshold (e.g., -50 dBFS), skip transcription and warn user
- In chunked mode: skip silent chunks, don't produce empty JSON files
- Consider using Whisper's `no_speech_threshold` parameter if mlx-whisper exposes it

### 1.4 Error handling
- Validate WAV file size > 0 before transcription
- Handle ffmpeg conversion failures gracefully
- Timeout on ScreenCaptureKit permission prompt (inform user to grant access)
- Handle disk-full scenarios during long recordings

---

## Phase 2 — YouTube / URL Transcription

### 2.1 yt-dlp integration
- Add `yt-dlp` to `pixi.toml` dependencies (available on conda-forge)
- New `--url` flag in `transcribe.sh`
- Flow: `yt-dlp -x --audio-format wav` → ffmpeg 16kHz mono → transcribe
- Skip the Swift capture tool entirely for URL mode

### 2.2 Long video support
- For videos > 30 min, automatically use chunked transcription (split with ffmpeg `-ss`/`-t`)
- Reuse existing chunk merge infrastructure
- Progress reporting: estimated time based on audio duration vs. processing speed

### 2.3 Playlist / batch support
- Accept multiple URLs or a playlist URL
- Output one `.md` per video, or a combined file with video titles as headings
- `--batch` flag with a text file of URLs (one per line)

---

## Phase 3 — Per-App Audio Capture

### 3.1 App targeting
- Add `--app <name>` and `--bundle-id <id>` flags to Swift CLI
- Use `SCShareableContent.excludingDesktopWindows` to enumerate running apps
- Create `SCContentFilter` targeting a specific `SCRunningApplication` instead of display-wide capture
- `--list-apps` flag to show currently running apps with audio output

### 3.2 Multi-app capture
- Support `--app "Zoom" --app "Chrome"` to capture from multiple apps simultaneously
- Mix audio streams or output separate channels per app
- Useful for: recording a Zoom meeting while also capturing browser-based presentation audio

---

## Phase 4 — Output Quality & Formatting

### 4.1 Speaker diarization
- Investigate `pyannote-audio` or similar for speaker identification
- Add `--diarize` flag: label segments with `Speaker 1:`, `Speaker 2:`, etc.
- Especially valuable for meeting transcription
- Note: may require PyTorch — evaluate Apple Silicon compatibility and model size

### 4.2 Intelligent formatting
- Paragraph detection: merge segments with short pauses into paragraphs
- Sentence boundary detection: don't break mid-sentence at chunk boundaries
- Configurable timestamp granularity: per-segment (default), per-paragraph, per-minute, or none

### 4.3 Obsidian integration
- `--obsidian-vault <path>` flag: write output directly to vault
- YAML frontmatter: date, source (app name / URL), duration, language, model
- Wikilink format for cross-referencing
- Daily note append mode: add transcription to today's daily note
- Template support: user-defined Markdown template for output structure

### 4.4 Post-processing with LLM (optional)
- `--summarize` flag: pipe transcription through a local LLM (e.g., mlx-lm) for summarization
- `--clean` flag: fix obvious Whisper artifacts (repeated phrases, hallucinated text)
- Keep this optional — core tool stays offline/local without requiring an LLM

---

## Phase 5 — Unified Python CLI

### 5.1 Replace shell orchestrator with Python
- `transcribe.sh` has grown complex (FIFO, signal handling, two code paths)
- Rewrite as a Python CLI using `click` or `typer`
- Manage Swift subprocess, ffmpeg, and Whisper from Python
- Better error handling, progress bars, and structured logging
- Entry point: `pixi run transcribe` (registered in pixi.toml tasks)

### 5.2 Configuration file
- `~/.config/scribe-md/config.toml` or project-local `.scribe-md.toml`
- Default language, model, output directory, Obsidian vault path, chunk settings
- Override per-invocation with CLI flags

### 5.3 Pixi task definitions
- `pixi run capture` — just audio capture
- `pixi run transcribe <file>` — transcribe existing audio
- `pixi run transcribe --url <url>` — YouTube transcription
- `pixi run transcribe --live` — system audio capture + transcription
- `pixi run transcribe --live --app Zoom` — per-app live capture

---

## Phase 6 — Performance & Model Options

### 6.1 Model management
- `--model` presets: `tiny`, `base`, `small`, `medium`, `large-v3` (map to mlx-community HF repos)
- Auto-download on first use with progress bar
- `--list-models` to show available and downloaded models
- Recommend `small` for real-time chunked transcription, `large-v3` for offline/URL mode

### 6.2 Parallel chunk transcription
- Current chunked pipeline is sequential (transcribe chunk N while recording chunk N+1)
- For offline mode (yt-dlp / existing file): parallelize transcription across chunks
- Limit concurrency to avoid ANE contention (2-3 parallel workers max)

### 6.3 Incremental output
- In chunked mode, append to `.md` file as each chunk is transcribed (not just at the end)
- User sees results in real-time (tail -f or Obsidian auto-refresh)
- Final merge pass to clean up overlap artifacts

---

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | 1.1 Fix audio buffers | Small | Correctness |
| P0 | 1.2 Fix Ctrl+C | Small | Usability |
| P0 | 2.1 yt-dlp integration | Small | High — new input source |
| P1 | 1.3 Silence detection | Small | Prevents hallucination |
| P1 | 2.2 Long video chunking | Medium | Needed for 2h+ videos |
| P1 | 4.3 Obsidian integration | Small | User's primary workflow |
| P2 | 3.1 Per-app capture | Medium | Original vision |
| P2 | 5.1 Python CLI rewrite | Medium | Maintainability |
| P2 | 6.3 Incremental output | Small | UX for long recordings |
| P3 | 4.1 Speaker diarization | Large | Meeting use case |
| P3 | 4.4 LLM post-processing | Medium | Nice-to-have |
| P3 | 6.2 Parallel transcription | Medium | Performance |

---

## Tech Stack Summary

| Component | Current | Target |
|-----------|---------|--------|
| Audio capture | Swift + ScreenCaptureKit | Same (extend for per-app) |
| Audio download | — | yt-dlp (via Pixi) |
| Audio conversion | ffmpeg (system) | ffmpeg (via Pixi) |
| Transcription | mlx-whisper (large-v3) | Same + model presets |
| Diarization | — | pyannote-audio (optional) |
| Orchestrator | Bash (transcribe.sh) | Python CLI (click/typer) |
| Dependencies | Pixi | Pixi |
| Output | Markdown | Markdown + Obsidian frontmatter |

---

## File Structure (Target)

```
scribe-md/
  capture/                    # Swift CLI (ScreenCaptureKit)
    Package.swift
    Sources/main.swift
  scribe_md/                  # Python package
    __init__.py
    cli.py                    # Click/Typer CLI entry point
    transcribe.py             # Whisper transcription
    merge.py                  # Chunk merge logic
    audio.py                  # ffmpeg helpers, silence detection
    download.py               # yt-dlp wrapper
    format.py                 # Markdown + Obsidian formatting
  pixi.toml
  .scribe-md.toml             # Default config (example)
```
