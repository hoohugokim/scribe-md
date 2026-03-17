# scribe-md

Transcribe system audio, audio files, and YouTube videos to Markdown -- fully local on Apple Silicon.

Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast, on-device transcription with no cloud dependencies.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- [Pixi](https://pixi.sh) package manager
- Xcode Command Line Tools (for live capture only)

## Installation

```bash
# Install pixi if you don't have it
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install
git clone <repo-url> && cd scribe-md
pixi install

# Build the Swift audio capture binary (needed for live capture only)
pixi run build-capture
```

Verify the installation:

```bash
pixi run scribe-md --help
```

## Quick Start

```bash
# Transcribe a local audio file
pixi run scribe-md file recording.wav

# Transcribe a YouTube video
pixi run scribe-md url "https://youtube.com/watch?v=..."

# Capture and transcribe system audio (press Ctrl+C to stop)
pixi run scribe-md live
```

## Commands

### `scribe-md file`

Transcribe an existing audio file (WAV, MP3, M4A, FLAC, etc.) to Markdown.

```bash
scribe-md file recording.wav
scribe-md file meeting.m4a -o meeting.md -l ko
scribe-md file lecture.mp3 --model small --no-timestamps
```

### `scribe-md url`

Download and transcribe audio from a YouTube video or playlist.

```bash
scribe-md url "https://youtube.com/watch?v=dQw4w9WgXcQ"
scribe-md url "https://youtube.com/playlist?list=PLxxx" -l en
```

For playlists, each video is transcribed to its own `.md` file, named after the video title.

### `scribe-md live`

Capture system audio in real-time and transcribe when done.

```bash
# Capture all system audio
scribe-md live -o meeting.md

# Capture for a fixed duration (seconds)
scribe-md live -d 300 -o five-minutes.md

# Capture audio from specific app(s)
scribe-md live --app Zoom
scribe-md live --app Zoom --app Chrome

# Chunked live mode: transcribe every 5 minutes
scribe-md live --chunk-seconds 300 --overlap-seconds 5
```

The first time you run `live`, macOS will ask for screen recording permission (required by ScreenCaptureKit for audio capture).

### `scribe-md list-models`

Show available Whisper model presets:

```
Preset               HF Repo Path
──────────────────── ──────────────────────────────────────────────────
tiny                 mlx-community/whisper-tiny-mlx
base                 mlx-community/whisper-base-mlx
small                mlx-community/whisper-small-mlx
medium               mlx-community/whisper-medium-mlx
large                mlx-community/whisper-large-v3-mlx
large-v3             mlx-community/whisper-large-v3-mlx          *
large-v3-turbo       mlx-community/whisper-large-v3-turbo
```

The default is `large-v3`. Models are downloaded from HuggingFace on first use and cached locally.

### `scribe-md list-apps`

List running applications available for per-app audio capture.

### `scribe-md config`

Manage configuration files.

```bash
scribe-md config show    # Print resolved config (merged from all sources)
scribe-md config path    # Print config file path
scribe-md config init    # Create default config at ~/.config/scribe-md/config.toml
```

## Common Options

These options are available on `file`, `url`, and `live`:

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output Markdown file path |
| `--language` | `-l` | Language code (`en`, `ko`, `ja`, etc.). Auto-detected if omitted |
| `--model` | `-m` | Whisper model preset or full HuggingFace repo path |
| `--timestamps` / `--no-timestamps` | `-t` / `-T` | Include timestamps (default: on) |
| `--timestamp-mode` | | `segment`, `paragraph`, `minute`, or `none` |
| `--paragraph-gap` | | Seconds of silence to trigger a paragraph break (default: 2.0) |
| `--chunk-seconds` | | Split long audio into chunks of this duration (default: 1800) |
| `--overlap-seconds` | | Overlap between chunks in seconds (default: 5) |
| `--clean` | | Remove Whisper hallucination artifacts (rule-based, no LLM) |
| `--summarize` | | Append an LLM-generated summary (requires `mlx-lm`) |
| `--diarize` / `--no-diarize` | | Enable speaker identification (requires `pyannote-audio`) |

## Timestamp Modes

Control how timestamps appear in the output with `--timestamp-mode`:

- **`segment`** (default) -- Timestamp before every transcribed segment
- **`paragraph`** -- Timestamp only at the start of each paragraph
- **`minute`** -- Timestamp at the start of each new minute
- **`none`** -- No timestamps, pure flowing text

Example:

```bash
# Clean readable paragraphs with minute markers
scribe-md file lecture.wav --timestamp-mode minute --paragraph-gap 3.0
```

## Obsidian Integration

Write transcriptions directly into your Obsidian vault with YAML frontmatter.

```bash
# Write to vault with frontmatter (date, source, duration, model, tags)
scribe-md file meeting.wav --vault ~/Notes

# Append to today's daily note
scribe-md live --vault ~/Notes --daily-note

# Disable frontmatter
scribe-md url "https://..." --vault ~/Notes --no-frontmatter
```

When `--vault` is set, frontmatter is enabled by default. The output file is placed in the vault's root (or resolved relative to it).

Daily note mode (`--daily-note`) appends a `## Transcription (HH:MM)` section to today's note under the configured daily notes folder (default: `Daily Notes`).

## Post-Processing

### Artifact Cleaning (`--clean`)

Removes common Whisper hallucinations without any LLM -- just pattern matching:

- "Thank you for watching", "Please subscribe", etc.
- Subtitle credits in multiple languages (EN, KO, FR, ES, DE, JA, ZH)
- Consecutive duplicate sentences
- Excessive whitespace

```bash
scribe-md file noisy-audio.wav --clean
```

### LLM Summarization (`--summarize`)

Appends a `## Summary` section generated by a local LLM via `mlx-lm`. This is optional and requires installing `mlx-lm`:

```bash
pip install mlx-lm
scribe-md file lecture.wav --summarize
scribe-md file lecture.wav --summarize --summary-model "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
```

## Speaker Diarization (`--diarize`)

Identify and label speakers using [pyannote-audio](https://github.com/pyannote/pyannote-audio). Speaker changes create paragraph breaks with **Speaker N:** labels.

### Setup

1. Install pyannote-audio:
   ```bash
   pip install pyannote-audio
   ```

2. Get a [HuggingFace token](https://huggingface.co/settings/tokens) and accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

3. Provide the token via any of:
   - `--hf-token YOUR_TOKEN`
   - `hf_token` in config file
   - `HF_TOKEN` environment variable

### Usage

```bash
# Auto-detect number of speakers
scribe-md file meeting.wav --diarize --hf-token hf_xxx

# Specify known speaker count for better accuracy
scribe-md file interview.wav --diarize --num-speakers 2

# With environment variable
export HF_TOKEN=hf_xxx
scribe-md file meeting.wav --diarize
```

### Output Example

```markdown
[00:00:00] **Speaker 1:** Welcome to today's meeting. Let's start with the quarterly review.

[00:00:15] **Speaker 2:** Thanks. Revenue is up twelve percent compared to last quarter.

[00:00:45] **Speaker 1:** That's great news. What about the customer retention numbers?
```

Diarization runs on CPU (Apple Silicon MPS is not yet supported by pyannote). Expect roughly real-time processing speed (30 min audio takes ~15-30 min).

## Parallel Transcription

For long files and YouTube videos, chunked transcription runs in parallel by default:

```bash
# Default: 2 parallel workers
scribe-md file long-lecture.wav

# Use more workers (max 4)
scribe-md file long-lecture.wav --workers 4

# Disable parallelism
scribe-md file long-lecture.wav --no-parallel
```

Parallel transcription uses threads (not processes) because mlx-whisper shares GPU/ANE state within a single process.

## Incremental Output

In chunked mode, see results as they're transcribed:

```bash
# Watch progress in another terminal: tail -f output.md
scribe-md file long-audio.wav --incremental -o output.md
```

For live capture, incremental output is on by default. The final merge pass overwrites with properly deduped text.

## Configuration

Settings are resolved in priority order: CLI flags > project config > user config > defaults.

### Create a config file

```bash
scribe-md config init
# Creates ~/.config/scribe-md/config.toml
```

### Config file format

```toml
[defaults]
model = "large-v3"
language = ""              # empty = auto-detect
timestamps = true
timestamp_mode = "segment"
paragraph_gap = 2.0
chunk_seconds = 1800
overlap_seconds = 5
incremental = false
parallel = true
workers = 2
clean = false
summary_model = ""

[output]
directory = "."

[obsidian]
vault = ""
daily_note_folder = "Daily Notes"

[diarization]
diarize = false
hf_token = ""
num_speakers = 0           # 0 = auto-detect

[live]
keep_audio = false
incremental = true
```

### Project-local config

Place a `.scribe-md.toml` file in your project directory. It uses the same format and overrides the user config.

### View resolved config

```bash
scribe-md config show
```

## Architecture

```
scribe-md/
  capture/                    # Swift CLI (ScreenCaptureKit audio capture)
    Package.swift
    Sources/main.swift
  scribe_md/                  # Python package
    cli.py                    # Typer CLI: file, url, live, config
    transcriber.py            # mlx-whisper transcription
    merger.py                 # Overlap-aware chunk merge with formatting
    audio.py                  # ffmpeg helpers, silence detection
    downloader.py             # yt-dlp wrapper
    capture.py                # Swift binary management
    config.py                 # TOML config loading and merging
    obsidian.py               # Vault integration, frontmatter, daily notes
    postprocess.py            # Artifact cleaning and LLM summarization
    diarize.py                # Speaker diarization (pyannote-audio)
    utils.py                  # Shared utilities
```

## Examples

```bash
# Quick transcription of a short file
scribe-md file voice-memo.m4a -m tiny

# Korean lecture with paragraph timestamps
scribe-md url "https://youtube.com/watch?v=..." -l ko --timestamp-mode paragraph

# Meeting recording with speaker labels, cleaning, and summary
scribe-md file meeting.wav --diarize --clean --summarize --hf-token hf_xxx

# Live capture from Zoom, saved to Obsidian vault
scribe-md live --app Zoom --vault ~/ObsidianVault --daily-note

# Batch transcribe a YouTube playlist
scribe-md url "https://youtube.com/playlist?list=PLxxx" -l en --clean

# Fast transcription with small model, no timestamps
scribe-md file podcast.mp3 -m small -T -o podcast.md
```
