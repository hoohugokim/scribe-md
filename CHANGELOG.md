# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Chunked transcription no longer writes an empty Markdown file and exits 0
  when every chunk fails (e.g. a broken whisper.cpp binary): a total failure
  now raises and exits non-zero, a partial failure logs a warning, and a
  chunk that crashed is no longer mislabeled "silent or no speech".

## [0.2.0] - 2026-06-02

### Added
- **Linux support** via a vendored **whisper.cpp** backend for `scribe-md file`
  and `scribe-md url`. The transcription backend is selected by platform —
  mlx-whisper on macOS, whisper.cpp on Linux — and can be forced with
  `SCRIBE_MD_BACKEND`.
- whisper.cpp builds on first use with auto-detected GPU acceleration
  (CUDA › Vulkan › CPU), overridable via `SCRIBE_MD_WHISPER_ACCEL`;
  `pixi run build-whisper` pre-builds the binary. Added a `linux-64` platform
  and an optional `cuda` Pixi environment.

### Changed
- Live system-audio capture and `--summarize` are macOS-only; on Linux they now
  fail fast with a clear message instead of failing mid-run.
- `--clean` now preserves paragraph and line structure (timestamps included),
  deduplicates timestamped and CJK (punctuation-less) repetition, and no longer
  strips the pronoun `you`/`You` as an inline hallucination.
- whisper.cpp rebuilds automatically when the detected accelerator changes
  (recorded in a build marker), and logs/`describe()` report the accelerator the
  binary was actually built with.
- Accelerator auto-detection now probes for a working device (runs
  `nvidia-smi`/`vulkaninfo`) rather than trusting tool presence, so a headless
  machine falls back to CPU instead of mis-building for an absent GPU.

### Fixed
- `scribe-md file`/`url` no longer loop forever on `--chunk-seconds 0` (or a
  non-positive config value); `split_audio` validates chunk/overlap bounds.
- yt-dlp failures (private/deleted/age-restricted videos, network errors,
  malformed metadata) surface as a clean `DownloadError` message instead of a
  raw traceback, on both the single-video and playlist paths.
- The playlist loop no longer swallows control-flow exits (`typer.Exit`) into a
  meaningless "Skipping" message.
- `--daily-note` without a vault now fails fast instead of silently writing
  plain output.
- Incremental drafts target the resolved final output path and are skipped for
  daily-note output, so a stale draft is no longer orphaned when Obsidian
  redirects where the final transcript is written.
- `live --keep-audio` no longer crashes on a re-run (`dirs_exist_ok`).
- The live chunked pipeline now reaps the capture recorder subprocess on
  failure, preventing an orphaned recorder and a race with temp-dir cleanup.

### Security
- whisper.cpp model downloads are validated (HTML/truncated bodies rejected) and
  published atomically via a process-unique temp file, so a proxy/error page or
  a concurrent run cannot poison the model cache.
- Config-sourced paths are confined: a `model` name from an auto-discovered
  `.scribe-md.toml` cannot steer the download outside the model cache, and
  `daily_note_folder` cannot escape the vault.

## [0.1.1] - 2026-05-06

Pre-release security and correctness pass.

### Security
- `scribe-md config show` redacts `hf_token` to `<set>` so credentials never
  reach stdout.
- README recommends `HF_TOKEN` env var or config file over `--hf-token` on the
  command line (which would be recorded in shell history and visible in `ps`).
- YAML frontmatter values are properly quoted and escaped so untrusted source
  strings (e.g. YouTube titles containing `"` or `:`) cannot break the
  frontmatter.
- TOML rendering in `config show` escapes `\` and `"` in string fields.

### Fixed
- `audio.is_silent` raises a clear `AudioConversionError` with an install hint
  when ffmpeg is missing, matching the other ffmpeg call sites.
- `--keep-audio` saves the kept WAV / chunks directory next to the markdown
  output rather than the current working directory.

## [0.1.0] - 2026-05-04

Initial release.

### Added
- `scribe-md file` — transcribe a local audio file (any format ffmpeg accepts).
- `scribe-md url` — transcribe a YouTube URL or playlist via yt-dlp.
- `scribe-md live` — capture system audio in real time via a Swift
  ScreenCaptureKit binary, with optional per-app filtering (`--app`).
- `scribe-md list-apps` and `scribe-md list-models`.
- `scribe-md config {show,path,init}` for config inspection and bootstrap.
- mlx-whisper backend with model presets (`tiny` … `large-v3-turbo`).
- Chunked transcription for long inputs with overlap-aware,
  sentence-boundary-preferring merging.
- Timestamp modes: `segment`, `paragraph`, `minute`, `none`.
- Incremental output (`--incremental`) so partial transcriptions stream
  to disk while chunks complete.
- Speaker diarization via pyannote-audio (`--diarize`, optional dep).
  Speaker changes trigger paragraph breaks with `**Speaker N:**` labels.
- Rule-based artifact cleaning (`--clean`) for known Whisper hallucinations.
- LLM summarization (`--summarize`) via mlx-lm (optional dep).
- Obsidian integration: YAML frontmatter, daily-note appending, vault-relative
  output paths.
- Layered configuration: built-in defaults → `~/.config/scribe-md/config.toml`
  → project-local `.scribe-md.toml` → CLI flags.
- Configurable output directory.

[Unreleased]: https://github.com/hoohugokim/scribe-md/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hoohugokim/scribe-md/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/hoohugokim/scribe-md/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/hoohugokim/scribe-md/releases/tag/v0.1.0
