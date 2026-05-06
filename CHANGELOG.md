# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-06

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

### Security
- `scribe-md config show` redacts `hf_token` to `<set>` so tokens never reach
  stdout.
- README recommends `HF_TOKEN` env var or config file over `--hf-token` on
  the command line (which would be recorded in shell history and visible in
  `ps`).
- YAML frontmatter values are properly quoted and escaped so untrusted source
  strings (e.g. YouTube titles containing `"` or `:`) cannot break the
  frontmatter.
- TOML rendering in `config show` escapes `\` and `"` in string fields.

### Fixed
- `audio.is_silent` raises a clear `AudioConversionError` with an install hint
  when ffmpeg is missing, matching the other ffmpeg call sites.
- `--keep-audio` saves the kept WAV / chunks directory next to the markdown
  output rather than the current working directory.

[Unreleased]: https://github.com/hoohugokim/scribe-md/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hoohugokim/scribe-md/releases/tag/v0.1.0
