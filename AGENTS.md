<!-- statutor: plane=constitution | policy=constitution | writer=human | budget: soft 120 / hard 200 lines -->
# AGENTS.md

> scribe-md: local transcription of system audio / files / YouTube to Markdown — mlx-whisper (default) or Qwen3-ASR on Apple Silicon; whisper.cpp or Nemotron 3.5 Transcribe (CUDA default-when-available) on Linux. Fully offline (Nemotron talks to a local vLLM). Pixi is the only env manager.

## Commands
- Build (macOS): `pixi run build-capture` — Swift ScreenCaptureKit binary in `capture/`
- Build (Linux): `pixi run build-whisper` — `vendor/whisper.cpp` submodule + `python -m scribe_md.backends.whispercpp build` (Vulkan default; `pixi run -e cuda build-whisper` for CUDA)
- Test: `pixi run pytest` / `pixi run python -m pytest tests/ -q` (18 test modules)
- Run CLI: `pixi run scribe-md --help` / `pixi run scribe-md file|url|live ...`
- Lint: no dedicated task; `pixi run python -m py_compile scribe_md/*.py` smoke check

## Conventions that differ from defaults
- Pixi only. `pixi install` provisions ffmpeg+python; never `pip install` outside pixi except optional `pyannote-audio`/`mlx-lm`/`mlx-audio`.
- Backend selection via `platform_support.py` + `scribe_md/backends/` protocol; override with `--engine` / `[defaults].engine` / `SCRIBE_MD_BACKEND` (`mlx|qwen|whispercpp|nemotron`).
- Config layered: CLI flags > `.scribe-md.toml` (project) > `~/.config/scribe-md/config.toml` (user) > defaults in `config.py`.
- Long audio chunked by default: `audio.py` ffmpeg split + `merger.py` overlap dedup; `--incremental` writes interim output.
- Obsidian output via `obsidian.py`: vault-relative paths, YAML frontmatter, `--daily-note` append.
- Multi-input and `--gpus` are on main (v0.3.0) via `scheduler.py`/`gpu.py`. Parallelism is CUDA-only; Vulkan/MLX stay sequential. main is the only long-lived branch (D-0002).

## Pitfalls (hard-won, one line each)
- <add only after an agent actually made the mistake>

## Boundaries
- Do not edit: `vendor/whisper.cpp` (submodule), `capture/.build/`, `output/`, `plans/archive/`, generated vault `.md`.
- Do not hand-edit `CHANGELOG.md` — generated from conventional commits.
- Ledger discipline: HANDOFF.md (overwrite-only, 40 lines), TASKS.md (queue, stable IDs), DECISIONS.md (append-only).
- `AGENT.md` (singular) is transcript-cleanup prompt, not the constitution — leave it.
