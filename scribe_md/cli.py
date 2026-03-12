"""scribe-md CLI: Transcribe system audio and YouTube videos to Markdown."""

import json
import signal
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import audio, capture, downloader, merger, obsidian, transcriber
from .audio import AudioConversionError, DiskFullError
from .capture import CaptureError
from .config import (
    ScribeMdConfig,
    USER_CONFIG_PATH,
    config_as_toml,
    init_user_config,
    load_config,
)
from .transcriber import DEFAULT_MODEL, MODEL_PRESETS, TranscriptionError
from .utils import log, sanitize_filename

app = typer.Typer(
    name="scribe-md",
    help="Transcribe system audio and YouTube videos to Markdown.",
    no_args_is_help=True,
)
console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Config subcommand group
# ---------------------------------------------------------------------------

config_app = typer.Typer(help="Manage scribe-md configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Print the resolved configuration (merged from all sources)."""
    cfg = load_config()
    console.print("[bold]Resolved configuration:[/bold]\n")
    console.print(config_as_toml(cfg))
    console.print("[dim]Sources (lowest to highest priority):[/dim]")
    for src in cfg._sources:
        console.print(f"  - {src}")


@config_app.command("path")
def config_path() -> None:
    """Print the user config file path."""
    exists = USER_CONFIG_PATH.exists()
    console.print(str(USER_CONFIG_PATH))
    if exists:
        console.print("[dim](file exists)[/dim]")
    else:
        console.print("[dim](file does not exist — run 'scribe-md config init' to create)[/dim]")


@config_app.command("init")
def config_init() -> None:
    """Create a default config file at ~/.config/scribe-md/config.toml."""
    try:
        path = init_user_config()
        console.print(f"Created config file: {path}")
    except FileExistsError:
        console.print(f"[yellow]Config file already exists:[/yellow] {USER_CONFIG_PATH}")
        console.print("Edit it directly or delete it to re-initialize.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Config resolution helpers
# ---------------------------------------------------------------------------


def _resolve(cli_value, config_value):
    """Return cli_value if it is not None, otherwise config_value."""
    return cli_value if cli_value is not None else config_value


def _resolve_language(cli_value: str | None, cfg: ScribeMdConfig) -> str | None:
    """Resolve language: CLI flag > config > None (auto-detect).

    An empty string in config means auto-detect.
    """
    if cli_value is not None:
        return cli_value
    return cfg.language or None


_VALID_TIMESTAMP_MODES = ("segment", "paragraph", "minute", "none")


def _validate_timestamp_mode(mode: str) -> None:
    """Raise a Typer error if *mode* is not a recognised timestamp mode."""
    if mode not in _VALID_TIMESTAMP_MODES:
        console.print(
            f"[red]Error:[/red] invalid --timestamp-mode '{mode}'. "
            f"Choose from: {', '.join(_VALID_TIMESTAMP_MODES)}"
        )
        raise typer.Exit(1)


def _resolve_timestamp_flags(
    timestamps: bool,
    timestamp_mode: str,
) -> tuple[bool, str]:
    """Reconcile the legacy ``--timestamps/--no-timestamps`` flag with
    ``--timestamp-mode``.

    Returns ``(effective_timestamps_bool, effective_mode)`` ready for
    ``merge_segments``.
    """
    if not timestamps:
        # --no-timestamps always forces "none"
        return False, "none"
    if timestamp_mode == "none":
        return False, "none"
    return True, timestamp_mode


# ---------------------------------------------------------------------------
# Obsidian output helpers
# ---------------------------------------------------------------------------


def _build_obsidian_metadata(
    source: str,
    duration: float | None,
    language: str | None,
    model: str,
) -> dict:
    """Build an Obsidian frontmatter metadata dict."""
    from datetime import datetime

    meta: dict = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": source,
        "model": model,
        "tags": ["transcription"],
    }
    if duration is not None:
        meta["duration"] = obsidian.format_duration(duration)
    if language:
        meta["language"] = language
    return meta


def _write_obsidian_output(
    text: str,
    output: Path,
    vault: str,
    daily_note: bool,
    frontmatter: bool,
    metadata: dict,
    daily_note_folder: str,
) -> None:
    """Write transcription output with Obsidian integration.

    Handles three modes:
    - daily_note=True: append to today's daily note in the vault
    - frontmatter=True: write to output path with YAML frontmatter
    - fallback: write plain text (no Obsidian features)
    """
    vault_path = Path(vault).expanduser().resolve() if vault else None

    if daily_note and vault_path:
        path = obsidian.append_to_daily_note(
            vault_path, daily_note_folder, text, metadata,
        )
        log(f"Appended to daily note: {path}")
        return

    if frontmatter:
        # If vault is set and output is just a filename, resolve within vault
        if vault_path and not output.is_absolute():
            output = obsidian.resolve_vault_output(vault_path, output.name)
        obsidian.write_with_frontmatter(output, text, metadata)
        log(f"Wrote {output} (with frontmatter)")
        return

    # Plain write (no Obsidian)
    if vault_path and not output.is_absolute():
        output = obsidian.resolve_vault_output(vault_path, output.name)
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    log(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Shared transcription pipeline
# ---------------------------------------------------------------------------


def _append_incremental(output: Path, segments: list[dict]) -> None:
    """Append a chunk's raw transcription text to the output file.

    This provides real-time incremental output so users can watch progress
    with ``tail -f`` or via Obsidian's auto-refresh.  The final merge pass
    will overwrite this draft with properly deduped text.
    """
    if not segments:
        return
    text = " ".join(seg["text"].strip() for seg in segments)
    with open(output, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


def _transcribe_single(
    audio_path: Path,
    output: Path,
    model: str,
    language: str | None,
    timestamps: bool,
    *,
    timestamp_mode: str = "segment",
    paragraph_gap: float = 2.0,
    write_fn=None,
) -> None:
    """Transcribe a single audio file and write Markdown output.

    If *write_fn* is provided it is called as ``write_fn(text, output)``
    instead of writing directly to *output*.
    """
    if audio.is_silent(audio_path):
        log(f"Skipping {audio_path.name}: audio is silent")
        return

    result = transcriber.transcribe_audio(audio_path, model=model, language=language)
    segments = transcriber.extract_segments(result)

    if not segments:
        log(f"Skipping {audio_path.name}: no speech detected")
        return

    text = merger.merge_segments(
        [segments], chunk_duration=0, overlap=0, timestamps=timestamps,
        timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
    )
    if write_fn is not None:
        write_fn(text, output)
    else:
        output.write_text(text, encoding="utf-8")
        log(f"Wrote {output}")


def _transcribe_chunked(
    audio_path: Path,
    output: Path,
    model: str,
    language: str | None,
    timestamps: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    *,
    timestamp_mode: str = "segment",
    paragraph_gap: float = 2.0,
    incremental: bool = False,
    write_fn=None,
) -> None:
    """Split a long audio file into chunks, transcribe each, and merge.

    When *incremental* is True, each chunk's preliminary transcription is
    appended to the output file as soon as it completes.  The final merge
    pass then overwrites the file with the properly deduped result.

    If *write_fn* is provided it is called as ``write_fn(text, output)``
    instead of writing directly to *output*.
    """
    with tempfile.TemporaryDirectory(prefix="scribe-md-chunks-") as tmp:
        tmp_dir = Path(tmp)

        log(f"Splitting into {chunk_seconds}s chunks...")
        chunks = audio.split_audio(audio_path, tmp_dir, chunk_seconds, overlap_seconds)

        # Clear the output file before writing incremental results
        if incremental:
            output.write_text("", encoding="utf-8")

        all_segments: list[list[dict]] = []
        for i, chunk_path in enumerate(chunks):
            console.print(f"  Transcribing chunk {i + 1}/{len(chunks)}...")
            if audio.is_silent(chunk_path):
                log(f"  Chunk {i}: silent, skipping")
                all_segments.append([])
                continue
            result = transcriber.transcribe_audio(chunk_path, model=model, language=language)
            segments = transcriber.extract_segments(result)
            all_segments.append(segments)

            if incremental:
                _append_incremental(output, segments)

        text = merger.merge_segments(
            all_segments,
            chunk_duration=chunk_seconds,
            overlap=overlap_seconds,
            timestamps=timestamps,
            timestamp_mode=timestamp_mode,
            paragraph_gap=paragraph_gap,
        )
        if write_fn is not None:
            write_fn(text, output)
        else:
            output.write_text(text, encoding="utf-8")
            log(f"Wrote {output} ({len(chunks)} chunks merged)")


# ---------------------------------------------------------------------------
# scribe-md file
# ---------------------------------------------------------------------------


@app.command()
def file(
    audio_file: Path = typer.Argument(..., help="Path to audio file (WAV, MP3, etc.)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output markdown path"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (en, ko, etc.)"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Whisper model name or preset (tiny, base, small, medium, large-v3)",
    ),
    timestamps: Optional[bool] = typer.Option(None, "--timestamps/--no-timestamps", "-t/-T", help="Include timestamps"),
    timestamp_mode: Optional[str] = typer.Option(
        None, "--timestamp-mode",
        help="Timestamp granularity: segment, paragraph, minute, or none",
    ),
    paragraph_gap: Optional[float] = typer.Option(
        None, "--paragraph-gap",
        help="Seconds of silence to trigger a paragraph break",
    ),
    chunk_seconds: Optional[float] = typer.Option(
        None, "--chunk-seconds", help="Chunk duration for long files (seconds)",
    ),
    overlap_seconds: Optional[float] = typer.Option(None, "--overlap-seconds", help="Overlap between chunks"),
    incremental: Optional[bool] = typer.Option(
        None, "--incremental/--no-incremental",
        help="Write chunks to output file incrementally (default: off)",
    ),
    vault: Optional[str] = typer.Option(None, "--vault", help="Obsidian vault path (overrides config)"),
    daily_note: bool = typer.Option(False, "--daily-note", help="Append to today's daily note"),
    frontmatter: Optional[bool] = typer.Option(None, "--frontmatter/--no-frontmatter", help="Include YAML frontmatter (default: on when vault is set)"),
) -> None:
    """Transcribe an existing audio file to Markdown."""
    if not audio_file.exists():
        console.print(f"[red]Error:[/red] {audio_file} not found")
        raise typer.Exit(1)

    if audio_file.stat().st_size == 0:
        console.print(f"[red]Error:[/red] {audio_file} is empty (0 bytes)")
        raise typer.Exit(1)

    cfg = load_config()
    r_model = _resolve(model, cfg.model)
    r_language = _resolve_language(language, cfg)
    r_timestamps = _resolve(timestamps, cfg.timestamps)
    r_timestamp_mode = _resolve(timestamp_mode, cfg.timestamp_mode)
    r_paragraph_gap = _resolve(paragraph_gap, cfg.paragraph_gap)
    r_chunk_seconds = _resolve(chunk_seconds, cfg.chunk_seconds)
    r_overlap_seconds = _resolve(overlap_seconds, cfg.overlap_seconds)
    r_incremental = _resolve(incremental, cfg.incremental)
    r_vault = _resolve(vault, cfg.vault)
    r_daily_note_folder = cfg.daily_note_folder

    # Frontmatter defaults to True when vault is set
    r_frontmatter = frontmatter if frontmatter is not None else bool(r_vault)

    _validate_timestamp_mode(r_timestamp_mode)
    ts, ts_mode = _resolve_timestamp_flags(r_timestamps, r_timestamp_mode)
    out = output or audio_file.with_suffix(".md")

    source = f"file: {audio_file.name}"

    try:
        with tempfile.TemporaryDirectory(prefix="scribe-md-") as tmp:
            # Convert to 16kHz mono WAV
            converted = Path(tmp) / "converted.wav"
            log("Converting to 16kHz mono...")
            audio.convert_to_16k_mono(audio_file, converted)

            file_duration = audio.get_duration(converted)
            metadata = _build_obsidian_metadata(
                source=source, duration=file_duration,
                language=r_language, model=r_model,
            )

            def write_fn(text: str, output_path: Path) -> None:
                _write_obsidian_output(
                    text, output_path, r_vault, daily_note, r_frontmatter,
                    metadata, r_daily_note_folder,
                )

            if file_duration > r_chunk_seconds:
                _transcribe_chunked(
                    converted, out, r_model, r_language, ts,
                    r_chunk_seconds, r_overlap_seconds,
                    timestamp_mode=ts_mode, paragraph_gap=r_paragraph_gap,
                    incremental=r_incremental,
                    write_fn=write_fn,
                )
            else:
                _transcribe_single(
                    converted, out, r_model, r_language, ts,
                    timestamp_mode=ts_mode, paragraph_gap=r_paragraph_gap,
                    write_fn=write_fn,
                )
    except AudioConversionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except TranscriptionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except DiskFullError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except OSError as e:
        if e.errno == 28:  # ENOSPC — No space left on device
            console.print(
                "[red]Error:[/red] Disk full. Free up space and try again."
            )
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# scribe-md url
# ---------------------------------------------------------------------------


@app.command()
def url(
    video_url: str = typer.Argument(..., help="YouTube URL or playlist URL"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output markdown path"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (en, ko, etc.)"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Whisper model name or preset (tiny, base, small, medium, large-v3)",
    ),
    timestamps: Optional[bool] = typer.Option(None, "--timestamps/--no-timestamps", "-t/-T", help="Include timestamps"),
    timestamp_mode: Optional[str] = typer.Option(
        None, "--timestamp-mode",
        help="Timestamp granularity: segment, paragraph, minute, or none",
    ),
    paragraph_gap: Optional[float] = typer.Option(
        None, "--paragraph-gap",
        help="Seconds of silence to trigger a paragraph break",
    ),
    chunk_seconds: Optional[float] = typer.Option(
        None, "--chunk-seconds", help="Chunk duration for long videos (seconds)",
    ),
    overlap_seconds: Optional[float] = typer.Option(None, "--overlap-seconds", help="Overlap between chunks"),
    incremental: Optional[bool] = typer.Option(
        None, "--incremental/--no-incremental",
        help="Write chunks to output file incrementally (default: off)",
    ),
    vault: Optional[str] = typer.Option(None, "--vault", help="Obsidian vault path (overrides config)"),
    daily_note: bool = typer.Option(False, "--daily-note", help="Append to today's daily note"),
    frontmatter: Optional[bool] = typer.Option(None, "--frontmatter/--no-frontmatter", help="Include YAML frontmatter (default: on when vault is set)"),
) -> None:
    """Transcribe audio from a YouTube URL to Markdown."""
    cfg = load_config()
    r_model = _resolve(model, cfg.model)
    r_language = _resolve_language(language, cfg)
    r_timestamps = _resolve(timestamps, cfg.timestamps)
    r_timestamp_mode = _resolve(timestamp_mode, cfg.timestamp_mode)
    r_paragraph_gap = _resolve(paragraph_gap, cfg.paragraph_gap)
    r_chunk_seconds = _resolve(chunk_seconds, cfg.chunk_seconds)
    r_overlap_seconds = _resolve(overlap_seconds, cfg.overlap_seconds)
    r_incremental = _resolve(incremental, cfg.incremental)
    r_vault = _resolve(vault, cfg.vault)
    r_daily_note_folder = cfg.daily_note_folder

    # Frontmatter defaults to True when vault is set
    r_frontmatter = frontmatter if frontmatter is not None else bool(r_vault)

    _validate_timestamp_mode(r_timestamp_mode)
    ts, ts_mode = _resolve_timestamp_flags(r_timestamps, r_timestamp_mode)

    try:
        # Check if this is a playlist
        if downloader.is_playlist(video_url):
            entries = downloader.get_playlist_entries(video_url)
            log(f"Playlist with {len(entries)} videos")

            for i, entry in enumerate(entries):
                entry_url = entry.get("url") or entry.get("webpage_url", "")
                entry_title = entry.get("title", f"video_{i}")
                log(f"\n[{i + 1}/{len(entries)}] {entry_title}")

                try:
                    _transcribe_url(
                        entry_url, output=None, model=r_model, language=r_language,
                        timestamps=ts, chunk_seconds=r_chunk_seconds,
                        overlap_seconds=r_overlap_seconds,
                        timestamp_mode=ts_mode, paragraph_gap=r_paragraph_gap,
                        incremental=r_incremental,
                        vault=r_vault, daily_note=daily_note,
                        frontmatter=r_frontmatter,
                        daily_note_folder=r_daily_note_folder,
                    )
                except DiskFullError:
                    raise  # Disk-full is fatal even for playlists
                except Exception as e:
                    console.print(f"[yellow]Skipping: {e}[/yellow]")
        else:
            _transcribe_url(
                video_url, output=output, model=r_model, language=r_language,
                timestamps=ts, chunk_seconds=r_chunk_seconds,
                overlap_seconds=r_overlap_seconds,
                timestamp_mode=ts_mode, paragraph_gap=r_paragraph_gap,
                incremental=r_incremental,
                vault=r_vault, daily_note=daily_note,
                frontmatter=r_frontmatter,
                daily_note_folder=r_daily_note_folder,
            )
    except (AudioConversionError, TranscriptionError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except DiskFullError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except OSError as e:
        if e.errno == 28:
            console.print(
                "[red]Error:[/red] Disk full. Free up space and try again."
            )
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _transcribe_url(
    video_url: str,
    output: Path | None,
    model: str,
    language: str | None,
    timestamps: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    *,
    timestamp_mode: str = "segment",
    paragraph_gap: float = 2.0,
    incremental: bool = False,
    vault: str = "",
    daily_note: bool = False,
    frontmatter: bool = False,
    daily_note_folder: str = "Daily Notes",
) -> None:
    """Download and transcribe a single video URL."""
    with tempfile.TemporaryDirectory(prefix="scribe-md-dl-") as tmp:
        tmp_dir = Path(tmp)

        # Download audio
        raw_audio, title = downloader.download_audio(video_url, tmp_dir)

        # Convert to 16kHz mono
        converted = tmp_dir / "converted.wav"
        log("Converting to 16kHz mono...")
        audio.convert_to_16k_mono(raw_audio, converted)

        # Determine output path
        out = output or Path(f"{sanitize_filename(title)}.md")

        duration = audio.get_duration(converted)
        log(f"Duration: {duration / 60:.1f} min")

        source = f"YouTube: {title}"
        metadata = _build_obsidian_metadata(
            source=source, duration=duration,
            language=language, model=model,
        )

        def write_fn(text: str, output_path: Path) -> None:
            _write_obsidian_output(
                text, output_path, vault, daily_note, frontmatter,
                metadata, daily_note_folder,
            )

        if duration > chunk_seconds:
            _transcribe_chunked(
                converted, out, model, language, timestamps,
                chunk_seconds, overlap_seconds,
                timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
                incremental=incremental,
                write_fn=write_fn,
            )
        else:
            _transcribe_single(
                converted, out, model, language, timestamps,
                timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
                write_fn=write_fn,
            )


# ---------------------------------------------------------------------------
# scribe-md live
# ---------------------------------------------------------------------------


@app.command()
def live(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output markdown path"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Recording duration (seconds)"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (en, ko, etc.)"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Whisper model name or preset (tiny, base, small, medium, large-v3)",
    ),
    timestamps: Optional[bool] = typer.Option(None, "--timestamps/--no-timestamps", "-t/-T", help="Include timestamps"),
    timestamp_mode: Optional[str] = typer.Option(
        None, "--timestamp-mode",
        help="Timestamp granularity: segment, paragraph, minute, or none",
    ),
    paragraph_gap: Optional[float] = typer.Option(
        None, "--paragraph-gap",
        help="Seconds of silence to trigger a paragraph break",
    ),
    chunk_seconds: Optional[float] = typer.Option(
        None, "--chunk-seconds", help="Enable chunked pipeline (transcribe every N seconds)",
    ),
    overlap_seconds: Optional[float] = typer.Option(None, "--overlap-seconds", help="Overlap between chunks"),
    keep_audio: Optional[bool] = typer.Option(None, "--keep-audio", help="Keep intermediate WAV files"),
    app_name: Optional[list[str]] = typer.Option(None, "--app", "-a", help="Capture from specific app(s) (repeatable)"),
    incremental: Optional[bool] = typer.Option(
        None, "--incremental/--no-incremental",
        help="Write chunks to output file incrementally (default: on for live)",
    ),
    vault: Optional[str] = typer.Option(None, "--vault", help="Obsidian vault path (overrides config)"),
    daily_note: bool = typer.Option(False, "--daily-note", help="Append to today's daily note"),
    frontmatter: Optional[bool] = typer.Option(None, "--frontmatter/--no-frontmatter", help="Include YAML frontmatter (default: on when vault is set)"),
) -> None:
    """Capture and transcribe system audio in real-time."""
    cfg = load_config()
    r_model = _resolve(model, cfg.model)
    r_language = _resolve_language(language, cfg)
    r_timestamps = _resolve(timestamps, cfg.timestamps)
    r_timestamp_mode = _resolve(timestamp_mode, cfg.timestamp_mode)
    r_paragraph_gap = _resolve(paragraph_gap, cfg.paragraph_gap)
    r_chunk_seconds = _resolve(chunk_seconds, 0)  # live default: no chunking
    r_overlap_seconds = _resolve(overlap_seconds, cfg.overlap_seconds)
    r_keep_audio = _resolve(keep_audio, cfg.keep_audio)
    r_incremental = _resolve(incremental, cfg.live_incremental)
    r_vault = _resolve(vault, cfg.vault)
    r_daily_note_folder = cfg.daily_note_folder
    r_output = output or Path("transcription.md")

    # Frontmatter defaults to True when vault is set
    r_frontmatter = frontmatter if frontmatter is not None else bool(r_vault)

    _validate_timestamp_mode(r_timestamp_mode)
    ts, ts_mode = _resolve_timestamp_flags(r_timestamps, r_timestamp_mode)

    # Build source and metadata for Obsidian
    apps = app_name if app_name else None
    if apps:
        source = f"live: {', '.join(apps)}"
    else:
        source = "live: system audio"

    metadata = _build_obsidian_metadata(
        source=source, duration=None,
        language=r_language, model=r_model,
    )

    def write_fn(text: str, output_path: Path) -> None:
        _write_obsidian_output(
            text, output_path, r_vault, daily_note, r_frontmatter,
            metadata, r_daily_note_folder,
        )

    try:
        if r_chunk_seconds > 0:
            _live_chunked(
                r_output, duration, r_language, r_model, ts,
                r_chunk_seconds, r_overlap_seconds, r_keep_audio, apps,
                timestamp_mode=ts_mode, paragraph_gap=r_paragraph_gap,
                incremental=r_incremental,
                write_fn=write_fn,
            )
        else:
            _live_single(
                r_output, duration, r_language, r_model, ts, r_keep_audio, apps,
                timestamp_mode=ts_mode, paragraph_gap=r_paragraph_gap,
                write_fn=write_fn,
            )
    except CaptureError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except (AudioConversionError, TranscriptionError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except DiskFullError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except OSError as e:
        if e.errno == 28:
            console.print(
                "[red]Error:[/red] Disk full. Free up space and try again."
            )
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _live_single(
    output: Path,
    duration: float | None,
    language: str | None,
    model: str,
    timestamps: bool,
    keep_audio: bool,
    app: str | list[str] | None = None,
    *,
    timestamp_mode: str = "segment",
    paragraph_gap: float = 2.0,
    write_fn=None,
) -> None:
    """Single-file live capture pipeline."""
    with tempfile.TemporaryDirectory(prefix="scribe-md-live-") as tmp:
        tmp_dir = Path(tmp)
        raw_wav = tmp_dir / "recording.wav"

        proc = capture.run_capture(raw_wav, duration=duration, app=app)

        # Let Ctrl+C propagate to the capture subprocess
        original_sigint = signal.getsignal(signal.SIGINT)
        cancelled = False

        def _handle_sigint(signum, frame):
            nonlocal cancelled
            cancelled = True
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)

        signal.signal(signal.SIGINT, _handle_sigint)

        try:
            proc.wait()
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        if cancelled or proc.returncode != 0:
            log("Recording cancelled.")
            raise typer.Exit(1)

        if not raw_wav.exists() or raw_wav.stat().st_size == 0:
            log("No audio recorded.")
            raise typer.Exit(1)

        # Convert and transcribe
        converted = tmp_dir / "converted.wav"
        log("Converting to 16kHz mono...")
        audio.convert_to_16k_mono(raw_wav, converted)
        _transcribe_single(
            converted, output, model, language, timestamps,
            timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
            write_fn=write_fn,
        )

        if keep_audio:
            import shutil
            saved = Path(f"recording_{output.stem}.wav")
            shutil.copy2(converted, saved)
            log(f"Audio saved: {saved}")


def _live_chunked(
    output: Path,
    duration: float | None,
    language: str | None,
    model: str,
    timestamps: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    keep_audio: bool,
    app: str | list[str] | None = None,
    *,
    timestamp_mode: str = "segment",
    paragraph_gap: float = 2.0,
    incremental: bool = False,
    write_fn=None,
) -> None:
    """Chunked live capture pipeline with concurrent transcription."""
    with tempfile.TemporaryDirectory(prefix="scribe-md-chunks-") as tmp:
        tmp_dir = Path(tmp)
        chunk_base = tmp_dir / "chunk.wav"

        proc = capture.run_capture(
            chunk_base, duration=duration,
            chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds,
            app=app,
        )

        # Handle Ctrl+C: let capture finish current chunk
        original_sigint = signal.getsignal(signal.SIGINT)
        cancelled = False

        def _handle_sigint(signum, frame):
            nonlocal cancelled
            cancelled = True
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)

        signal.signal(signal.SIGINT, _handle_sigint)

        chunk_jsons: list[Path] = []
        chunk_idx = 0

        # Clear the output file before writing incremental results
        if incremental:
            output.write_text("", encoding="utf-8")

        try:
            # Read chunk paths from capture's stdout
            assert proc.stdout is not None
            for line in proc.stdout:
                chunk_raw = Path(line.decode().strip())
                if not chunk_raw.exists():
                    continue

                idx_str = f"{chunk_idx:03d}"
                chunk_16k = tmp_dir / f"chunk_{idx_str}_16k.wav"
                chunk_json = tmp_dir / f"chunk_{idx_str}.json"

                log(f"Chunk {chunk_idx}: processing...")
                try:
                    audio.convert_to_16k_mono(chunk_raw, chunk_16k)
                    if audio.is_silent(chunk_16k):
                        log(f"  Chunk {chunk_idx}: silent, skipping")
                        chunk_idx += 1
                        continue
                    result = transcriber.transcribe_audio(
                        chunk_16k, model=model, language=language,
                    )
                    segments = transcriber.extract_segments(result)
                    data = {"chunk_index": chunk_idx, "segments": segments}
                    chunk_json.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    chunk_jsons.append(chunk_json)

                    if incremental:
                        _append_incremental(output, segments)
                except DiskFullError:
                    raise  # Disk-full is fatal — stop immediately
                except Exception as e:
                    log(f"Chunk {chunk_idx} failed: {e}")

                chunk_idx += 1

            proc.wait()
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        if not chunk_jsons:
            log("No chunks were transcribed.")
            raise typer.Exit(1)

        # Merge all chunks
        all_segments: list[list[dict]] = []
        for cj in sorted(chunk_jsons):
            data = json.loads(cj.read_text(encoding="utf-8"))
            all_segments.append(data.get("segments", []))

        text = merger.merge_segments(
            all_segments,
            chunk_duration=chunk_seconds,
            overlap=overlap_seconds,
            timestamps=timestamps,
            timestamp_mode=timestamp_mode,
            paragraph_gap=paragraph_gap,
        )
        if write_fn is not None:
            write_fn(text, output)
        else:
            output.write_text(text, encoding="utf-8")
            log(f"Done: {output} ({len(chunk_jsons)} chunks)")

        if keep_audio:
            log(f"Audio chunks saved: {tmp_dir}")
            # Prevent cleanup by copying to a persistent location
            import shutil
            saved_dir = Path(f"chunks_{output.stem}")
            shutil.copytree(tmp_dir, saved_dir)
            log(f"Audio saved: {saved_dir}")


# ---------------------------------------------------------------------------
# scribe-md list-models / list-apps
# ---------------------------------------------------------------------------


@app.command("list-models")
def list_models() -> None:
    """List available Whisper model presets."""
    console.print(f"{'Preset':<20} HF Repo Path")
    console.print(f"{'─' * 20} {'─' * 50}")
    for name, path in MODEL_PRESETS.items():
        marker = " *" if name == DEFAULT_MODEL else ""
        console.print(f"{name:<20} {path}{marker}")
    console.print(f"\n[dim]* = default model ({DEFAULT_MODEL})[/dim]")


@app.command("list-apps")
def list_apps() -> None:
    """List running apps available for per-app audio capture."""
    try:
        apps = capture.list_apps()
    except CaptureError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not apps:
        console.print("[yellow]No apps found.[/yellow]")
        raise typer.Exit(1)

    console.print(f"{'App Name':<40} Bundle ID")
    console.print(f"{'─' * 40} {'─' * 40}")
    for a in apps:
        console.print(f"{a['name']:<40} {a['bundle_id']}")
