"""scribe-md CLI: Transcribe system audio and YouTube videos to Markdown."""

import json
import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import audio, capture, diarize, downloader, gpu, merger, obsidian, postprocess, transcriber
from . import platform_support, scheduler
from .audio import AudioConversionError, DiskFullError
from .capture import CaptureError
from .diarize import DiarizationError
from .config import (
    ScribeMdConfig,
    USER_CONFIG_PATH,
    config_as_toml,
    init_user_config,
    load_config,
)
from .scheduler import PreparedSource
from .transcriber import DEFAULT_MODEL, MODEL_PRESETS, TranscriptionError
from .utils import log, sanitize_filename

app = typer.Typer(
    name="scribe-md",
    help="Transcribe system audio and YouTube videos to Markdown.",
    no_args_is_help=True,
)
console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Shared CLI option types
#
# These options are byte-for-byte identical across the file/url/live commands,
# so they are declared once here as Annotated aliases. Options whose help text
# legitimately differs per command (chunk_seconds, incremental) stay inline.
# ---------------------------------------------------------------------------

_Output = Annotated[Optional[Path], typer.Option("--output", "-o", help="Output markdown path")]
_Language = Annotated[Optional[str], typer.Option("--language", "-l", help="Language code (en, ko, etc.)")]
_Model = Annotated[Optional[str], typer.Option(
    "--model", "-m",
    help="Whisper model name or preset (tiny, base, small, medium, large-v3)",
)]
_Timestamps = Annotated[Optional[bool], typer.Option("--timestamps/--no-timestamps", "-t/-T", help="Include timestamps")]
_TimestampMode = Annotated[Optional[str], typer.Option(
    "--timestamp-mode", help="Timestamp granularity: segment, paragraph, minute, or none",
)]
_ParagraphGap = Annotated[Optional[float], typer.Option(
    "--paragraph-gap", help="Seconds of silence to trigger a paragraph break",
)]
_OverlapSeconds = Annotated[Optional[float], typer.Option("--overlap-seconds", help="Overlap between chunks")]
_Vault = Annotated[Optional[str], typer.Option("--vault", help="Obsidian vault path (overrides config)")]
_DailyNote = Annotated[bool, typer.Option("--daily-note", help="Append to today's daily note")]
_Frontmatter = Annotated[Optional[bool], typer.Option("--frontmatter/--no-frontmatter", help="Include YAML frontmatter (default: on when vault is set)")]
_Clean = Annotated[Optional[bool], typer.Option("--clean", help="Apply rule-based artifact cleaning to the transcription")]
_Summarize = Annotated[bool, typer.Option("--summarize", help="Append an LLM-generated summary (requires mlx-lm)")]
_SummaryModel = Annotated[Optional[str], typer.Option("--summary-model", help="Override the LLM model for summarization")]
_Diarize = Annotated[Optional[bool], typer.Option("--diarize/--no-diarize", help="Enable speaker diarization (requires pyannote-audio)")]
_HfToken = Annotated[Optional[str], typer.Option("--hf-token", help="HuggingFace token for diarization model")]
_NumSpeakers = Annotated[Optional[int], typer.Option("--num-speakers", help="Number of speakers (0 = auto-detect)")]
_FromFile = Annotated[Optional[Path], typer.Option(
    "--from-file", help="Read inputs (one per line; '#' comments allowed) from a file",
)]
_Gpus = Annotated[Optional[str], typer.Option(
    "--gpus", help="GPUs for parallel transcription: 'auto', N, or '0,1' (CUDA only)",
)]

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


@dataclass(frozen=True)
class _Resolved:
    """Settings shared by file/url/live after merging CLI flags with config.

    ``chunk_seconds``/``incremental`` are resolved per-command (their config
    source differs) and ``keep_audio`` is live-only, so they are not here.
    ``ts``/``ts_mode`` are the timestamp flags already reconciled via
    :func:`_resolve_timestamp_flags`.
    """

    model: str
    language: str | None
    paragraph_gap: float
    overlap_seconds: float
    vault: str
    daily_note_folder: str
    clean: bool
    summary_model: str
    diarize: bool
    hf_token: str
    num_speakers: int
    frontmatter: bool
    ts: bool
    ts_mode: str


def _resolve_common_options(
    cfg: ScribeMdConfig,
    *,
    model,
    language,
    timestamps,
    timestamp_mode,
    paragraph_gap,
    overlap_seconds,
    vault,
    daily_note,
    frontmatter,
    clean,
    summary_model,
    diarize_flag,
    hf_token,
    num_speakers,
) -> _Resolved:
    """Merge the CLI options common to file/url/live with *cfg*.

    Also runs the shared validation (``--daily-note`` needs a vault, the
    timestamp mode is recognised) and reconciles the timestamp flags, raising
    ``typer.Exit`` on invalid input — exactly as the inline blocks did.
    """
    r_vault = _resolve(vault, cfg.vault)
    r_timestamp_mode = _resolve(timestamp_mode, cfg.timestamp_mode)

    # Frontmatter defaults to True when a vault is in play.
    r_frontmatter = frontmatter if frontmatter is not None else bool(r_vault)

    _validate_daily_note(daily_note, r_vault)
    _validate_timestamp_mode(r_timestamp_mode)
    ts, ts_mode = _resolve_timestamp_flags(
        _resolve(timestamps, cfg.timestamps), r_timestamp_mode,
    )

    return _Resolved(
        model=_resolve(model, cfg.model),
        language=_resolve_language(language, cfg),
        paragraph_gap=_resolve(paragraph_gap, cfg.paragraph_gap),
        overlap_seconds=_resolve(overlap_seconds, cfg.overlap_seconds),
        vault=r_vault,
        daily_note_folder=cfg.daily_note_folder,
        clean=_resolve(clean, cfg.clean),
        summary_model=_resolve(summary_model, cfg.summary_model),
        diarize=_resolve(diarize_flag, cfg.diarize),
        hf_token=_resolve(hf_token, cfg.hf_token),
        num_speakers=_resolve(num_speakers, cfg.num_speakers),
        frontmatter=r_frontmatter,
        ts=ts,
        ts_mode=ts_mode,
    )


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


def _validate_daily_note(daily_note: bool, vault: str) -> None:
    """Fail fast when daily-note output is requested without a vault."""
    if daily_note and not vault:
        console.print(
            "[red]Error:[/red] --daily-note requires --vault or obsidian.vault "
            "in the config."
        )
        raise typer.Exit(1)


def _collect_inputs(positional: list[str], from_file: Path | None) -> list[str]:
    """Merge positional inputs with a --from-file list; fail fast if empty."""
    inputs = list(positional or [])
    if from_file is not None:
        for line in from_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                inputs.append(line)
    if not inputs:
        console.print("[red]Error:[/red] no inputs given (positional or --from-file).")
        raise typer.Exit(1)
    return inputs


def _validate_single_output(inputs: list, output: Path | None) -> None:
    """`-o/--output` names one file, so reject it with multiple inputs."""
    if output is not None and len(inputs) > 1:
        console.print(
            "[red]Error:[/red] --output/-o works with a single input only; "
            "with multiple inputs, outputs are written to the output directory."
        )
        raise typer.Exit(1)


def _backend_is_cuda() -> bool:
    """True only when the active backend is whisper.cpp built for CUDA."""
    from .backends import get_backend
    from .backends.whispercpp import _read_built_accel, detect_accel

    backend = get_backend()
    if backend.name != "whispercpp":
        return False
    return (_read_built_accel() or detect_accel()) == "cuda"


def _resolve_gpu_ids(spec: str | None) -> list[int]:
    """Resolve --gpus to device ids, or [] to mean 'run sequentially'.

    Returns [] for the default/single case and for non-CUDA backends (with a
    one-line notice), so callers treat [] as the existing sequential path.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "1"):
        return []
    if not _backend_is_cuda():
        console.print(
            "[yellow]Note:[/yellow] --gpus needs the CUDA whisper.cpp backend; "
            "running sequentially on this platform."
        )
        return []
    try:
        ids = gpu.resolve_gpu_spec(spec, gpu.discover_cuda_devices())
    except gpu.GpuSpecError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    return ids if len(ids) > 1 else []


def _output_path_for(
    src: Path | None,
    single_output: Path | None,
    output_directory: str,
    *,
    title: str | None = None,
) -> Path:
    """Determine the output path for one source."""
    if single_output is not None:
        return single_output
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(title) if title is not None else src.stem
    return out_dir / f"{stem}.md"


def _should_chunk(duration: float, chunk_seconds: float) -> bool:
    """Return whether an input should use the chunked pipeline."""
    return chunk_seconds > 0 and duration > chunk_seconds


def _resolve_incremental_output(
    output: Path,
    *,
    vault: str,
    daily_note: bool,
    incremental: bool,
) -> tuple[bool, Path | None]:
    """Resolve where incremental drafts should be written, if enabled."""
    if not incremental:
        return False, None

    vault_path = Path(vault).expanduser().resolve() if vault else None
    if daily_note and vault_path:
        log("Incremental output disabled for daily-note output.")
        return False, None

    if vault_path and not output.is_absolute():
        return True, obsidian.resolve_vault_output(vault_path, output.name)

    return True, output


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _guard_summarize_on_linux(summarize: bool) -> None:
    """Fail fast if --summarize is requested on Linux (mlx-lm is macOS-only).

    Called at command entry so a long transcription is not wasted before the
    user learns summarization is unavailable.
    """
    if summarize and platform_support.is_linux():
        console.print(
            "[red]Error:[/red] Summarization (mlx-lm) is macOS-only for now."
        )
        raise typer.Exit(1)


def _apply_postprocessing(
    text: str,
    *,
    clean: bool = False,
    summarize: bool = False,
    summary_model: str = "",
) -> str:
    """Apply optional post-processing steps to the merged transcription text.

    Steps (in order):
    1. ``--clean``: rule-based artifact removal (no LLM).
    2. ``--summarize``: append an LLM-generated ``## Summary`` section.

    Returns the (possibly modified) text.
    """
    if clean:
        text = postprocess.clean_transcription(text)

    if summarize:
        _guard_summarize_on_linux(summarize)
        try:
            model = summary_model or None
            summary = postprocess.summarize_with_llm(text, model=model)
            text = text.rstrip() + "\n\n## Summary\n\n" + summary + "\n"
        except ImportError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    return text


# ---------------------------------------------------------------------------
# Diarization helper
# ---------------------------------------------------------------------------


def _run_diarization(
    audio_path: Path,
    *,
    hf_token: str = "",
    num_speakers: int = 0,
) -> list[dict]:
    """Run speaker diarization on an audio file, returning turns.

    Returns an empty list if diarization is not requested.
    """
    log("Running speaker diarization (this may take a while)...")
    kwargs: dict = {}
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    return diarize.diarize_audio(audio_path, hf_token=hf_token, **kwargs)


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
    clean: bool = False,
    summarize: bool = False,
    summary_model: str = "",
    diarize_turns: list[dict] | None = None,
) -> None:
    """Transcribe a single audio file and write Markdown output.

    If *write_fn* is provided it is called as ``write_fn(text, output)``
    instead of writing directly to *output*.

    If *diarize_turns* is provided, speaker labels are assigned to each
    segment before merging.
    """
    if audio.is_silent(audio_path):
        log(f"Skipping {audio_path.name}: audio is silent")
        return

    result = transcriber.transcribe_audio(audio_path, model=model, language=language)
    segments = transcriber.extract_segments(result)

    if diarize_turns is not None:
        segments = diarize.assign_speakers(segments, diarize_turns)

    if not segments:
        log(f"Skipping {audio_path.name}: no speech detected")
        return

    text = merger.merge_segments(
        [segments], chunk_duration=0, overlap=0, timestamps=timestamps,
        timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
    )
    text = _apply_postprocessing(
        text, clean=clean, summarize=summarize, summary_model=summary_model,
    )
    if write_fn is not None:
        write_fn(text, output)
    else:
        output.write_text(text, encoding="utf-8")
        log(f"Wrote {output}")


def _transcribe_chunk(
    chunk_path: Path,
    model: str,
    language: str | None,
) -> list[dict]:
    """Transcribe a single chunk file, returning its segments.

    Returns an empty list if the chunk is silent or has no speech.
    """
    return scheduler.transcribe_chunk(chunk_path, model, language)


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
    incremental_output: Path | None = None,
    write_fn=None,
    clean: bool = False,
    summarize: bool = False,
    summary_model: str = "",
    diarize_turns: list[dict] | None = None,
) -> None:
    """Split a long audio file into chunks, transcribe each, and merge.

    Chunks are always transcribed sequentially because mlx-whisper
    saturates the GPU with a single inference — parallel threads cause
    Metal command-buffer crashes on Apple Silicon.

    When *incremental* is True, each chunk's preliminary transcription is
    appended to the output file as soon as it completes.  The final merge
    pass then overwrites the file with the properly deduped result.

    If *write_fn* is provided it is called as ``write_fn(text, output)``
    instead of writing directly to *output*.

    If *diarize_turns* is provided, speaker labels are assigned to each
    chunk's segments (using the global timeline) before merging.
    """
    with tempfile.TemporaryDirectory(prefix="scribe-md-chunks-") as tmp:
        tmp_dir = Path(tmp)

        log(f"Splitting into {chunk_seconds}s chunks...")
        chunks = audio.split_audio(audio_path, tmp_dir, chunk_seconds, overlap_seconds)
        draft_output = incremental_output or output

        # Clear the output file before writing incremental results
        if incremental:
            draft_output.parent.mkdir(parents=True, exist_ok=True)
            draft_output.write_text("", encoding="utf-8")

        all_segments = _transcribe_chunks_sequential(
            chunks, model, language, draft_output,
            incremental=incremental,
        )

        # Assign speaker labels if diarization was performed
        if diarize_turns is not None:
            for idx, segs in enumerate(all_segments):
                offset = 0.0 if idx == 0 else idx * chunk_seconds - overlap_seconds
                all_segments[idx] = diarize.assign_speakers(
                    segs, diarize_turns, time_offset=offset,
                )

        text = merger.merge_segments(
            all_segments,
            chunk_duration=chunk_seconds,
            overlap=overlap_seconds,
            timestamps=timestamps,
            timestamp_mode=timestamp_mode,
            paragraph_gap=paragraph_gap,
        )
        text = _apply_postprocessing(
            text, clean=clean, summarize=summarize, summary_model=summary_model,
        )
        if write_fn is not None:
            write_fn(text, output)
        else:
            output.write_text(text, encoding="utf-8")
            log(f"Wrote {output} ({len(chunks)} chunks merged)")


def _transcribe_chunks_sequential(
    chunks: list[Path],
    model: str,
    language: str | None,
    output: Path,
    *,
    incremental: bool = False,
) -> list[list[dict]]:
    """Transcribe chunks one at a time (original sequential pipeline).

    A chunk that *raises* (e.g. a backend crash) is logged as a failure and
    kept empty so the rest of the file still proceeds. But if **every** chunk
    fails, the transcript would be empty for a reason that is not silence — so
    we raise ``TranscriptionError`` to surface a real error (non-zero exit)
    instead of silently writing a blank file.
    """
    all_segments: list[list[dict]] = []
    failures = 0
    last_error: Exception | None = None
    for i, chunk_path in enumerate(chunks):
        console.print(f"  Transcribing chunk {i + 1}/{len(chunks)}...")
        try:
            segments = _transcribe_chunk(chunk_path, model, language)
        except Exception as e:
            log(f"  Chunk {i} failed: {e}")
            segments = []
            failures += 1
            last_error = e
        else:
            if not segments:
                log(f"  Chunk {i}: silent or no speech, skipping")
        all_segments.append(segments)

        if incremental:
            _append_incremental(output, segments)

    if chunks and failures == len(chunks):
        raise TranscriptionError(
            f"All {len(chunks)} chunk(s) failed to transcribe — refusing to "
            f"write an empty transcript. Last error: {last_error}"
        )
    if failures:
        log(
            f"  Warning: {failures}/{len(chunks)} chunk(s) failed; "
            "the transcript is incomplete."
        )

    return all_segments


# ---------------------------------------------------------------------------
# scribe-md file
# ---------------------------------------------------------------------------


@app.command()
def file(
    audio_files: list[Path] = typer.Argument(None, help="Audio file(s) (WAV, MP3, ...)"),
    from_file: _FromFile = None,
    gpus: _Gpus = None,
    output: _Output = None,
    language: _Language = None,
    model: _Model = None,
    timestamps: _Timestamps = None,
    timestamp_mode: _TimestampMode = None,
    paragraph_gap: _ParagraphGap = None,
    chunk_seconds: Optional[float] = typer.Option(
        None, "--chunk-seconds", help="Chunk duration for long files (seconds)",
    ),
    overlap_seconds: _OverlapSeconds = None,
    incremental: Optional[bool] = typer.Option(
        None, "--incremental/--no-incremental",
        help="Write chunks to output file incrementally (default: off)",
    ),
    vault: _Vault = None,
    daily_note: _DailyNote = False,
    frontmatter: _Frontmatter = None,
    clean: _Clean = None,
    summarize: _Summarize = False,
    summary_model: _SummaryModel = None,
    diarize_flag: _Diarize = None,
    hf_token: _HfToken = None,
    num_speakers: _NumSpeakers = None,
) -> None:
    """Transcribe one or more existing audio files to Markdown."""
    _guard_summarize_on_linux(summarize)
    inputs = [Path(p) for p in _collect_inputs([str(p) for p in (audio_files or [])], from_file)]
    _validate_single_output(inputs, output)
    for p in inputs:
        if not p.exists():
            console.print(f"[red]Error:[/red] {p} not found")
            raise typer.Exit(1)
        if p.stat().st_size == 0:
            console.print(f"[red]Error:[/red] {p} is empty (0 bytes)")
            raise typer.Exit(1)

    cfg = load_config()
    opts = _resolve_common_options(
        cfg, model=model, language=language, timestamps=timestamps,
        timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
        overlap_seconds=overlap_seconds, vault=vault, daily_note=daily_note,
        frontmatter=frontmatter, clean=clean, summary_model=summary_model,
        diarize_flag=diarize_flag, hf_token=hf_token, num_speakers=num_speakers,
    )
    r_chunk_seconds = _resolve(chunk_seconds, cfg.chunk_seconds)
    r_incremental = _resolve(incremental, cfg.incremental)
    gpu_ids = _resolve_gpu_ids(_resolve(gpus, cfg.gpus))
    if gpu_ids and r_incremental:
        log("Incremental output disabled under multi-GPU parallelism.")
        r_incremental = False

    try:
        _run_batch(
            inputs, kind="file", single_output=output, cfg=cfg, opts=opts,
            chunk_seconds=r_chunk_seconds, overlap_seconds=opts.overlap_seconds,
            incremental=r_incremental, daily_note=daily_note, summarize=summarize,
            gpu_ids=gpu_ids,
        )
    except (DiarizationError, ImportError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
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
    urls: list[str] = typer.Argument(None, help="YouTube URL(s) or playlist URL(s)"),
    from_file: _FromFile = None,
    gpus: _Gpus = None,
    output: _Output = None,
    language: _Language = None,
    model: _Model = None,
    timestamps: _Timestamps = None,
    timestamp_mode: _TimestampMode = None,
    paragraph_gap: _ParagraphGap = None,
    chunk_seconds: Optional[float] = typer.Option(
        None, "--chunk-seconds", help="Chunk duration for long videos (seconds)",
    ),
    overlap_seconds: _OverlapSeconds = None,
    incremental: Optional[bool] = typer.Option(
        None, "--incremental/--no-incremental",
        help="Write chunks to output file incrementally (default: off)",
    ),
    vault: _Vault = None,
    daily_note: _DailyNote = False,
    frontmatter: _Frontmatter = None,
    clean: _Clean = None,
    summarize: _Summarize = False,
    summary_model: _SummaryModel = None,
    diarize_flag: _Diarize = None,
    hf_token: _HfToken = None,
    num_speakers: _NumSpeakers = None,
) -> None:
    """Transcribe audio from one or more YouTube URLs to Markdown."""
    _guard_summarize_on_linux(summarize)
    inputs = _collect_inputs(list(urls or []), from_file)
    _validate_single_output(inputs, output)
    cfg = load_config()
    opts = _resolve_common_options(
        cfg, model=model, language=language, timestamps=timestamps,
        timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
        overlap_seconds=overlap_seconds, vault=vault, daily_note=daily_note,
        frontmatter=frontmatter, clean=clean, summary_model=summary_model,
        diarize_flag=diarize_flag, hf_token=hf_token, num_speakers=num_speakers,
    )
    r_chunk_seconds = _resolve(chunk_seconds, cfg.chunk_seconds)
    r_incremental = _resolve(incremental, cfg.incremental)
    gpu_ids = _resolve_gpu_ids(_resolve(gpus, cfg.gpus))
    if gpu_ids and r_incremental:
        log("Incremental output disabled under multi-GPU parallelism.")
        r_incremental = False

    # Expand each input URL: playlists expand to (entry_url, title) pairs;
    # single URLs produce one pair. This keeps the loop uniform.
    try:
        expanded: list[tuple[str, str | None]] = []
        for raw_url in inputs:
            entries = downloader.get_playlist_entries(raw_url)
            if len(entries) > 1:
                log(f"Playlist with {len(entries)} videos")
                for i, entry in enumerate(entries):
                    entry_url = entry.get("url") or entry.get("webpage_url", "")
                    entry_title = entry.get("title", f"video_{i}")
                    expanded.append((entry_url, entry_title))
            else:
                single_title = entries[0].get("title") if entries else None
                expanded.append((raw_url, single_title))

        _run_batch(
            expanded, kind="url", single_output=output, cfg=cfg, opts=opts,
            chunk_seconds=r_chunk_seconds, overlap_seconds=opts.overlap_seconds,
            incremental=r_incremental, daily_note=daily_note, summarize=summarize,
            gpu_ids=gpu_ids,
        )
    except (DiarizationError, ImportError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except (AudioConversionError, TranscriptionError, downloader.DownloadError) as e:
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


def _run_batch(
    inputs: list,
    *,
    kind: str,
    single_output: Path | None,
    cfg: ScribeMdConfig,
    opts: _Resolved,
    chunk_seconds: float,
    overlap_seconds: float,
    incremental: bool,
    daily_note: bool,
    summarize: bool,
    gpu_ids: list[int],
) -> None:
    """Dispatch a batch of sources to sequential or parallel pipelines.

    *kind* is ``"file"`` or ``"url"``.  For ``kind="file"`` each element of
    *inputs* is a ``Path``; for ``kind="url"`` each element is a
    ``(url_str, title_or_None)`` tuple.

    When *gpu_ids* is empty the existing sequential helpers are used unchanged;
    when *gpu_ids* has more than one device the scheduler fan-out is used.
    """
    if not gpu_ids:
        _run_batch_sequential(
            inputs, kind=kind, single_output=single_output,
            cfg=cfg, opts=opts, chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds, incremental=incremental,
            daily_note=daily_note, summarize=summarize,
        )
        return

    log(f"Transcribing {len(inputs)} source(s) across GPUs {gpu_ids}...")
    # Warm up once so workers don't race to build the binary / download the model.
    from .backends import get_backend
    from .backends import whispercpp
    if get_backend().name == "whispercpp":
        whispercpp.ensure_whisper_binary()
        whispercpp._ensure_model_file(opts.model)

    def prepare(source) -> PreparedSource:
        tmpdir = Path(tempfile.mkdtemp(prefix="scribe-md-"))
        if kind == "url":
            entry_url, title = source
            raw, resolved_title = downloader.download_audio(entry_url, tmpdir, title=title)
            src_label = f"YouTube: {resolved_title}"
            out = _output_path_for(None, single_output, cfg.output_directory, title=resolved_title)
        else:
            title = source.stem
            raw = source
            src_label = f"file: {source.name}"
            out = _output_path_for(source, single_output, cfg.output_directory)
        converted = tmpdir / "converted.wav"
        log(f"Converting {title} to 16kHz mono...")
        audio.convert_to_16k_mono(raw, converted)
        duration = audio.get_duration(converted)
        if _should_chunk(duration, chunk_seconds):
            chunks = audio.split_audio(converted, tmpdir, chunk_seconds, overlap_seconds)
        else:
            chunks = [converted]
        turns = (
            _run_diarization(converted, hf_token=opts.hf_token, num_speakers=opts.num_speakers)
            if opts.diarize
            else None
        )
        payload = {"out": out, "duration": duration, "turns": turns, "source": src_label}
        return PreparedSource(
            key=out.name, chunk_paths=chunks,
            cleanup=lambda: shutil.rmtree(tmpdir, ignore_errors=True),
            payload=payload,
        )

    def finalize(prepared: PreparedSource, ordered: list[list[dict]]) -> None:
        p = prepared.payload
        if p["turns"] is not None:
            for idx, segs in enumerate(ordered):
                offset = 0.0 if idx == 0 else idx * chunk_seconds - overlap_seconds
                ordered[idx] = diarize.assign_speakers(segs, p["turns"], time_offset=offset)
        text = merger.merge_segments(
            ordered, chunk_duration=chunk_seconds, overlap=overlap_seconds,
            timestamps=opts.ts, timestamp_mode=opts.ts_mode,
            paragraph_gap=opts.paragraph_gap,
        )
        text = _apply_postprocessing(
            text, clean=opts.clean, summarize=summarize, summary_model=opts.summary_model,
        )
        metadata = _build_obsidian_metadata(
            source=p["source"], duration=p["duration"],
            language=opts.language, model=opts.model,
        )
        _write_obsidian_output(
            text, p["out"], opts.vault, daily_note, opts.frontmatter,
            metadata, opts.daily_note_folder,
        )

    summary = scheduler.transcribe_in_parallel(
        inputs, gpu_ids=gpu_ids, model=opts.model, language=opts.language,
        prepare=prepare, finalize=finalize, max_inflight=max(2, len(gpu_ids)),
    )
    log(f"Done: {len(summary.succeeded)} written, {len(summary.skipped)} skipped.")
    if summary.all_failed:
        console.print("[red]Error:[/red] all sources failed to transcribe.")
        raise typer.Exit(1)


def _run_batch_sequential(
    inputs: list,
    *,
    kind: str,
    single_output: Path | None,
    cfg: ScribeMdConfig,
    opts: _Resolved,
    chunk_seconds: float,
    overlap_seconds: float,
    incremental: bool,
    daily_note: bool,
    summarize: bool,
) -> None:
    """Run the existing sequential pipeline for each source in *inputs*.

    For ``kind="file"`` each element is a ``Path``; for ``kind="url"`` each
    element is a ``(url_str, title_or_None)`` tuple.
    """
    if kind == "file":
        for audio_file in inputs:
            out = _output_path_for(audio_file, single_output, cfg.output_directory)
            source = f"file: {audio_file.name}"
            with tempfile.TemporaryDirectory(prefix="scribe-md-") as tmp:
                converted = Path(tmp) / "converted.wav"
                log("Converting to 16kHz mono...")
                audio.convert_to_16k_mono(audio_file, converted)
                file_duration = audio.get_duration(converted)
                metadata = _build_obsidian_metadata(
                    source=source, duration=file_duration,
                    language=opts.language, model=opts.model,
                )

                def write_fn(text: str, output_path: Path, _meta=metadata) -> None:
                    _write_obsidian_output(
                        text, output_path, opts.vault, daily_note, opts.frontmatter,
                        _meta, opts.daily_note_folder,
                    )

                turns = None
                if opts.diarize:
                    turns = _run_diarization(
                        converted, hf_token=opts.hf_token,
                        num_speakers=opts.num_speakers,
                    )
                if _should_chunk(file_duration, chunk_seconds):
                    incremental_enabled, incremental_output = _resolve_incremental_output(
                        out, vault=opts.vault, daily_note=daily_note,
                        incremental=incremental,
                    )
                    _transcribe_chunked(
                        converted, out, opts.model, opts.language, opts.ts,
                        chunk_seconds, overlap_seconds,
                        timestamp_mode=opts.ts_mode, paragraph_gap=opts.paragraph_gap,
                        incremental=incremental_enabled,
                        incremental_output=incremental_output,
                        write_fn=write_fn,
                        clean=opts.clean, summarize=summarize,
                        summary_model=opts.summary_model,
                        diarize_turns=turns,
                    )
                else:
                    _transcribe_single(
                        converted, out, opts.model, opts.language, opts.ts,
                        timestamp_mode=opts.ts_mode, paragraph_gap=opts.paragraph_gap,
                        write_fn=write_fn,
                        clean=opts.clean, summarize=summarize,
                        summary_model=opts.summary_model,
                        diarize_turns=turns,
                    )
    else:
        # kind == "url": each element is (url_str, title_or_None)
        for entry_url, entry_title in inputs:
            try:
                _transcribe_url(
                    entry_url,
                    output=single_output if len(inputs) == 1 else None,
                    model=opts.model, language=opts.language,
                    timestamps=opts.ts, chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    timestamp_mode=opts.ts_mode, paragraph_gap=opts.paragraph_gap,
                    incremental=incremental,
                    vault=opts.vault, daily_note=daily_note,
                    frontmatter=opts.frontmatter,
                    daily_note_folder=opts.daily_note_folder,
                    clean=opts.clean, summarize=summarize,
                    summary_model=opts.summary_model,
                    diarize_enabled=opts.diarize, hf_token=opts.hf_token,
                    num_speakers=opts.num_speakers,
                    output_directory=cfg.output_directory,
                    title=entry_title,
                )
            except DiskFullError:
                raise  # Disk-full is fatal even for multi-url
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"[yellow]Skipping: {e}[/yellow]")


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
    clean: bool = False,
    summarize: bool = False,
    summary_model: str = "",
    diarize_enabled: bool = False,
    hf_token: str = "",
    num_speakers: int = 0,
    output_directory: str = ".",
    title: str | None = None,
) -> None:
    """Download and transcribe a single video URL."""
    with tempfile.TemporaryDirectory(prefix="scribe-md-dl-") as tmp:
        tmp_dir = Path(tmp)

        # Download audio (reusing a known title skips a metadata round-trip)
        raw_audio, title = downloader.download_audio(video_url, tmp_dir, title=title)

        # Convert to 16kHz mono
        converted = tmp_dir / "converted.wav"
        log("Converting to 16kHz mono...")
        audio.convert_to_16k_mono(raw_audio, converted)

        # Determine output path
        if output:
            out = output
        else:
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{sanitize_filename(title)}.md"

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

        # Run diarization on the full audio if requested
        turns = None
        if diarize_enabled:
            turns = _run_diarization(
                converted, hf_token=hf_token, num_speakers=num_speakers,
            )

        if _should_chunk(duration, chunk_seconds):
            incremental_enabled, incremental_output = _resolve_incremental_output(
                out, vault=vault, daily_note=daily_note,
                incremental=incremental,
            )
            _transcribe_chunked(
                converted, out, model, language, timestamps,
                chunk_seconds, overlap_seconds,
                timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
                incremental=incremental_enabled,
                incremental_output=incremental_output,
                write_fn=write_fn,
                clean=clean, summarize=summarize,
                summary_model=summary_model,
                diarize_turns=turns,
            )
        else:
            _transcribe_single(
                converted, out, model, language, timestamps,
                timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
                write_fn=write_fn,
                clean=clean, summarize=summarize,
                summary_model=summary_model,
                diarize_turns=turns,
            )


# ---------------------------------------------------------------------------
# scribe-md live
# ---------------------------------------------------------------------------


@app.command()
def live(
    output: _Output = None,
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Recording duration (seconds)"),
    language: _Language = None,
    model: _Model = None,
    timestamps: _Timestamps = None,
    timestamp_mode: _TimestampMode = None,
    paragraph_gap: _ParagraphGap = None,
    chunk_seconds: Optional[float] = typer.Option(
        None, "--chunk-seconds", help="Enable chunked pipeline (transcribe every N seconds)",
    ),
    overlap_seconds: _OverlapSeconds = None,
    keep_audio: Optional[bool] = typer.Option(None, "--keep-audio", help="Keep intermediate WAV files"),
    app_name: Optional[list[str]] = typer.Option(None, "--app", "-a", help="Capture from specific app(s) (repeatable)"),
    incremental: Optional[bool] = typer.Option(
        None, "--incremental/--no-incremental",
        help="Write chunks to output file incrementally (default: on for live)",
    ),
    vault: _Vault = None,
    daily_note: _DailyNote = False,
    frontmatter: _Frontmatter = None,
    clean: _Clean = None,
    summarize: _Summarize = False,
    summary_model: _SummaryModel = None,
    diarize_flag: _Diarize = None,
    hf_token: _HfToken = None,
    num_speakers: _NumSpeakers = None,
) -> None:
    """Capture and transcribe system audio in real-time."""
    if platform_support.is_linux():
        console.print(
            "[red]Error:[/red] Live system-audio capture is macOS-only for now. "
            "Use 'scribe-md file' or 'scribe-md url' on Linux."
        )
        raise typer.Exit(1)
    cfg = load_config()
    opts = _resolve_common_options(
        cfg, model=model, language=language, timestamps=timestamps,
        timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
        overlap_seconds=overlap_seconds, vault=vault, daily_note=daily_note,
        frontmatter=frontmatter, clean=clean, summary_model=summary_model,
        diarize_flag=diarize_flag, hf_token=hf_token, num_speakers=num_speakers,
    )
    r_chunk_seconds = _resolve(chunk_seconds, 0)  # live default: no chunking
    r_incremental = _resolve(incremental, cfg.live_incremental)
    r_keep_audio = _resolve(keep_audio, cfg.keep_audio)
    if output:
        r_output = output
    else:
        out_dir = Path(cfg.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        r_output = out_dir / "transcription.md"

    # Build source and metadata for Obsidian
    apps = app_name if app_name else None
    if apps:
        source = f"live: {', '.join(apps)}"
    else:
        source = "live: system audio"

    metadata = _build_obsidian_metadata(
        source=source, duration=None,
        language=opts.language, model=opts.model,
    )

    def write_fn(text: str, output_path: Path) -> None:
        _write_obsidian_output(
            text, output_path, opts.vault, daily_note, opts.frontmatter,
            metadata, opts.daily_note_folder,
        )

    try:
        if r_chunk_seconds > 0:
            incremental_enabled, incremental_output = _resolve_incremental_output(
                r_output, vault=opts.vault, daily_note=daily_note,
                incremental=r_incremental,
            )
            _live_chunked(
                r_output, duration, opts.language, opts.model, opts.ts,
                r_chunk_seconds, opts.overlap_seconds, r_keep_audio, apps,
                timestamp_mode=opts.ts_mode, paragraph_gap=opts.paragraph_gap,
                incremental=incremental_enabled,
                incremental_output=incremental_output,
                write_fn=write_fn,
                clean=opts.clean, summarize=summarize,
                summary_model=opts.summary_model,
                diarize_enabled=opts.diarize, hf_token=opts.hf_token,
                num_speakers=opts.num_speakers,
            )
        else:
            _live_single(
                r_output, duration, opts.language, opts.model, opts.ts, r_keep_audio, apps,
                timestamp_mode=opts.ts_mode, paragraph_gap=opts.paragraph_gap,
                write_fn=write_fn,
                clean=opts.clean, summarize=summarize,
                summary_model=opts.summary_model,
                diarize_enabled=opts.diarize, hf_token=opts.hf_token,
                num_speakers=opts.num_speakers,
            )
    except (DiarizationError, ImportError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
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
    clean: bool = False,
    summarize: bool = False,
    summary_model: str = "",
    diarize_enabled: bool = False,
    hf_token: str = "",
    num_speakers: int = 0,
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
            capture.terminate_capture(proc)

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

        # Run diarization on the full recording if requested
        turns = None
        if diarize_enabled:
            turns = _run_diarization(
                converted, hf_token=hf_token, num_speakers=num_speakers,
            )

        _transcribe_single(
            converted, output, model, language, timestamps,
            timestamp_mode=timestamp_mode, paragraph_gap=paragraph_gap,
            write_fn=write_fn,
            clean=clean, summarize=summarize,
            summary_model=summary_model,
            diarize_turns=turns,
        )

        if keep_audio:
            import shutil
            saved = output.parent / f"recording_{output.stem}.wav"
            saved.parent.mkdir(parents=True, exist_ok=True)
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
    incremental_output: Path | None = None,
    write_fn=None,
    clean: bool = False,
    summarize: bool = False,
    summary_model: str = "",
    diarize_enabled: bool = False,
    hf_token: str = "",
    num_speakers: int = 0,
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
        draft_output = incremental_output or output

        # Clear the output file before writing incremental results
        if incremental:
            draft_output.parent.mkdir(parents=True, exist_ok=True)
            draft_output.write_text("", encoding="utf-8")

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
                        _append_incremental(draft_output, segments)
                except DiskFullError:
                    raise  # Disk-full is fatal — stop immediately
                except Exception as e:
                    log(f"Chunk {chunk_idx} failed: {e}")

                chunk_idx += 1

            proc.wait()
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            capture.terminate_capture(proc)

        if not chunk_jsons:
            log("No chunks were transcribed.")
            raise typer.Exit(1)

        # Merge all chunks
        all_segments: list[list[dict]] = []
        for cj in sorted(chunk_jsons):
            data = json.loads(cj.read_text(encoding="utf-8"))
            all_segments.append(data.get("segments", []))

        # Diarization for live chunked mode is not supported (no full audio
        # available during capture).  Users should use single-capture mode
        # with --diarize, or use file mode on the saved audio.
        if diarize_enabled:
            log("Note: diarization is not supported in live chunked mode. Skipping.")

        text = merger.merge_segments(
            all_segments,
            chunk_duration=chunk_seconds,
            overlap=overlap_seconds,
            timestamps=timestamps,
            timestamp_mode=timestamp_mode,
            paragraph_gap=paragraph_gap,
        )
        text = _apply_postprocessing(
            text, clean=clean, summarize=summarize, summary_model=summary_model,
        )
        if write_fn is not None:
            write_fn(text, output)
        else:
            output.write_text(text, encoding="utf-8")
            log(f"Done: {output} ({len(chunk_jsons)} chunks)")

        if keep_audio:
            # Copy to a persistent location next to the markdown output before
            # the temp dir is cleaned up.
            import shutil
            saved_dir = output.parent / f"chunks_{output.stem}"
            saved_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_dir, saved_dir, dirs_exist_ok=True)
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
