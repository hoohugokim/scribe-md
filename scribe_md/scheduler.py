"""Multi-GPU parallel transcription scheduler.

Owns concurrency, GPU assignment, per-source ordering, and bounded resource
use. Decoupled from CLI/Obsidian specifics via prepare/finalize callbacks.
"""

from __future__ import annotations

from pathlib import Path

from . import audio, transcriber
from .utils import log


def transcribe_chunk(
    chunk_path: Path,
    model: str,
    language: str | None,
    device: str | None = None,
) -> list[dict]:
    """Transcribe one chunk, returning its segments ([] if silent/no speech)."""
    if audio.is_silent(chunk_path):
        return []
    result = transcriber.transcribe_audio(
        chunk_path, model=model, language=language, device=device
    )
    return transcriber.extract_segments(result)
