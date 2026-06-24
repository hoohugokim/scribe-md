"""Backend protocol shared by all transcription engines."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """A transcription engine.

    Implementations must return the same result shape as mlx-whisper:
    ``{"segments": [{"start": float, "end": float, "text": str,
    "no_speech_prob": float}, ...]}``.
    """

    name: str

    def transcribe(
        self, audio_path: Path, *, model: str, language: str | None,
        device: str | None = None,
    ) -> dict:
        """Transcribe a 16 kHz mono WAV, returning a result dict.

        ``device`` optionally pins a specific accelerator (e.g. a CUDA device
        index); backends that cannot target a device ignore it.
        """
        ...

    def describe(self) -> str:
        """Human-readable engine + device string for logging."""
        ...
