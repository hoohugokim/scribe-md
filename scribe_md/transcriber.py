"""Transcription facade.

Validates input, then delegates to the platform-selected backend
(scribe_md.backends). MODEL_PRESETS/DEFAULT_MODEL remain here as the
canonical, user-facing preset vocabulary (the MLX backend reuses them; the
whisper.cpp backend maps the same names to GGML files).
"""

from pathlib import Path

from .utils import log

MODEL_PRESETS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
DEFAULT_MODEL = "large-v3"


class TranscriptionError(RuntimeError):
    """Raised when transcription fails due to invalid input or runtime errors."""


def transcribe_audio(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str | None = None,
) -> dict:
    """Validate audio_path and transcribe it via the active backend."""
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    file_size = audio_path.stat().st_size
    if file_size == 0:
        raise TranscriptionError(
            f"Audio file is empty (0 bytes): {audio_path.name}. "
            "The recording may have failed or been interrupted."
        )
    if file_size < 44:
        raise TranscriptionError(
            f"Audio file is too small ({file_size} bytes): {audio_path.name}. "
            "The file may be corrupt."
        )

    from .backends import get_backend

    backend = get_backend()
    log(f"Transcribing {audio_path.name} via {backend.describe()}...")
    try:
        return backend.transcribe(
            audio_path, model=model, language=language, device=device
        )
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(
            f"Transcription failed for {audio_path.name}: {e}"
        ) from e


def extract_segments(
    result: dict,
    no_speech_threshold: float = 0.6,
) -> list[dict]:
    """Extract normalized segments from a backend result.

    Filters out segments with high no_speech_prob to prevent hallucination
    on silent audio (e.g. Whisper generating "자막제공자" on silence).
    """
    segments = []
    for s in result.get("segments", []):
        if s.get("no_speech_prob", 0) > no_speech_threshold:
            continue
        text = s["text"].strip()
        if not text:
            continue
        segments.append({"start": s["start"], "end": s["end"], "text": text})
    return segments
