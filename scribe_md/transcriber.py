"""Whisper transcription via mlx-whisper."""

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


def resolve_model(model: str) -> str:
    """Resolve a model preset name or full path to a HF repo path."""
    return MODEL_PRESETS.get(model, model)


class TranscriptionError(RuntimeError):
    """Raised when transcription fails due to invalid input or runtime errors."""


def transcribe_audio(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
) -> dict:
    """Transcribe a single audio file, returning the raw mlx-whisper result."""
    # Validate input file
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    file_size = audio_path.stat().st_size
    if file_size == 0:
        raise TranscriptionError(
            f"Audio file is empty (0 bytes): {audio_path.name}. "
            "The recording may have failed or been interrupted."
        )

    # A valid WAV header is at least 44 bytes
    if file_size < 44:
        raise TranscriptionError(
            f"Audio file is too small ({file_size} bytes): {audio_path.name}. "
            "The file may be corrupt."
        )

    import mlx_whisper

    kwargs = {"path_or_hf_repo": resolve_model(model)}
    if language:
        kwargs["language"] = language

    log(f"Transcribing {audio_path.name}...")
    try:
        return mlx_whisper.transcribe(str(audio_path), **kwargs)
    except Exception as e:
        raise TranscriptionError(
            f"Transcription failed for {audio_path.name}: {e}"
        ) from e


def extract_segments(
    result: dict,
    no_speech_threshold: float = 0.6,
) -> list[dict]:
    """Extract normalized segments from a Whisper result.

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
