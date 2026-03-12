"""Whisper transcription via mlx-whisper."""

from pathlib import Path

from .utils import log


def transcribe_audio(
    audio_path: Path,
    model: str = "mlx-community/whisper-large-v3-mlx",
    language: str | None = None,
) -> dict:
    """Transcribe a single audio file, returning the raw mlx-whisper result."""
    import mlx_whisper

    kwargs = {"path_or_hf_repo": model}
    if language:
        kwargs["language"] = language

    log(f"Transcribing {audio_path.name}...")
    return mlx_whisper.transcribe(str(audio_path), **kwargs)


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
