"""Qwen3-ASR transcription backend (macOS / Apple Silicon via mlx-audio)."""

from __future__ import annotations

from pathlib import Path

from ..platform_support import is_macos
from ..transcriber import TranscriptionError
from .segments import segments_from_text, segments_from_timed

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"

QWEN_PRESETS = {
    "qwen": DEFAULT_MODEL,
    "qwen-1.7b": DEFAULT_MODEL,
    "qwen-0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
}


def _wav_duration_seconds(path: Path) -> float:
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (OSError, wave.Error):
        return 0.0


def _result_from_mlx_audio(raw, *, duration: float) -> dict:
    """Normalize mlx-audio STT outputs (object or dict) to segments."""
    if raw is None:
        return {"segments": []}

    if isinstance(raw, dict):
        if "segments" in raw and isinstance(raw["segments"], list):
            return segments_from_timed(raw["segments"])
        text = raw.get("text", "")
        return segments_from_text(str(text), duration=duration)

    text = getattr(raw, "text", None)
    segs = getattr(raw, "segments", None)
    if segs:
        items = []
        for s in segs:
            if isinstance(s, dict):
                items.append(s)
            else:
                items.append(
                    {
                        "start": getattr(s, "start", 0.0),
                        "end": getattr(s, "end", 0.0),
                        "text": getattr(s, "text", ""),
                    }
                )
        result = segments_from_timed(items)
        if result["segments"]:
            return result
    if text is not None:
        return segments_from_text(str(text), duration=duration)
    return segments_from_text(str(raw), duration=duration)


class QwenBackend:
    """Transcription via Qwen3-ASR through mlx-audio."""

    name = "qwen"

    def __init__(self, *, default_model: str = DEFAULT_MODEL) -> None:
        self.default_model = default_model

    def resolve_model(self, model: str) -> str:
        if not model or model in ("large-v3", "large", "default"):
            return self.default_model
        return QWEN_PRESETS.get(model, model)

    def describe(self) -> str:
        return "Qwen3-ASR (MLX)"

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str | None,
        device: str | None = None,
    ) -> dict:
        _ = device
        if not is_macos():
            raise TranscriptionError(
                "Qwen3-ASR via mlx-audio requires macOS Apple Silicon. "
                "On Linux use --engine whispercpp or --engine nemotron."
            )

        repo = self.resolve_model(model)
        try:
            from mlx_audio.stt.utils import load_model
            from mlx_audio.stt.generate import generate_transcription
        except ImportError as e:
            raise TranscriptionError(
                "Qwen engine requires mlx-audio. Install on macOS with: "
                "pixi run pip install 'mlx-audio[stt]' "
                "(or pip install mlx-audio)."
            ) from e

        try:
            stt_model = load_model(repo)
            # mlx-audio APIs vary slightly by version; prefer generate_transcription.
            try:
                raw = generate_transcription(
                    model=stt_model,
                    audio_path=str(audio_path),
                    language=language,
                )
            except TypeError:
                # Older signatures may not accept language=
                raw = generate_transcription(
                    model=stt_model,
                    audio_path=str(audio_path),
                )
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Qwen3-ASR transcription failed for {audio_path.name}: {e}"
            ) from e

        duration = _wav_duration_seconds(audio_path)
        return _result_from_mlx_audio(raw, duration=duration)
