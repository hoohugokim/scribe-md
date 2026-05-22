"""MLX (Apple Silicon) transcription backend."""

from __future__ import annotations

from pathlib import Path

from ..transcriber import MODEL_PRESETS


class MLXBackend:
    """Transcription via mlx-whisper. macOS / Apple Silicon only."""

    name = "mlx"

    def resolve_model(self, model: str) -> str:
        return MODEL_PRESETS.get(model, model)

    def describe(self) -> str:
        return "MLX (Apple Silicon)"

    def transcribe(self, audio_path: Path, *, model: str, language: str | None) -> dict:
        import mlx_whisper

        kwargs = {"path_or_hf_repo": self.resolve_model(model)}
        if language:
            kwargs["language"] = language
        return mlx_whisper.transcribe(str(audio_path), **kwargs)
