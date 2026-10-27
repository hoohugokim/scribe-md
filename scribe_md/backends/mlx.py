"""MLX (Apple Silicon) transcription backend."""

from __future__ import annotations

from pathlib import Path

from ..transcriber import MODEL_PRESETS  # shared preset vocabulary (facade owns it)


class MLXBackend:
    """Transcription via mlx-whisper. macOS / Apple Silicon only."""

    name = "mlx"

    def resolve_model(self, model: str) -> str:
        return MODEL_PRESETS.get(model, model)

    def describe(self) -> str:
        return "MLX (Apple Silicon)"

    def transcribe(
        self, audio_path: Path, *, model: str, language: str | None,
        device: str | None = None,
    ) -> dict:
        # device is ignored: Apple Silicon is a single unified-memory device.
        # Deferred import: mlx_whisper is macOS-only and unavailable on Linux.
        import mlx_whisper

        kwargs = {"path_or_hf_repo": self.resolve_model(model)}
        if language:
            kwargs["language"] = language
        return mlx_whisper.transcribe(str(audio_path), **kwargs)
