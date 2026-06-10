"""Transcription backend selection.

``get_backend()`` picks MLX on macOS and whisper.cpp on Linux. Backend modules
are imported lazily so that importing this package never pulls in
platform-specific dependencies (e.g. ``mlx_whisper``).
"""

import os
import sys

from ..platform_support import is_macos, is_linux
from .base import Backend

__all__ = ["Backend", "get_backend"]


def get_backend() -> Backend:
    """Return the transcription backend for the current platform.

    The ``SCRIBE_MD_BACKEND`` env var (``mlx`` or ``whispercpp``) overrides
    auto-detection, primarily for testing.
    """
    override = os.environ.get("SCRIBE_MD_BACKEND", "").strip().lower()
    if override:
        if override == "mlx":
            from .mlx import MLXBackend
            return MLXBackend()
        if override in ("whispercpp", "whisper.cpp"):
            from .whispercpp import WhisperCppBackend
            return WhisperCppBackend()
        raise ValueError(
            f"Unknown SCRIBE_MD_BACKEND={override!r}; expected 'mlx' or 'whispercpp'."
        )

    if is_macos():
        from .mlx import MLXBackend
        return MLXBackend()
    if is_linux():
        from .whispercpp import WhisperCppBackend
        return WhisperCppBackend()
    raise RuntimeError(
        f"scribe-md has no transcription backend for platform {sys.platform!r}."
    )
