"""Transcription backend selection.

``get_backend()`` picks:

* explicit ``engine`` / ``SCRIBE_MD_BACKEND`` when set
  (``mlx``, ``whispercpp``, ``qwen``, ``nemotron``)
* otherwise auto policy: MLX Whisper on macOS; on Linux, Nemotron when CUDA
  is present and a local Nemotron server is healthy, else whisper.cpp

Backend modules are imported lazily so importing this package never pulls in
platform-specific dependencies.
"""

from __future__ import annotations

import os
import sys

from ..platform_support import is_linux, is_macos
from .base import Backend

__all__ = ["Backend", "get_backend", "normalize_engine"]

_ENGINE_ALIASES = {
    "mlx": "mlx",
    "whisper": "mlx",
    "whispercpp": "whispercpp",
    "whisper.cpp": "whispercpp",
    "qwen": "qwen",
    "qwen3": "qwen",
    "qwen3-asr": "qwen",
    "nemotron": "nemotron",
    "nemotron3.5": "nemotron",
    "nemotron-3.5": "nemotron",
    "n3.5t": "nemotron",
}


def normalize_engine(raw: str) -> str:
    """Map a user engine string to a canonical backend name."""
    key = raw.strip().lower()
    if key not in _ENGINE_ALIASES:
        raise ValueError(
            f"Unknown engine {raw!r}; expected one of: "
            "mlx, qwen, whispercpp, nemotron."
        )
    return _ENGINE_ALIASES[key]


def _cuda_available() -> bool:
    from ..gpu import discover_cuda_devices

    return bool(discover_cuda_devices())


def _nemotron_ready() -> bool:
    from . import nemotron as nemo

    base = os.environ.get("SCRIBE_MD_NEMOTRON_URL", nemo.DEFAULT_BASE_URL)
    return nemo.health_ok(base)


def _instantiate(name: str) -> Backend:
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend()
    if name == "whispercpp":
        from .whispercpp import WhisperCppBackend

        return WhisperCppBackend()
    if name == "qwen":
        from .qwen import DEFAULT_MODEL as QWEN_DEFAULT
        from .qwen import QwenBackend

        return QwenBackend(
            default_model=os.environ.get("SCRIBE_MD_QWEN_MODEL", QWEN_DEFAULT),
        )
    if name == "nemotron":
        from . import nemotron as nemo

        timeout_raw = os.environ.get("SCRIBE_MD_NEMOTRON_TIMEOUT", "")
        timeout = float(timeout_raw) if timeout_raw else nemo.DEFAULT_TIMEOUT
        return nemo.NemotronBackend(
            base_url=os.environ.get("SCRIBE_MD_NEMOTRON_URL", nemo.DEFAULT_BASE_URL),
            model=os.environ.get("SCRIBE_MD_NEMOTRON_MODEL", nemo.DEFAULT_MODEL),
            prompt=os.environ.get("SCRIBE_MD_NEMOTRON_PROMPT", nemo.DEFAULT_PROMPT),
            timeout_seconds=timeout,
        )
    raise ValueError(f"Unknown backend {name!r}")


def _auto_backend() -> Backend:
    if is_macos():
        return _instantiate("mlx")
    if is_linux():
        if _cuda_available() and _nemotron_ready():
            return _instantiate("nemotron")
        return _instantiate("whispercpp")
    raise RuntimeError(
        f"scribe-md has no transcription backend for platform {sys.platform!r}."
    )


def get_backend(engine: str | None = None) -> Backend:
    """Return the transcription backend for *engine* or the auto policy.

    Precedence: ``engine`` argument → ``SCRIBE_MD_BACKEND`` env → auto.
    """
    raw = (engine or os.environ.get("SCRIBE_MD_BACKEND", "") or "").strip()
    if raw:
        return _instantiate(normalize_engine(raw))
    return _auto_backend()
