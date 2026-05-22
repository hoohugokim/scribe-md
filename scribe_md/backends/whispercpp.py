"""whisper.cpp transcription backend (Linux).

Drives the ``whisper-cli`` binary built from the vendored submodule as a
subprocess, so a Vulkan/CUDA crash cannot take down the Python process. The GPU
engine (CPU / Vulkan / CUDA) is a build-time flag of the same binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from ..transcriber import TranscriptionError
from ..utils import log

# --- Paths -----------------------------------------------------------------
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "whisper.cpp"
WHISPER_BIN = VENDOR_DIR / "build" / "bin" / "whisper-cli"
MODEL_CACHE = Path.home() / ".cache" / "scribe-md" / "models"
HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"

# --- Model presets (GGML single-file weights) ------------------------------
GGML_MODELS = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large": "ggml-large-v3.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}


class WhisperCppError(TranscriptionError):
    """Raised when the whisper.cpp build or invocation fails."""


# --- Pure helpers (no GPU, no subprocess) ----------------------------------
def resolve_model_filename(model: str) -> str:
    """Map a preset name to a GGML filename; pass through unknown values."""
    return GGML_MODELS.get(model, model)


def parse_whispercpp_json(data: dict) -> dict:
    """Convert whisper.cpp ``-oj`` JSON into the shared result shape.

    whisper.cpp emits ``transcription[].offsets`` in **milliseconds** and has no
    per-segment no-speech probability, so we default it to 0.0 (below the 0.6
    filter threshold in ``extract_segments``).
    """
    segments = []
    for item in data.get("transcription", []):
        offsets = item.get("offsets", {})
        segments.append({
            "start": offsets.get("from", 0) / 1000.0,
            "end": offsets.get("to", 0) / 1000.0,
            "text": item.get("text", "").strip(),
            "no_speech_prob": 0.0,
        })
    return {"segments": segments}


def accel_cmake_flags(accel: str) -> list[str]:
    """Map an accelerator name to whisper.cpp cmake flags."""
    return {"cuda": ["-DGGML_CUDA=1"], "vulkan": ["-DGGML_VULKAN=1"], "cpu": []}[accel]


def detect_accel() -> str:
    """Pick the best build-time accelerator: cuda > vulkan > cpu.

    Overridable with ``SCRIBE_MD_WHISPER_ACCEL=cuda|vulkan|cpu``.
    """
    override = os.environ.get("SCRIBE_MD_WHISPER_ACCEL", "").strip().lower()
    if override in ("cuda", "vulkan", "cpu"):
        return override
    if shutil.which("nvcc") and shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("vulkaninfo"):
        return "vulkan"
    return "cpu"


def _build_command(
    binary: str | Path,
    model_path: Path,
    wav: Path,
    out_prefix: Path,
    language: str | None,
) -> list[str]:
    """Build the whisper-cli argument list."""
    return [
        str(binary),
        "-m", str(model_path),
        "-f", str(wav),
        "-oj",
        "-of", str(out_prefix),
        "--no-prints",
        "-l", language if language else "auto",
    ]


# Replaced with the real implementation in Task 5.
class WhisperCppBackend:
    name = "whispercpp"
