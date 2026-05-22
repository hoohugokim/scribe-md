"""whisper.cpp transcription backend (Linux).

Drives the ``whisper-cli`` binary built from the vendored submodule as a
subprocess, so a Vulkan/CUDA crash cannot take down the Python process. The GPU
engine (CPU / Vulkan / CUDA) is a build-time flag of the same binary.
"""

from __future__ import annotations

import json
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


# --- Build, download, transcribe -------------------------------------------
def ensure_whisper_binary() -> Path:
    """Build whisper.cpp on first use; return the whisper-cli path."""
    if WHISPER_BIN.exists():
        return WHISPER_BIN

    if not (VENDOR_DIR / "CMakeLists.txt").exists():
        raise WhisperCppError(
            "whisper.cpp source not found. Initialize the submodule:\n"
            "  git submodule update --init vendor/whisper.cpp"
        )
    if shutil.which("cmake") is None:
        raise WhisperCppError(
            "'cmake' not found. Install the Linux build toolchain with "
            "'pixi install', or 'sudo apt install cmake build-essential'."
        )

    accel = detect_accel()
    flags = accel_cmake_flags(accel)
    log(f"Building whisper.cpp (accel={accel}, first run only)...")
    try:
        subprocess.run(
            ["cmake", "-B", "build", *flags],
            cwd=VENDOR_DIR, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["cmake", "--build", "build", "-j", "--config", "Release"],
            cwd=VENDOR_DIR, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise WhisperCppError(
            f"whisper.cpp build failed (accel={accel}): {e.stderr or e}"
        )

    if not WHISPER_BIN.exists():
        raise WhisperCppError(
            f"Build succeeded but whisper-cli not found at {WHISPER_BIN}."
        )
    return WHISPER_BIN


def _ensure_model_file(model: str) -> Path:
    """Return a local GGML model path, downloading it on first use."""
    # Allow a direct path to an existing .bin file.
    direct = Path(model)
    if direct.is_file():
        return direct

    fname = resolve_model_filename(model)
    dest = MODEL_CACHE / fname
    if dest.exists():
        return dest

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    url = HF_BASE + fname
    log(f"Downloading whisper.cpp model {fname} (first use only)...")
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)  # don't leave a partial download behind
        raise WhisperCppError(f"Failed to download model {fname} from {url}: {e}")
    tmp.rename(dest)
    return dest


class WhisperCppBackend:
    """Transcription via the whisper.cpp whisper-cli subprocess."""

    name = "whispercpp"

    def resolve_model(self, model: str) -> str:
        return resolve_model_filename(model)

    def describe(self) -> str:
        return f"whisper.cpp ({detect_accel()})"

    def transcribe(self, audio_path: Path, *, model: str, language: str | None) -> dict:
        binary = ensure_whisper_binary()
        model_path = _ensure_model_file(model)
        with tempfile.TemporaryDirectory() as td:
            out_prefix = Path(td) / "out"
            cmd = _build_command(binary, model_path, audio_path, out_prefix, language)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            except OSError as e:
                raise WhisperCppError(f"Failed to run whisper-cli: {e}")
            if result.returncode != 0:
                raise WhisperCppError(
                    f"whisper.cpp failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            json_path = out_prefix.with_suffix(".json")
            if not json_path.exists():
                raise WhisperCppError(
                    f"whisper.cpp produced no JSON output at {json_path}."
                )
            data = json.loads(json_path.read_text(encoding="utf-8"))
        return parse_whispercpp_json(data)


def _main() -> None:
    """python -m scribe_md.backends.whispercpp build pre-builds the binary."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        path = ensure_whisper_binary()
        print(f"Built whisper.cpp: {path}")
    else:
        print("Usage: python -m scribe_md.backends.whispercpp build")


if __name__ == "__main__":
    _main()
