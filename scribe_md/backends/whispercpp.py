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
BUILD_DIR = VENDOR_DIR / "build"
WHISPER_BIN = BUILD_DIR / "bin" / "whisper-cli"
ACCEL_MARKER = BUILD_DIR / ".scribe_accel"  # records the accel the binary was built with
MODEL_CACHE = Path.home() / ".cache" / "scribe-md" / "models"
HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"

# The smallest GGML weights (tiny) are tens of MB; anything well below this is a
# truncated download or an HTML error page, never a real model.
_MIN_MODEL_BYTES = 1_000_000

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


def _probe_ok(args: list[str]) -> bool:
    """Run a quick GPU probe; True only if it exits 0 (a usable device).

    Mere presence of a tool on PATH is not enough — pixi installs
    ``vulkan-tools`` on every Linux box, so ``vulkaninfo`` exists even where no
    GPU/driver does. Running it and checking the exit code distinguishes a
    usable device from a headless/driverless machine.
    """
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def detect_accel() -> str:
    """Pick the best build-time accelerator: cuda > vulkan > cpu.

    Overridable with ``SCRIBE_MD_WHISPER_ACCEL=cuda|vulkan|cpu``. Auto-detection
    requires both the tool on PATH *and* a successful probe, so a missing
    driver falls back to CPU instead of building for a device that isn't there.
    """
    override = os.environ.get("SCRIBE_MD_WHISPER_ACCEL", "").strip().lower()
    if override in ("cuda", "vulkan", "cpu"):
        return override
    if shutil.which("nvcc") and shutil.which("nvidia-smi") and _probe_ok(["nvidia-smi"]):
        return "cuda"
    if shutil.which("vulkaninfo") and _probe_ok(["vulkaninfo"]):
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
def _read_built_accel() -> str | None:
    """Return the accelerator recorded for the cached binary, if any."""
    try:
        return ACCEL_MARKER.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _record_built_accel(accel: str) -> None:
    """Record which accelerator the freshly built binary uses.

    A missing marker only costs a future rebuild, never correctness, so any
    write failure is swallowed.
    """
    try:
        ACCEL_MARKER.write_text(accel, encoding="utf-8")
    except OSError:
        pass


def ensure_whisper_binary() -> Path:
    """Build whisper.cpp on first use; return the whisper-cli path.

    Rebuilds when the cached binary was built for a different accelerator than
    the one now detected (e.g. a CUDA toolkit was installed, or
    ``SCRIBE_MD_WHISPER_ACCEL`` changed). A binary with no recorded
    accelerator — an older build, or one produced by ``pixi run
    build-whisper`` — is trusted as-is to avoid surprise rebuilds.
    """
    accel = detect_accel()

    if WHISPER_BIN.exists():
        built = _read_built_accel()
        if built is None or built == accel:
            return WHISPER_BIN
        log(f"Rebuilding whisper.cpp: accelerator changed ({built} -> {accel}).")
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

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

    flags = accel_cmake_flags(accel)
    log(f"Building whisper.cpp (accel={accel})...")
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
    _record_built_accel(accel)
    return WHISPER_BIN


def _validate_download(path: Path, headers, fname: str, url: str) -> None:
    """Reject obviously-bad downloads before they poison the cache.

    Catches the two failure modes ``urlretrieve`` does not: a 200 response whose
    body is an HTML error page (captive portal / proxy / HF maintenance), and a
    body silently truncated to a few bytes.
    """
    content_type = ""
    if headers is not None:
        content_type = (headers.get("Content-Type") or "").lower()
    if "html" in content_type:
        raise WhisperCppError(
            f"Download of {fname} from {url} returned HTML, not a model file "
            "(a proxy, captive portal, or error page?). Check your connection."
        )
    size = path.stat().st_size
    if size < _MIN_MODEL_BYTES:
        raise WhisperCppError(
            f"Downloaded model {fname} is only {size} bytes — too small to be a "
            "valid model (truncated download or error page). Try again."
        )


def _ensure_model_file(model: str) -> Path:
    """Return a local GGML model path, downloading it on first use."""
    # Allow a direct path to an existing .bin file.
    direct = Path(model)
    if direct.is_file():
        return direct

    fname = resolve_model_filename(model)
    dest = MODEL_CACHE / fname
    # A model name from an auto-discovered .scribe-md.toml is semi-trusted;
    # never let it steer the download outside the cache.
    if not dest.resolve().is_relative_to(MODEL_CACHE.resolve()):
        raise WhisperCppError(
            f"Model name {model!r} resolves outside the model cache; refusing."
        )
    if dest.exists():
        return dest

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    url = HF_BASE + fname
    log(f"Downloading whisper.cpp model {fname} (first use only)...")
    # A process-unique temp name keeps two concurrent runs from clobbering a
    # shared path; os.replace then publishes the result atomically.
    tmp = dest.with_name(f"{fname}.{os.getpid()}.tmp")
    try:
        _, headers = urllib.request.urlretrieve(url, tmp)
        _validate_download(tmp, headers, fname, url)
    except WhisperCppError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp.unlink(missing_ok=True)  # don't leave a partial download behind
        raise WhisperCppError(f"Failed to download model {fname} from {url}: {e}")
    os.replace(tmp, dest)
    return dest


class WhisperCppBackend:
    """Transcription via the whisper.cpp whisper-cli subprocess."""

    name = "whispercpp"

    def describe(self) -> str:
        # Report what the binary was actually built with (the marker), falling
        # back to live detection only when no build has been recorded yet.
        return f"whisper.cpp ({_read_built_accel() or detect_accel()})"

    def transcribe(
        self, audio_path: Path, *, model: str, language: str | None,
        device: str | None = None,
    ) -> dict:
        binary = ensure_whisper_binary()
        model_path = _ensure_model_file(model)
        env = None
        if device is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
        with tempfile.TemporaryDirectory() as td:
            out_prefix = Path(td) / "out"
            cmd = _build_command(binary, model_path, audio_path, out_prefix, language)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
