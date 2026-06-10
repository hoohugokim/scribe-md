# Linux Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scribe-md file` and `scribe-md url` run on Linux (Pop!_OS / Ubuntu 24.04) from one codebase, transcribing via whisper.cpp with Vulkan-by-default GPU acceleration (CUDA opt-in, CPU fallback), while macOS keeps using MLX.

**Architecture:** Introduce a `scribe_md/backends/` package with a `Backend` protocol. `get_backend()` selects `MLXBackend` on macOS and `WhisperCppBackend` on Linux (env override `SCRIBE_MD_BACKEND`). `transcriber.py` becomes a thin facade that validates input and delegates to the active backend; both backends return the same `{"segments": [...]}` dict so all downstream code is untouched. whisper.cpp is driven as a subprocess (built from a vendored submodule with a build-time accel flag), isolating GPU crashes from Python. macOS-only features (`live`, `--summarize`) emit clear messages on Linux instead of crashing.

**Tech Stack:** Python 3.12+, typer, pixi, whisper.cpp (C++ + GGML, Vulkan/CUDA), ffmpeg, mlx-whisper (macOS only), pytest.

---

## File Structure

**New:**
- `scribe_md/platform_support.py` — OS detection + platform-aware install hints (named to avoid shadowing stdlib `platform`).
- `scribe_md/backends/__init__.py` — `get_backend()` selection logic.
- `scribe_md/backends/base.py` — `Backend` protocol.
- `scribe_md/backends/mlx.py` — `MLXBackend` (moved from `transcriber.py`).
- `scribe_md/backends/whispercpp.py` — `WhisperCppBackend` + build/download/parse helpers.
- `tests/test_platform_support.py`
- `tests/test_backends.py`
- `tests/test_whispercpp.py`
- `tests/test_cli_degradation.py`
- `docs/LINUX.md` — on-hardware setup, GPU notes, benchmark checklist.

**Modified:**
- `scribe_md/transcriber.py` — becomes facade; keeps `MODEL_PRESETS`, `DEFAULT_MODEL`, `TranscriptionError`, validation, `extract_segments`.
- `scribe_md/audio.py` — platform-aware ffmpeg hint (4 sites).
- `scribe_md/cli.py` — degradation guards for `live` and `--summarize`.
- `pyproject.toml` — gate `mlx-whisper` to `sys_platform == 'darwin'`.
- `pixi.toml` — add `linux-64` platform, Linux build deps, `build-whisper` task, `cuda` feature/environment; move `build-capture` to the macOS target.
- `.gitmodules` / `vendor/whisper.cpp` — pinned submodule.
- `README.md` — Linux section pointer.

**Note on import cycles:** `transcriber.py` imports `from .backends import get_backend`. `backends/__init__.py` imports only `.base` at top level and imports `mlx`/`whispercpp` *lazily inside `get_backend()`*. `mlx.py` and `whispercpp.py` import `MODEL_PRESETS`/`TranscriptionError` from `transcriber`, which is safe because they are only imported after `transcriber` has finished loading. Keep it this way.

---

## Task 1: Platform detection module

**Files:**
- Create: `scribe_md/platform_support.py`
- Test: `tests/test_platform_support.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_platform_support.py
import scribe_md.platform_support as ps


def test_is_macos_true_on_darwin(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    assert ps.is_macos() is True
    assert ps.is_linux() is False


def test_is_linux_true_on_linux(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert ps.is_linux() is True
    assert ps.is_macos() is False


def test_ffmpeg_hint_is_apt_on_linux(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert "apt" in ps.ffmpeg_install_hint()


def test_ffmpeg_hint_is_brew_on_macos(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    assert "brew" in ps.ffmpeg_install_hint()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_platform_support.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scribe_md.platform_support'`

- [ ] **Step 3: Write minimal implementation**

```python
# scribe_md/platform_support.py
"""OS detection and platform-aware user hints.

Named ``platform_support`` rather than ``platform`` to avoid shadowing the
standard-library module.
"""

import sys


def is_macos() -> bool:
    """True on macOS."""
    return sys.platform == "darwin"


def is_linux() -> bool:
    """True on any Linux distribution."""
    return sys.platform.startswith("linux")


def ffmpeg_install_hint() -> str:
    """Return a platform-appropriate ffmpeg install instruction."""
    if is_linux():
        return "Install it with: sudo apt install ffmpeg"
    return "Install it with: brew install ffmpeg"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_platform_support.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scribe_md/platform_support.py tests/test_platform_support.py
git commit -m "Add platform_support module for OS detection"
```

---

## Task 2: Backend protocol and selection

**Files:**
- Create: `scribe_md/backends/__init__.py`, `scribe_md/backends/base.py`
- Test: `tests/test_backends.py`

This task creates the protocol and selection logic with placeholder backend classes that are replaced with real logic in Tasks 3 and 4–5. The placeholder classes let the selection test pass without pulling in `mlx_whisper`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends.py
import pytest

from scribe_md.backends import get_backend


def test_selects_mlx_on_macos(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_BACKEND", raising=False)
    monkeypatch.setattr("sys.platform", "darwin")
    assert type(get_backend()).__name__ == "MLXBackend"


def test_selects_whispercpp_on_linux(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_BACKEND", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    assert type(get_backend()).__name__ == "WhisperCppBackend"


def test_env_override_forces_backend(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("SCRIBE_MD_BACKEND", "whispercpp")
    assert type(get_backend()).__name__ == "WhisperCppBackend"


def test_unknown_override_raises(monkeypatch):
    monkeypatch.setenv("SCRIBE_MD_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_backend()


def test_unsupported_platform_raises(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_BACKEND", raising=False)
    monkeypatch.setattr("sys.platform", "win32")
    with pytest.raises(RuntimeError):
        get_backend()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scribe_md.backends'`

- [ ] **Step 3: Write the protocol**

```python
# scribe_md/backends/base.py
"""Backend protocol shared by all transcription engines."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """A transcription engine.

    Implementations must return the same result shape as mlx-whisper:
    ``{"segments": [{"start": float, "end": float, "text": str,
    "no_speech_prob": float}, ...]}``.
    """

    name: str

    def resolve_model(self, model: str) -> str:
        """Map a preset name (or pass-through path) to a backend-specific ref."""
        ...

    def transcribe(self, audio_path: Path, *, model: str, language: str | None) -> dict:
        """Transcribe a 16 kHz mono WAV, returning a result dict."""
        ...

    def describe(self) -> str:
        """Human-readable engine + device string for logging."""
        ...
```

- [ ] **Step 4: Write selection logic with placeholder backends**

```python
# scribe_md/backends/__init__.py
"""Transcription backend selection.

``get_backend()`` picks MLX on macOS and whisper.cpp on Linux. Backend modules
are imported lazily so that importing this package never pulls in
platform-specific dependencies (e.g. ``mlx_whisper``).
"""

import os
import sys

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

    if sys.platform == "darwin":
        from .mlx import MLXBackend
        return MLXBackend()
    if sys.platform.startswith("linux"):
        from .whispercpp import WhisperCppBackend
        return WhisperCppBackend()
    raise RuntimeError(
        f"scribe-md has no transcription backend for platform {sys.platform!r}."
    )
```

- [ ] **Step 5: Add temporary placeholder backend modules**

These are replaced in later tasks but let Task 2 pass independently.

```python
# scribe_md/backends/mlx.py
"""Placeholder — real implementation lands in Task 3."""


class MLXBackend:
    name = "mlx"
```

```python
# scribe_md/backends/whispercpp.py
"""Placeholder — real implementation lands in Tasks 4–5."""


class WhisperCppBackend:
    name = "whispercpp"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pixi run pytest tests/test_backends.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add scribe_md/backends/ tests/test_backends.py
git commit -m "Add backend protocol and platform-based selection"
```

---

## Task 3: Move MLX logic into MLXBackend; refactor transcriber into a facade

**Files:**
- Modify: `scribe_md/backends/mlx.py`
- Modify: `scribe_md/transcriber.py`
- Test: existing `tests/` must stay green; add `tests/test_backends.py::test_mlx_resolve_model`

- [ ] **Step 1: Write the failing test (append to tests/test_backends.py)**

```python
def test_mlx_resolve_model_maps_preset():
    from scribe_md.backends.mlx import MLXBackend
    b = MLXBackend()
    assert b.resolve_model("small") == "mlx-community/whisper-small-mlx"
    # unknown name passes through unchanged (treated as an HF path)
    assert b.resolve_model("some/custom-repo") == "some/custom-repo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_backends.py::test_mlx_resolve_model_maps_preset -v`
Expected: FAIL — `AttributeError: 'MLXBackend' object has no attribute 'resolve_model'`

- [ ] **Step 3: Implement the real MLXBackend**

```python
# scribe_md/backends/mlx.py
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
```

- [ ] **Step 4: Refactor transcriber.py into a facade**

Replace the body of `scribe_md/transcriber.py` with the following. `MODEL_PRESETS`, `DEFAULT_MODEL`, `TranscriptionError`, `resolve_model`, and `extract_segments` keep their names and signatures so `cli.py` and existing tests are unaffected.

```python
# scribe_md/transcriber.py
"""Transcription facade.

Validates input, then delegates to the platform-selected backend
(``scribe_md.backends``). ``MODEL_PRESETS``/``DEFAULT_MODEL`` remain here as the
canonical, user-facing preset vocabulary (the MLX backend reuses them; the
whisper.cpp backend maps the same names to GGML files).
"""

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
    """Resolve a preset name or full path to an MLX HF repo path.

    Retained for backward compatibility; per-backend resolution lives on each
    backend's ``resolve_model`` method.
    """
    return MODEL_PRESETS.get(model, model)


class TranscriptionError(RuntimeError):
    """Raised when transcription fails due to invalid input or runtime errors."""


def transcribe_audio(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
) -> dict:
    """Validate *audio_path* and transcribe it via the active backend."""
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    file_size = audio_path.stat().st_size
    if file_size == 0:
        raise TranscriptionError(
            f"Audio file is empty (0 bytes): {audio_path.name}. "
            "The recording may have failed or been interrupted."
        )
    if file_size < 44:
        raise TranscriptionError(
            f"Audio file is too small ({file_size} bytes): {audio_path.name}. "
            "The file may be corrupt."
        )

    from .backends import get_backend

    backend = get_backend()
    log(f"Transcribing {audio_path.name} via {backend.describe()}...")
    try:
        return backend.transcribe(audio_path, model=model, language=language)
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(
            f"Transcription failed for {audio_path.name}: {e}"
        ) from e


def extract_segments(
    result: dict,
    no_speech_threshold: float = 0.6,
) -> list[dict]:
    """Extract normalized segments from a backend result.

    Filters out segments with high no_speech_prob to prevent hallucination
    on silent audio.
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
```

- [ ] **Step 5: Run the full suite to verify nothing regressed**

Run: `pixi run pytest -v`
Expected: PASS — all previously passing tests still pass, plus the new `test_mlx_resolve_model_maps_preset`.

- [ ] **Step 6: Smoke-test the macOS path still transcribes (manual, macOS only)**

Run: `pixi run scribe-md file tests/fixtures/*.wav` if a fixture exists, otherwise any short WAV.
Expected: produces a Markdown transcription as before. (Skip if no audio fixture is available; the unit suite covers the wiring.)

- [ ] **Step 7: Commit**

```bash
git add scribe_md/backends/mlx.py scribe_md/transcriber.py tests/test_backends.py
git commit -m "Move MLX logic into MLXBackend; make transcriber a facade"
```

---

## Task 4: whisper.cpp pure helpers (JSON parsing, model + accel resolution)

**Files:**
- Modify: `scribe_md/backends/whispercpp.py` (add pure, GPU-free helpers)
- Test: `tests/test_whispercpp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_whispercpp.py
import pytest

from scribe_md.backends import whispercpp as w


SAMPLE_JSON = {
    "transcription": [
        {"offsets": {"from": 0, "to": 2000}, "text": " Hello world"},
        {"offsets": {"from": 2000, "to": 4500}, "text": " second segment "},
    ]
}


def test_parse_json_converts_ms_to_seconds_and_strips_text():
    result = w.parse_whispercpp_json(SAMPLE_JSON)
    segs = result["segments"]
    assert segs[0] == {"start": 0.0, "end": 2.0, "text": "Hello world", "no_speech_prob": 0.0}
    assert segs[1] == {"start": 2.0, "end": 4.5, "text": "second segment", "no_speech_prob": 0.0}


def test_parse_json_empty_transcription():
    assert w.parse_whispercpp_json({"transcription": []}) == {"segments": []}


def test_resolve_model_maps_presets_to_ggml():
    assert w.resolve_model_filename("small") == "ggml-small.bin"
    assert w.resolve_model_filename("large-v3") == "ggml-large-v3.bin"
    assert w.resolve_model_filename("large") == "ggml-large-v3.bin"


def test_resolve_model_passthrough_for_unknown():
    assert w.resolve_model_filename("ggml-custom.bin") == "ggml-custom.bin"


def test_accel_cmake_flags():
    assert w.accel_cmake_flags("vulkan") == ["-DGGML_VULKAN=1"]
    assert w.accel_cmake_flags("cuda") == ["-DGGML_CUDA=1"]
    assert w.accel_cmake_flags("cpu") == []


def test_detect_accel_env_override(monkeypatch):
    monkeypatch.setenv("SCRIBE_MD_WHISPER_ACCEL", "cpu")
    assert w.detect_accel() == "cpu"


def test_detect_accel_prefers_cuda_when_toolkit_present(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/" + name
                        if name in ("nvcc", "nvidia-smi") else None)
    assert w.detect_accel() == "cuda"


def test_detect_accel_falls_back_to_vulkan(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/vulkaninfo"
                        if name == "vulkaninfo" else None)
    assert w.detect_accel() == "vulkan"


def test_detect_accel_cpu_when_nothing_present(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: None)
    assert w.detect_accel() == "cpu"


def test_build_command_shape(tmp_path):
    cmd = w._build_command("/bin/whisper-cli", tmp_path / "m.bin",
                           tmp_path / "in.wav", tmp_path / "out", "en")
    assert cmd[0] == "/bin/whisper-cli"
    assert "-oj" in cmd and "--no-prints" in cmd
    assert cmd[cmd.index("-l") + 1] == "en"


def test_build_command_auto_language(tmp_path):
    cmd = w._build_command("/bin/whisper-cli", tmp_path / "m.bin",
                           tmp_path / "in.wav", tmp_path / "out", None)
    assert cmd[cmd.index("-l") + 1] == "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_whispercpp.py -v`
Expected: FAIL — `AttributeError: module 'scribe_md.backends.whispercpp' has no attribute 'parse_whispercpp_json'`

- [ ] **Step 3: Implement the pure helpers (replace placeholder module)**

```python
# scribe_md/backends/whispercpp.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_whispercpp.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add scribe_md/backends/whispercpp.py tests/test_whispercpp.py
git commit -m "Add whisper.cpp pure helpers (JSON parse, model + accel resolution)"
```

---

## Task 5: whisper.cpp build, model download, and transcribe

**Files:**
- Modify: `scribe_md/backends/whispercpp.py` (append build/download/backend class)
- Test: `tests/test_whispercpp.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_whispercpp.py)**

```python
def test_ensure_binary_returns_existing(monkeypatch, tmp_path):
    fake_bin = tmp_path / "whisper-cli"
    fake_bin.write_text("")
    monkeypatch.setattr(w, "WHISPER_BIN", fake_bin)
    assert w.ensure_whisper_binary() == fake_bin


def test_ensure_binary_errors_when_source_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "WHISPER_BIN", tmp_path / "absent" / "whisper-cli")
    monkeypatch.setattr(w, "VENDOR_DIR", tmp_path / "empty-vendor")
    with pytest.raises(w.WhisperCppError) as excinfo:
        w.ensure_whisper_binary()
    assert "submodule" in str(excinfo.value)


def test_ensure_model_uses_existing_file(monkeypatch, tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "ggml-small.bin").write_text("weights")
    monkeypatch.setattr(w, "MODEL_CACHE", cache)
    called = {"download": False}
    monkeypatch.setattr(w.urllib.request, "urlretrieve",
                        lambda *a, **k: called.__setitem__("download", True))
    path = w._ensure_model_file("small")
    assert path == cache / "ggml-small.bin"
    assert called["download"] is False


def test_transcribe_runs_cli_and_parses(monkeypatch, tmp_path):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"\x00" * 100)
    monkeypatch.setattr(w, "ensure_whisper_binary", lambda: Path("/bin/whisper-cli"))
    monkeypatch.setattr(w, "_ensure_model_file", lambda m: tmp_path / "m.bin")

    def fake_run(cmd, **kwargs):
        # whisper-cli writes <out_prefix>.json; recover the prefix from -of.
        out_prefix = Path(cmd[cmd.index("-of") + 1])
        out_prefix.with_suffix(".json").write_text(
            '{"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": " hi"}]}'
        )
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(w.subprocess, "run", fake_run)
    result = w.WhisperCppBackend().transcribe(wav, model="small", language="en")
    assert result["segments"][0]["text"] == "hi"
    assert result["segments"][0]["end"] == 1.0


def test_transcribe_raises_on_nonzero_exit(monkeypatch, tmp_path):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"\x00" * 100)
    monkeypatch.setattr(w, "ensure_whisper_binary", lambda: Path("/bin/whisper-cli"))
    monkeypatch.setattr(w, "_ensure_model_file", lambda m: tmp_path / "m.bin")

    def fake_run(cmd, **kwargs):
        class R:
            returncode = 1
            stderr = "vulkan device lost"
        return R()

    monkeypatch.setattr(w.subprocess, "run", fake_run)
    with pytest.raises(w.WhisperCppError):
        w.WhisperCppBackend().transcribe(wav, model="small", language=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_whispercpp.py -k "ensure or transcribe" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'ensure_whisper_binary'`

- [ ] **Step 3: Append build/download/backend implementation**

```python
# --- Build, download, transcribe (append to whispercpp.py) ------------------
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
        raise WhisperCppError(f"Failed to download model {fname} from {url}: {e}")
    tmp.rename(dest)
    return dest


class WhisperCppBackend:
    """Transcription via the whisper.cpp ``whisper-cli`` subprocess."""

    name = "whispercpp"

    def resolve_model(self, model: str) -> str:
        return resolve_model_filename(model)

    def describe(self) -> str:
        return f"whisper.cpp ({detect_accel()})"

    def transcribe(self, audio_path: Path, *, model: str, language: str | None) -> dict:
        import json

        binary = ensure_whisper_binary()
        model_path = _ensure_model_file(model)
        with tempfile.TemporaryDirectory() as td:
            out_prefix = Path(td) / "out"
            cmd = _build_command(binary, model_path, audio_path, out_prefix, language)
            result = subprocess.run(cmd, capture_output=True, text=True)
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
    """``python -m scribe_md.backends.whispercpp build`` pre-builds the binary."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        path = ensure_whisper_binary()
        print(f"Built whisper.cpp: {path}")
    else:
        print("Usage: python -m scribe_md.backends.whispercpp build")


if __name__ == "__main__":
    _main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_whispercpp.py -v`
Expected: PASS (all whispercpp tests pass)

- [ ] **Step 5: Run full suite**

Run: `pixi run pytest -v`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add scribe_md/backends/whispercpp.py tests/test_whispercpp.py
git commit -m "Add whisper.cpp build, model download, and transcribe"
```

---

## Task 6: CLI degradation for macOS-only features on Linux

**Files:**
- Modify: `scribe_md/cli.py` (add guard at top of `live`; add guard in `_apply_postprocessing` summarize branch)
- Test: `tests/test_cli_degradation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_degradation.py
import pytest
import typer
from typer.testing import CliRunner

from scribe_md.cli import app, _apply_postprocessing

runner = CliRunner()


def test_live_is_macos_only_message_on_linux(monkeypatch):
    monkeypatch.setattr("scribe_md.cli.platform_support.is_linux", lambda: True)
    result = runner.invoke(app, ["live"])
    assert result.exit_code == 1
    assert "macOS-only" in result.output


def test_summarize_blocked_on_linux(monkeypatch):
    monkeypatch.setattr("scribe_md.cli.platform_support.is_linux", lambda: True)
    with pytest.raises(typer.Exit):
        _apply_postprocessing("some transcript text", summarize=True)


def test_summarize_allowed_on_macos(monkeypatch):
    # On macOS the guard must not fire; stub the LLM call to avoid loading mlx-lm.
    monkeypatch.setattr("scribe_md.cli.platform_support.is_linux", lambda: False)
    monkeypatch.setattr(
        "scribe_md.cli.postprocess.summarize_with_llm",
        lambda text, model=None: "a summary",
    )
    out = _apply_postprocessing("transcript", summarize=True)
    assert "## Summary" in out
    assert "a summary" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_cli_degradation.py -v`
Expected: FAIL — `AttributeError: module 'scribe_md.cli' has no attribute 'platform_support'`

- [ ] **Step 3: Add the import**

In `scribe_md/cli.py`, add to the imports near line 12:

```python
from . import platform_support
```

- [ ] **Step 4: Guard the `live` command**

In `scribe_md/cli.py`, the `live` function body begins (line ~850) with:

```python
    """Capture and transcribe system audio in real-time."""
    cfg = load_config()
```

Insert the guard immediately after the docstring, before `cfg = load_config()`:

```python
    """Capture and transcribe system audio in real-time."""
    if platform_support.is_linux():
        console.print(
            "[red]Error:[/red] Live system-audio capture is macOS-only for now. "
            "Use 'scribe-md file' or 'scribe-md url' on Linux."
        )
        raise typer.Exit(1)
    cfg = load_config()
```

- [ ] **Step 5: Guard the summarize branch**

In `scribe_md/cli.py`, `_apply_postprocessing` (line ~217) has:

```python
    if summarize:
        try:
            model = summary_model or None
```

Insert the platform guard at the start of the `if summarize:` block:

```python
    if summarize:
        if platform_support.is_linux():
            console.print(
                "[red]Error:[/red] Summarization (mlx-lm) is macOS-only for now."
            )
            raise typer.Exit(1)
        try:
            model = summary_model or None
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pixi run pytest tests/test_cli_degradation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add scribe_md/cli.py tests/test_cli_degradation.py
git commit -m "Add Linux degradation messages for live and --summarize"
```

---

## Task 7: Platform-aware ffmpeg install hint

**Files:**
- Modify: `scribe_md/audio.py` (4 occurrences of the brew hint)
- Test: `tests/test_audio_hint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_hint.py
import pytest

from scribe_md import audio
from scribe_md.audio import AudioConversionError


def test_convert_missing_ffmpeg_uses_platform_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "linux")

    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(audio.subprocess, "run", boom)
    monkeypatch.setattr(audio, "_check_disk_space", lambda *a, **k: None)
    with pytest.raises(AudioConversionError) as excinfo:
        audio.convert_to_16k_mono(tmp_path / "in.mp3", tmp_path / "out.wav")
    assert "apt" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_audio_hint.py -v`
Expected: FAIL — message contains "brew", not "apt".

- [ ] **Step 3: Replace the hardcoded hints**

In `scribe_md/audio.py`, add the import near the top (after `from .utils import log`):

```python
from .platform_support import ffmpeg_install_hint
```

Then replace each of the four occurrences of:

```python
            "ffmpeg not found. Install it with: brew install ffmpeg"
```

with:

```python
            f"ffmpeg not found. {ffmpeg_install_hint()}"
```

and the one ffprobe occurrence:

```python
            "ffprobe not found. Install it with: brew install ffmpeg"
```

with:

```python
            f"ffprobe not found. {ffmpeg_install_hint()}"
```

(Use editor find-all for `Install it with: brew install ffmpeg` — there are several.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_audio_hint.py -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `pixi run pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scribe_md/audio.py tests/test_audio_hint.py
git commit -m "Make ffmpeg install hint platform-aware"
```

---

## Task 8: Packaging — pyproject markers, pixi platforms, submodule

**Files:**
- Modify: `pyproject.toml`
- Modify: `pixi.toml`
- Create: `.gitmodules` + `vendor/whisper.cpp` (submodule)

This task has no unit tests; verification is via `pixi` resolution and the build task.

- [ ] **Step 1: Gate MLX in pyproject.toml**

Replace the `dependencies` list in `pyproject.toml` with:

```toml
dependencies = [
    "typer>=0.9",
    "yt-dlp",
    "rich",
    "mlx-whisper; sys_platform == 'darwin'",
]
```

(`mlx-lm` is install-on-demand for `--summarize` and already absent from
`dependencies`, so no change is needed there.)

- [ ] **Step 2: Add the whisper.cpp submodule**

```bash
git submodule add https://github.com/ggml-org/whisper.cpp vendor/whisper.cpp
cd vendor/whisper.cpp && git checkout "$(git describe --tags --abbrev=0)" && cd ../..
git add .gitmodules vendor/whisper.cpp
```

This pins the submodule to the latest release tag; the exact commit is recorded
in the gitlink. If `ggml-org/whisper.cpp` is unavailable, use the mirror
`https://github.com/ggerganov/whisper.cpp`.

- [ ] **Step 3: Rewrite pixi.toml for two platforms**

Replace `pixi.toml` with:

```toml
[workspace]
authors = ["HALCIVXIVC <25841608+hoohugokim@users.noreply.github.com>"]
channels = ["conda-forge"]
name = "scribe-md"
platforms = ["osx-arm64", "linux-64"]
version = "0.1.1"

[tasks]
sm = { cmd = "scribe-md", description = "Run scribe-md CLI" }

[dependencies]
python = ">=3.12,<3.14"
ffmpeg = ">=7"

[pypi-dependencies]
scribe-md = { path = ".", editable = true }
pytest = ">=9.0.2, <10"

# --- macOS: Swift capture binary -------------------------------------------
[target.osx-arm64.tasks]
build-capture = { cmd = "cd capture && swift build -c release", description = "Build the Swift audio capture CLI" }

# --- Linux: whisper.cpp build toolchain (Vulkan by default) ----------------
[target.linux-64.dependencies]
cmake = ">=3.20"
cxx-compiler = "*"
shaderc = "*"
vulkan-headers = "*"
vulkan-loader = "*"

[target.linux-64.tasks]
build-whisper = { cmd = "git submodule update --init vendor/whisper.cpp && python -m scribe_md.backends.whispercpp build", description = "Build whisper.cpp with auto-detected GPU backend (Vulkan/CUDA/CPU)" }

# --- Optional CUDA toolchain (NVIDIA peak performance) ---------------------
[feature.cuda.target.linux-64.dependencies]
cuda-toolkit = ">=12"

[environments]
default = { features = [] }
cuda = { features = ["cuda"] }
```

The runtime Vulkan driver is the system Mesa RADV ICD (preinstalled on Pop!_OS);
it is intentionally not a pixi dependency.

- [ ] **Step 4: Verify resolution on macOS**

Run: `pixi install`
Expected: resolves and installs without error (osx-arm64 environment, MLX present).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pixi.toml pixi.lock .gitmodules vendor/whisper.cpp
git commit -m "Add linux-64 platform, whisper.cpp submodule, and CUDA feature"
```

---

## Task 9: Linux documentation and on-hardware checklist

**Files:**
- Create: `docs/LINUX.md`
- Modify: `README.md` (add a short Linux pointer)

No tests. Content task.

- [ ] **Step 1: Write docs/LINUX.md**

```markdown
# Running scribe-md on Linux

Supported on Pop!_OS 24.04 / Ubuntu 24.04 for the `file` and `url` commands.
`live` capture and `--summarize` are macOS-only for now.

## Install

```bash
git clone <repo-url> && cd scribe-md
git submodule update --init vendor/whisper.cpp
pixi install
pixi run build-whisper      # builds whisper.cpp with auto-detected GPU backend
pixi run scribe-md file recording.wav
```

`build-whisper` auto-detects the accelerator: **Vulkan** if available (the
default for both AMD and NVIDIA), **CUDA** if the toolkit is present, else
**CPU**. Force a choice with `SCRIBE_MD_WHISPER_ACCEL=vulkan|cuda|cpu`.

## GPU notes

- **AMD (e.g. RX 5700 XT, RDNA1):** Vulkan via Mesa RADV is the reliable path.
  ROCm is not officially supported on RDNA1 and is not used here.
- **NVIDIA:** Vulkan works out-of-the-box. For peak performance, install the
  CUDA toolkit via the opt-in environment: `pixi install -e cuda` then
  `SCRIBE_MD_WHISPER_ACCEL=cuda pixi run build-whisper`.
- Confirm the device in use: scribe-md logs `Transcribing ... via whisper.cpp (vulkan)`.

## Benchmark (Vulkan vs CPU)

Time a fixed clip on each accelerator to get real numbers on your hardware:

```bash
rm -rf vendor/whisper.cpp/build
SCRIBE_MD_WHISPER_ACCEL=vulkan pixi run build-whisper
time SCRIBE_MD_WHISPER_ACCEL=vulkan pixi run scribe-md file sample.wav -m small

rm -rf vendor/whisper.cpp/build
SCRIBE_MD_WHISPER_ACCEL=cpu pixi run build-whisper
time SCRIBE_MD_WHISPER_ACCEL=cpu pixi run scribe-md file sample.wav -m small
```

## Known limitations on Linux

- `scribe-md live` → "Live system-audio capture is macOS-only for now."
- `--summarize` → "Summarization (mlx-lm) is macOS-only for now."
- `--diarize` (pyannote, CPU) is expected to work but is not verified on Linux.
```

- [ ] **Step 2: Add a Linux pointer to README.md**

Under the Requirements section in `README.md`, add:

```markdown
> **Linux (Pop!_OS / Ubuntu 24.04):** `file` and `url` commands are supported
> via whisper.cpp with Vulkan/CUDA GPU acceleration. See [docs/LINUX.md](docs/LINUX.md).
```

- [ ] **Step 3: Commit**

```bash
git add docs/LINUX.md README.md
git commit -m "Document Linux setup, GPU notes, and benchmark checklist"
```

---

## On-hardware verification (run on the RX 5700 XT machine)

These are not automated; perform after pulling the `linux-support` branch on Pop!_OS:

- [ ] `pixi install` succeeds (no MLX resolution error).
- [ ] `pixi run build-whisper` reports `accel=vulkan` and builds `whisper-cli`.
- [ ] `pixi run scribe-md file sample.wav` logs `via whisper.cpp (vulkan)` and produces Markdown.
- [ ] `pixi run scribe-md url <youtube-url>` produces Markdown.
- [ ] `pixi run scribe-md live` prints the macOS-only message and exits 1.
- [ ] Benchmark Vulkan vs CPU per `docs/LINUX.md`; record the speedup.

---

## Self-Review Notes

- **Spec coverage:** §1 backend abstraction → Tasks 2–3; §2 whisper.cpp Vulkan/CUDA → Tasks 4–5, 8; §3 packaging → Task 8; §4 graceful degradation → Tasks 6–7; §5 testing & delivery → unit tests across Tasks 1–7, on-hardware checklist + benchmark in Task 9 / final section. All sections mapped.
- **Type consistency:** `Backend.transcribe(audio_path, *, model, language)`, `.resolve_model(model)`, `.describe()` are used identically in `MLXBackend`, `WhisperCppBackend`, and the `transcribe_audio` facade. `parse_whispercpp_json` returns the same `{"segments":[{start,end,text,no_speech_prob}]}` shape `extract_segments` consumes.
- **Placeholder scan:** Task 2 introduces deliberately-temporary placeholder backend classes, each replaced in a later task (3 and 4–5) — these are intentional scaffolding, not unfilled plan placeholders.
```
