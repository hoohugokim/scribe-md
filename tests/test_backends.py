"""Tests for transcription backend selection."""

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


def test_mlx_resolve_model_maps_preset():
    from scribe_md.backends.mlx import MLXBackend
    b = MLXBackend()
    assert b.resolve_model("small") == "mlx-community/whisper-small-mlx"
    assert b.resolve_model("some/custom-repo") == "some/custom-repo"


# ---------------------------------------------------------------------------
# Task 2: device parameter – protocol conformance and MLX accept-and-ignore
# ---------------------------------------------------------------------------


def test_mlx_backend_accepts_device_and_ignores_it(monkeypatch, tmp_path):
    """MLX backend must accept device= without error and ignore it (single unified-memory device)."""
    from scribe_md.backends.mlx import MLXBackend

    captured = {}

    def fake_transcribe(audio_path, **kwargs):
        captured["kwargs"] = kwargs
        return {"segments": []}

    import sys
    import types
    # Provide a minimal mlx_whisper stub so the deferred import succeeds on Linux.
    mlx_stub = types.ModuleType("mlx_whisper")
    mlx_stub.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", mlx_stub)

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00" * 100)
    result = MLXBackend().transcribe(wav, model="tiny", language="ko", device="0")

    # device must not be forwarded to mlx_whisper.transcribe
    assert "device" not in captured["kwargs"]
    assert result == {"segments": []}


def test_backend_protocol_transcribe_accepts_device_kwarg(monkeypatch):
    """Both concrete backends expose device: str | None = None in their transcribe signature."""
    import inspect
    from scribe_md.backends.mlx import MLXBackend
    from scribe_md.backends.whispercpp import WhisperCppBackend

    for cls in (MLXBackend, WhisperCppBackend):
        sig = inspect.signature(cls.transcribe)
        assert "device" in sig.parameters, (
            f"{cls.__name__}.transcribe must accept a 'device' keyword argument"
        )
        param = sig.parameters["device"]
        assert param.default is None, (
            f"{cls.__name__}.transcribe 'device' must default to None"
        )
