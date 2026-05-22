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
