"""Tests for the Nemotron OpenAI-compatible backend."""

import json
import wave
from pathlib import Path

import pytest

from scribe_md.backends import nemotron as n
from scribe_md.transcriber import TranscriptionError


def _write_silence_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)


class _FakeResp:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_health_ok_true(monkeypatch):
    monkeypatch.setattr(
        n.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(b'{"data":[]}'),
    )
    assert n.health_ok("http://127.0.0.1:8000/v1") is True


def test_health_ok_false_on_error(monkeypatch):
    def boom(*a, **k):
        raise n.urllib.error.URLError("down")

    monkeypatch.setattr(n.urllib.request, "urlopen", boom)
    assert n.health_ok("http://127.0.0.1:8000/v1") is False


def test_transcribe_text_only_response(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    _write_silence_wav(wav, seconds=2.0)

    payload = {
        "choices": [
            {"message": {"content": "Hello from Nemotron."}},
        ]
    }

    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/chat/completions")
        return _FakeResp(json.dumps(payload).encode())

    monkeypatch.setattr(n.urllib.request, "urlopen", fake_urlopen)
    result = n.NemotronBackend().transcribe(wav, model="ignored", language="en")
    assert result["segments"][0]["text"] == "Hello from Nemotron."
    assert result["segments"][0]["end"] == pytest.approx(2.0)


def test_transcribe_timed_segments(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    _write_silence_wav(wav)

    payload = {
        "choices": [{"message": {"content": "ignored when segments present"}}],
        "segments": [
            {"start": 0.0, "end": 0.5, "text": "Hello"},
            {"start": 0.5, "end": 1.0, "text": "world"},
        ],
    }
    monkeypatch.setattr(
        n.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(json.dumps(payload).encode()),
    )
    result = n.NemotronBackend().transcribe(wav, model="x", language=None)
    assert [s["text"] for s in result["segments"]] == ["Hello", "world"]


def test_transcribe_raises_when_server_down(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    _write_silence_wav(wav)

    def boom(*a, **k):
        raise n.urllib.error.URLError("connection refused")

    monkeypatch.setattr(n.urllib.request, "urlopen", boom)
    with pytest.raises(TranscriptionError, match="unreachable"):
        n.NemotronBackend().transcribe(wav, model="x", language=None)


def test_describe():
    assert "Nemotron" in n.NemotronBackend().describe()
