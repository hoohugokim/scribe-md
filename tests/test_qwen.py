"""Tests for the Qwen3-ASR MLX backend."""

import sys
import types
import wave
from pathlib import Path

import pytest

from scribe_md.backends.qwen import QwenBackend, _result_from_mlx_audio
from scribe_md.transcriber import TranscriptionError


def _write_silence_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)


def test_resolve_model_presets():
    b = QwenBackend()
    assert "1.7B" in b.resolve_model("qwen")
    assert "0.6B" in b.resolve_model("qwen-0.6b")
    assert b.resolve_model("mlx-community/custom") == "mlx-community/custom"
    assert "1.7B" in b.resolve_model("large-v3")


def test_result_from_text_object():
    class R:
        text = "  hi there  "
        segments = None

    out = _result_from_mlx_audio(R(), duration=3.0)
    assert out["segments"][0]["text"] == "hi there"
    assert out["segments"][0]["end"] == 3.0


def test_result_from_segments_dict():
    out = _result_from_mlx_audio(
        {"segments": [{"start": 0, "end": 1, "text": "a"}]},
        duration=9,
    )
    assert out["segments"][0]["text"] == "a"


def test_transcribe_requires_macos(monkeypatch, tmp_path):
    monkeypatch.setattr("scribe_md.backends.qwen.is_macos", lambda: False)
    wav = tmp_path / "a.wav"
    _write_silence_wav(wav)
    with pytest.raises(TranscriptionError, match="macOS"):
        QwenBackend().transcribe(wav, model="qwen", language="ko")


def test_transcribe_with_stubbed_mlx_audio(monkeypatch, tmp_path):
    monkeypatch.setattr("scribe_md.backends.qwen.is_macos", lambda: True)

    class FakeModel:
        pass

    def fake_load(repo):
        assert "Qwen3" in repo
        return FakeModel()

    def fake_generate(*, model, audio_path, language=None):
        assert language == "ko"
        return {"text": "안녕하세요"}

    utils = types.ModuleType("mlx_audio.stt.utils")
    utils.load_model = fake_load
    gen = types.ModuleType("mlx_audio.stt.generate")
    gen.generate_transcription = fake_generate
    stt = types.ModuleType("mlx_audio.stt")
    stt.utils = utils
    stt.generate = gen
    root = types.ModuleType("mlx_audio")
    root.stt = stt

    monkeypatch.setitem(sys.modules, "mlx_audio", root)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.generate", gen)

    wav = tmp_path / "a.wav"
    _write_silence_wav(wav, seconds=1.5)
    result = QwenBackend().transcribe(wav, model="qwen", language="ko")
    assert result["segments"][0]["text"] == "안녕하세요"
    assert result["segments"][0]["end"] == pytest.approx(1.5)


def test_missing_mlx_audio_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("scribe_md.backends.qwen.is_macos", lambda: True)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("mlx_audio"):
            raise ImportError("no mlx_audio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    wav = tmp_path / "a.wav"
    _write_silence_wav(wav)
    with pytest.raises(TranscriptionError, match="mlx-audio"):
        QwenBackend().transcribe(wav, model="qwen", language=None)
