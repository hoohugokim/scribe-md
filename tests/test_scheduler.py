# tests/test_scheduler.py
from pathlib import Path
from scribe_md import scheduler


def test_transcribe_chunk_returns_empty_for_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler.audio, "is_silent", lambda p: True)
    assert scheduler.transcribe_chunk(tmp_path / "c.wav", "tiny", "ko") == []


def test_transcribe_chunk_passes_device_through(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(scheduler.audio, "is_silent", lambda p: False)

    def fake_transcribe(path, *, model, language, device=None):
        seen["device"] = device
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "no_speech_prob": 0.0}]}

    monkeypatch.setattr(scheduler.transcriber, "transcribe_audio", fake_transcribe)
    out = scheduler.transcribe_chunk(tmp_path / "c.wav", "tiny", "ko", device="2")
    assert seen["device"] == "2"
    assert out == [{"start": 0.0, "end": 1.0, "text": "hi"}]
