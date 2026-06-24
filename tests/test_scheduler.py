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


# tests/test_scheduler.py  (add)
import threading
from pathlib import Path
from scribe_md import scheduler
from scribe_md.scheduler import PreparedSource


def _prep(n_chunks):
    def prepare(source):
        return PreparedSource(
            key=str(source),
            chunk_paths=[Path(f"{source}-{i}.wav") for i in range(n_chunks)],
            cleanup=lambda: None,
        )
    return prepare


def test_parallel_uses_all_gpus_and_preserves_order(monkeypatch):
    seen_devices = set()
    lock = threading.Lock()

    def fake_chunk(path, model, language, device=None):
        with lock:
            seen_devices.add(device)
        idx = int(path.stem.split("-")[-1])
        return [{"start": float(idx), "end": idx + 1.0, "text": f"seg{idx}"}]

    monkeypatch.setattr(scheduler, "transcribe_chunk", fake_chunk)

    written = {}

    def finalize(prepared, ordered):
        written[prepared.key] = [segs[0]["text"] for segs in ordered]

    summary = scheduler.transcribe_in_parallel(
        ["A"], gpu_ids=[0, 1], model="tiny", language="ko",
        prepare=_prep(4), finalize=finalize, max_inflight=2,
    )

    assert seen_devices == {"0", "1"}
    assert written["A"] == ["seg0", "seg1", "seg2", "seg3"]  # in order
    assert summary.succeeded == ["A"]


def test_parallel_skips_a_fully_failed_source_but_continues(monkeypatch):
    def fake_chunk(path, model, language, device=None):
        if path.stem.startswith("BAD"):
            raise RuntimeError("boom")
        return [{"start": 0.0, "end": 1.0, "text": "ok"}]

    monkeypatch.setattr(scheduler, "transcribe_chunk", fake_chunk)
    written = []
    summary = scheduler.transcribe_in_parallel(
        ["GOOD", "BAD"], gpu_ids=[0], model="tiny", language=None,
        prepare=_prep(2), finalize=lambda p, o: written.append(p.key),
        max_inflight=2,
    )
    assert written == ["GOOD"]
    assert [k for k, _ in summary.skipped] == ["BAD"]
    assert not summary.all_failed


def test_parallel_all_sources_failed_sets_all_failed(monkeypatch):
    def boom(path, model, language, device=None):
        raise RuntimeError("x")

    monkeypatch.setattr(scheduler, "transcribe_chunk", boom)
    summary = scheduler.transcribe_in_parallel(
        ["A", "B"], gpu_ids=[0], model="tiny", language=None,
        prepare=_prep(1), finalize=lambda p, o: None, max_inflight=2,
    )
    assert summary.succeeded == []
    assert summary.all_failed
