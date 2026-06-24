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


def test_parallel_prepare_failure_skips_source_and_releases_inflight(monkeypatch):
    """prepare() raising must skip the source and release the inflight semaphore."""
    monkeypatch.setattr(scheduler, "transcribe_chunk", lambda *a, **k: [])

    call_count = 0

    def bad_prepare(source):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad source")

    summary = scheduler.transcribe_in_parallel(
        ["X", "Y"], gpu_ids=[0], model="tiny", language=None,
        prepare=bad_prepare, finalize=lambda p, o: None, max_inflight=2,
    )
    assert call_count == 2
    assert len(summary.skipped) == 2
    assert all("prepare failed" in reason for _, reason in summary.skipped)
    assert summary.all_failed


def test_parallel_no_audio_chunks_skips_source_and_calls_cleanup(monkeypatch):
    """An empty chunk_paths list must skip the source, call cleanup, and release inflight."""
    monkeypatch.setattr(scheduler, "transcribe_chunk", lambda *a, **k: [])

    cleanup_called = []

    def prepare_empty(source):
        def cleanup():
            cleanup_called.append(source)
        return PreparedSource(
            key=str(source),
            chunk_paths=[],
            cleanup=cleanup,
        )

    summary = scheduler.transcribe_in_parallel(
        ["empty"], gpu_ids=[0], model="tiny", language=None,
        prepare=prepare_empty, finalize=lambda p, o: None, max_inflight=2,
    )
    assert cleanup_called == ["empty"], "cleanup must be called for no-chunks sources"
    assert summary.skipped == [("empty", "no audio chunks")]
    assert summary.all_failed


def test_parallel_cleanup_called_on_fully_failed_source(monkeypatch):
    """cleanup() must be called even when all chunks fail."""
    def boom(path, model, language, device=None):
        raise RuntimeError("chunk error")

    monkeypatch.setattr(scheduler, "transcribe_chunk", boom)

    cleanup_called = []

    def prepare_with_cleanup(source):
        def cleanup():
            cleanup_called.append(source)
        return PreparedSource(
            key=str(source),
            chunk_paths=[Path(f"{source}-0.wav")],
            cleanup=cleanup,
        )

    summary = scheduler.transcribe_in_parallel(
        ["Z"], gpu_ids=[0], model="tiny", language=None,
        prepare=prepare_with_cleanup, finalize=lambda p, o: None, max_inflight=2,
    )
    assert cleanup_called == ["Z"], "cleanup must be called even when all chunks fail"
    assert summary.skipped
    assert summary.all_failed


def test_parallel_finalize_raises_does_not_deadlock(monkeypatch):
    """If finalize() raises, the finalizer thread must not crash (deadlock prevention)."""
    monkeypatch.setattr(scheduler, "transcribe_chunk",
                        lambda path, model, language, device=None: [])

    cleanup_called = []

    def prepare_with_cleanup(source):
        def cleanup():
            cleanup_called.append(source)
        return PreparedSource(
            key=str(source),
            chunk_paths=[Path(f"{source}-0.wav")],
            cleanup=cleanup,
        )

    def bad_finalize(prepared, ordered):
        raise RuntimeError("finalize boom")

    summary = scheduler.transcribe_in_parallel(
        ["F1", "F2"], gpu_ids=[0], model="tiny", language=None,
        prepare=prepare_with_cleanup, finalize=bad_finalize, max_inflight=2,
    )
    # Both sources should be skipped (finalize failed), not hung
    assert len(summary.skipped) == 2
    assert all("finalize failed" in reason for _, reason in summary.skipped)
    # cleanup must still have been called for both
    assert sorted(cleanup_called) == ["F1", "F2"]


def test_parallel_chunk_baseexception_does_not_deadlock(monkeypatch):
    """If transcribe_chunk raises a BaseException (e.g. KeyboardInterrupt),
    the scheduler must not deadlock — the job must still complete and the
    source must be marked skipped rather than hanging forever."""
    monkeypatch.setattr(scheduler, "transcribe_chunk",
                        lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))

    result_holder = {}

    def run():
        summary = scheduler.transcribe_in_parallel(
            ["A"], gpu_ids=[0], model="t", language=None,
            prepare=_prep(1),
            finalize=lambda p, o: None,
            max_inflight=1,
        )
        result_holder["summary"] = summary

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "transcribe_in_parallel deadlocked on BaseException"
    # The source must have been skipped (chunk raised), not succeeded
    summary = result_holder.get("summary")
    assert summary is not None
    assert summary.succeeded == []
    assert len(summary.skipped) == 1
