"""Live chunked merge keeps capture indexes when a chunk is skipped."""

import io

from scribe_md.cli import _align_indexed_chunks, _live_chunked


def test_align_indexed_chunks_keeps_a_hole():
    aligned = _align_indexed_chunks([
        (0, [{"start": 0.0, "end": 1.0, "text": "early"}]),
        (2, [{"start": 0.0, "end": 1.0, "text": "later"}]),
    ])
    assert aligned[1] == []
    assert aligned[2][0]["text"] == "later"


class _Proc:
    def __init__(self, lines: list[bytes]):
        self.stdout = io.BytesIO(b"".join(line + b"\n" for line in lines))
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0

    def send_signal(self, _sig):
        return None


def test_live_chunked_silent_middle_keeps_later_timestamp(monkeypatch, tmp_path):
    """A silent middle chunk must not pull the next chunk's timestamps forward."""
    raws = []
    for i in range(3):
        path = tmp_path / f"raw_{i}.wav"
        path.write_bytes(b"x")
        raws.append(path)

    monkeypatch.setattr(
        "scribe_md.cli.capture.run_capture",
        lambda *a, **k: _Proc([str(p).encode() for p in raws]),
    )
    monkeypatch.setattr(
        "scribe_md.cli.audio.convert_to_16k_mono",
        lambda src, dst: dst.write_bytes(b"x"),
    )
    monkeypatch.setattr(
        "scribe_md.cli.audio.is_silent",
        lambda path: "chunk_001" in path.name,
    )

    def fake_transcribe(path, model, language):
        word = "early" if "chunk_000" in path.name else "later"
        return {
            "segments": [{
                "start": 6.0, "end": 7.0, "text": word, "no_speech_prob": 0.0,
            }],
        }

    monkeypatch.setattr("scribe_md.cli.transcriber.transcribe_audio", fake_transcribe)

    written: list[str] = []

    def write_fn(text, _output):
        written.append(text)

    _live_chunked(
        tmp_path / "out.md",
        None,
        "ko",
        "large-v3",
        True,
        30,
        5,
        False,
        write_fn=write_fn,
    )

    text = written[0]
    # Chunk 2 starts at 2 * 30 - 5 = 55s. The segment is past the 5s overlap,
    # so its timestamp is 55 + 6 = 61s. Treating the chunk as index 1 would
    # stamp it at 31s.
    assert "[00:01:01]" in text
    assert "[00:00:31]" not in text
