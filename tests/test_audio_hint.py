import pytest

from scribe_md import audio
from scribe_md.audio import AudioConversionError


def test_convert_missing_ffmpeg_uses_platform_hint(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.platform', 'linux')

    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(audio.subprocess, 'run', boom)
    monkeypatch.setattr(audio, '_check_disk_space', lambda *a, **k: None)
    with pytest.raises(AudioConversionError) as excinfo:
        audio.convert_to_16k_mono(tmp_path / 'in.mp3', tmp_path / 'out.wav')
    assert 'apt' in str(excinfo.value)


@pytest.mark.parametrize("chunk_seconds", [0, -1])
def test_split_audio_rejects_non_positive_chunk_seconds(monkeypatch, tmp_path, chunk_seconds):
    monkeypatch.setattr(audio, "get_duration", lambda path: 10.0)

    with pytest.raises(AudioConversionError, match="chunk_seconds"):
        audio.split_audio(
            tmp_path / "in.wav",
            tmp_path,
            chunk_seconds=chunk_seconds,
            overlap_seconds=0,
        )


def test_split_audio_rejects_overlap_at_or_above_chunk_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "get_duration", lambda path: 10.0)

    with pytest.raises(AudioConversionError, match="overlap_seconds"):
        audio.split_audio(
            tmp_path / "in.wav",
            tmp_path,
            chunk_seconds=5,
            overlap_seconds=5,
        )
