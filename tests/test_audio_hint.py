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
