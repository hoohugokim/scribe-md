import pytest
import typer
from typer.testing import CliRunner

from scribe_md.cli import app, _apply_postprocessing

runner = CliRunner()


def test_live_is_macos_only_message_on_linux(monkeypatch):
    monkeypatch.setattr("scribe_md.cli.platform_support.is_linux", lambda: True)
    result = runner.invoke(app, ["live"])
    assert result.exit_code == 1
    assert "macOS-only" in result.output


def test_summarize_blocked_on_linux(monkeypatch):
    monkeypatch.setattr("scribe_md.cli.platform_support.is_linux", lambda: True)
    with pytest.raises(typer.Exit) as exc_info:
        _apply_postprocessing("some transcript text", summarize=True)
    assert exc_info.value.exit_code == 1


def test_file_summarize_fails_fast_on_linux(monkeypatch, tmp_path):
    # The guard must fire before any transcription work begins, even though
    # the audio file exists.
    monkeypatch.setattr("scribe_md.cli.platform_support.is_linux", lambda: True)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00" * 100)
    result = runner.invoke(app, ["file", str(audio), "--summarize"])
    assert result.exit_code == 1
    assert "macOS-only" in result.output


def test_summarize_allowed_on_macos(monkeypatch):
    monkeypatch.setattr("scribe_md.cli.platform_support.is_linux", lambda: False)
    monkeypatch.setattr(
        "scribe_md.cli.postprocess.summarize_with_llm",
        lambda text, model=None: "a summary",
    )
    out = _apply_postprocessing("transcript", summarize=True)
    assert "## Summary" in out
    assert "a summary" in out
