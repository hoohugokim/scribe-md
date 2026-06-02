import pytest
import typer
from pathlib import Path
from typer.testing import CliRunner

from scribe_md.cli import (
    app,
    _apply_postprocessing,
    _resolve_incremental_output,
    _should_chunk,
)
from scribe_md.config import ScribeMdConfig

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


def test_non_positive_chunk_seconds_means_no_chunking():
    assert not _should_chunk(duration=60, chunk_seconds=0)
    assert not _should_chunk(duration=60, chunk_seconds=-1)
    assert _should_chunk(duration=60, chunk_seconds=30)


def test_incremental_output_resolves_relative_output_inside_vault(tmp_path):
    vault = tmp_path / "vault"
    enabled, path = _resolve_incremental_output(
        tmp_path / "outside" / "transcription.md",
        vault=str(vault),
        daily_note=False,
        incremental=True,
    )

    assert enabled
    assert path == tmp_path / "outside" / "transcription.md"

    enabled, path = _resolve_incremental_output(
        Path("transcription.md"),
        vault=str(vault),
        daily_note=False,
        incremental=True,
    )

    assert enabled
    assert path == vault.resolve() / "transcription.md"


def test_incremental_output_disabled_for_daily_note_with_vault(tmp_path):
    enabled, path = _resolve_incremental_output(
        Path("transcription.md"),
        vault=str(tmp_path / "vault"),
        daily_note=True,
        incremental=True,
    )

    assert not enabled
    assert path is None


def test_daily_note_requires_vault(monkeypatch, tmp_path):
    monkeypatch.setattr("scribe_md.cli.load_config", lambda: ScribeMdConfig())
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00" * 100)

    result = runner.invoke(app, ["file", str(audio), "--daily-note"])

    assert result.exit_code == 1
    assert "--daily-note requires --vault" in result.output
