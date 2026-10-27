# tests/test_cli_batch.py
import pytest
from pathlib import Path
from typer.testing import CliRunner
from scribe_md import cli
from scribe_md.cli import app, _resolve_gpu_ids
from scribe_md.transcriber import TranscriptionError

runner = CliRunner()


def test_resolve_gpu_ids_sequential_default(monkeypatch):
    monkeypatch.setattr(cli.gpu, "discover_cuda_devices", lambda: [0, 1])
    # default (None / "1") => not parallel
    assert _resolve_gpu_ids(None) == []
    assert _resolve_gpu_ids("1") == []


def test_resolve_gpu_ids_auto(monkeypatch):
    monkeypatch.setattr(cli.gpu, "discover_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(cli, "_backend_is_cuda", lambda: True)
    assert _resolve_gpu_ids("auto") == [0, 1]


def test_resolve_gpu_ids_auto_single_gpu_returns_sequential(monkeypatch):
    """--gpus auto on a single-GPU box hits the `len(ids) > 1` guard and must
    return [] (sequential fallback) without raising, so batch callers use the
    existing sequential path.  This exercises cli.py line 381."""
    monkeypatch.setattr(cli.gpu, "discover_cuda_devices", lambda: [0])
    monkeypatch.setattr(cli, "_backend_is_cuda", lambda: True)
    # resolve_gpu_spec("auto", [0]) returns [0]; len([0]) == 1 → guard kicks in
    assert _resolve_gpu_ids("auto") == []


def test_resolve_gpu_ids_falls_back_when_not_cuda(monkeypatch):
    monkeypatch.setattr(cli.gpu, "discover_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(cli, "_backend_is_cuda", lambda: False)
    # non-CUDA backend: warn + sequential ([] means "run sequentially")
    assert _resolve_gpu_ids("auto") == []


def test_file_rejects_output_with_multiple_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("scribe_md.cli.load_config", lambda: cli.ScribeMdConfig())
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"\x00" * 100)
    b.write_bytes(b"\x00" * 100)
    result = runner.invoke(app, ["file", str(a), str(b), "-o", "out.md"])
    assert result.exit_code == 1
    assert "single input only" in result.output


def test_url_rejects_output_with_multiple_inputs(monkeypatch):
    """url command also calls _validate_single_output; verify it rejects -o
    when multiple URLs are given (mirrors test_file_rejects_output_with_multiple_inputs)."""
    monkeypatch.setattr("scribe_md.cli.load_config", lambda: cli.ScribeMdConfig())
    result = runner.invoke(
        app, ["url", "https://example.com/a", "https://example.com/b", "-o", "out.md"]
    )
    assert result.exit_code == 1
    assert "single input only" in result.output


def test_file_batch_continues_after_per_file_error(monkeypatch, tmp_path):
    """A TranscriptionError on the middle file must not abort the batch.

    The other files must still be processed and the command exits 0.
    """
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    c = tmp_path / "c.wav"
    for f in (a, b, c):
        f.write_bytes(b"\x00" * 100)

    monkeypatch.setattr("scribe_md.cli.load_config", lambda: cli.ScribeMdConfig())

    converted_files: list[Path] = []

    def fake_convert(src, dst):
        converted_files.append(src)
        # Raise for the middle file
        if src == b:
            raise TranscriptionError("simulated failure for b.wav")
        dst.write_bytes(b"\x00" * 100)

    monkeypatch.setattr("scribe_md.cli.audio.convert_to_16k_mono", fake_convert)
    monkeypatch.setattr("scribe_md.cli.audio.get_duration", lambda p: 1.0)
    monkeypatch.setattr("scribe_md.cli._transcribe_single",
                        lambda *a, **kw: None)
    monkeypatch.setattr("scribe_md.cli._resolve_gpu_ids", lambda spec: [])

    result = runner.invoke(app, ["file", str(a), str(b), str(c)])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}. Output:\n{result.output}"
    # a and c must have been attempted; b raised but should be skipped
    assert a in converted_files
    assert b in converted_files
    assert c in converted_files
