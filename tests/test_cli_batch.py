# tests/test_cli_batch.py
import pytest
from typer.testing import CliRunner
from scribe_md import cli
from scribe_md.cli import app, _resolve_gpu_ids

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
