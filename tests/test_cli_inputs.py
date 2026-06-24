import pytest
import typer
from scribe_md.cli import _collect_inputs, _validate_single_output


def test_collect_from_positional_only():
    assert _collect_inputs(["a.mp4", "b.mp4"], None) == ["a.mp4", "b.mp4"]


def test_collect_from_file_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("url1\n\n# comment\n  url2  \n")
    assert _collect_inputs([], f) == ["url1", "url2"]


def test_collect_merges_positional_and_file(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("url2\n")
    assert _collect_inputs(["url1"], f) == ["url1", "url2"]


def test_collect_empty_raises_exit():
    with pytest.raises(typer.Exit) as exc_info:
        _collect_inputs([], None)
    assert exc_info.value.exit_code == 1


def test_output_with_multiple_inputs_raises():
    from pathlib import Path
    with pytest.raises(typer.Exit) as exc_info:
        _validate_single_output(["a", "b"], Path("out.md"))
    assert exc_info.value.exit_code == 1
    # single input + -o is fine
    _validate_single_output(["a"], Path("out.md"))
