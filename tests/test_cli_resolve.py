"""Characterization tests for cli.py resolution/validation helpers.

These lock the existing behaviour of the pure decision helpers so the
file/url/live option-and-resolution consolidation can be refactored safely.
"""

from pathlib import Path

import pytest
import typer

from scribe_md import cli
from scribe_md.config import ScribeMdConfig


# ---------------------------------------------------------------------------
# _resolve  (CLI value wins unless None; falsy-but-set values are kept)
# ---------------------------------------------------------------------------


class TestResolve:
    def test_none_falls_back_to_config(self):
        assert cli._resolve(None, "cfg") == "cfg"

    def test_value_overrides_config(self):
        assert cli._resolve("cli", "cfg") == "cli"

    def test_falsy_but_set_value_is_kept(self):
        # 0 / "" / False are explicit choices, not "unset".
        assert cli._resolve(0, 1800) == 0
        assert cli._resolve("", "cfg") == ""
        assert cli._resolve(False, True) is False


# ---------------------------------------------------------------------------
# _resolve_language  (empty config language means auto-detect -> None)
# ---------------------------------------------------------------------------


class TestResolveLanguage:
    def test_cli_value_wins(self):
        cfg = ScribeMdConfig(language="ko")
        assert cli._resolve_language("en", cfg) == "en"

    def test_empty_config_language_is_auto(self):
        cfg = ScribeMdConfig(language="")
        assert cli._resolve_language(None, cfg) is None

    def test_config_language_used_when_no_cli(self):
        cfg = ScribeMdConfig(language="ko")
        assert cli._resolve_language(None, cfg) == "ko"


# ---------------------------------------------------------------------------
# _resolve_timestamp_flags
# ---------------------------------------------------------------------------


class TestResolveTimestampFlags:
    def test_no_timestamps_forces_none(self):
        assert cli._resolve_timestamp_flags(False, "segment") == (False, "none")

    def test_mode_none_disables(self):
        assert cli._resolve_timestamp_flags(True, "none") == (False, "none")

    def test_segment_mode_preserved(self):
        assert cli._resolve_timestamp_flags(True, "segment") == (True, "segment")

    def test_paragraph_mode_preserved(self):
        assert cli._resolve_timestamp_flags(True, "paragraph") == (True, "paragraph")


# ---------------------------------------------------------------------------
# _validate_timestamp_mode
# ---------------------------------------------------------------------------


class TestValidateTimestampMode:
    @pytest.mark.parametrize("mode", ["segment", "paragraph", "minute", "none"])
    def test_valid_modes_pass(self, mode):
        cli._validate_timestamp_mode(mode)  # must not raise

    def test_invalid_mode_exits(self):
        with pytest.raises(typer.Exit):
            cli._validate_timestamp_mode("hourly")


# ---------------------------------------------------------------------------
# _should_chunk
# ---------------------------------------------------------------------------


class TestShouldChunk:
    def test_zero_chunk_seconds_never_chunks(self):
        assert cli._should_chunk(100.0, 0) is False

    def test_chunks_when_duration_exceeds_chunk(self):
        assert cli._should_chunk(100.0, 50.0) is True

    def test_no_chunk_when_shorter_than_chunk(self):
        assert cli._should_chunk(50.0, 100.0) is False


# ---------------------------------------------------------------------------
# _validate_daily_note
# ---------------------------------------------------------------------------


class TestValidateDailyNote:
    def test_daily_note_without_vault_exits(self):
        with pytest.raises(typer.Exit):
            cli._validate_daily_note(True, "")

    def test_daily_note_with_vault_ok(self):
        cli._validate_daily_note(True, "/some/vault")  # must not raise

    def test_no_daily_note_ok(self):
        cli._validate_daily_note(False, "")  # must not raise


# ---------------------------------------------------------------------------
# _resolve_incremental_output
# ---------------------------------------------------------------------------


class TestResolveIncrementalOutput:
    def test_disabled_returns_no_draft(self):
        assert cli._resolve_incremental_output(
            Path("out.md"), vault="", daily_note=False, incremental=False
        ) == (False, None)

    def test_daily_note_disables_incremental(self, tmp_path):
        enabled, draft = cli._resolve_incremental_output(
            Path("out.md"), vault=str(tmp_path), daily_note=True, incremental=True
        )
        assert enabled is False and draft is None

    def test_relative_output_resolves_into_vault(self, tmp_path):
        enabled, draft = cli._resolve_incremental_output(
            Path("out.md"), vault=str(tmp_path), daily_note=False, incremental=True
        )
        assert enabled is True
        assert draft == (tmp_path.expanduser().resolve() / "out.md")

    def test_plain_output_used_as_is(self):
        enabled, draft = cli._resolve_incremental_output(
            Path("out.md"), vault="", daily_note=False, incremental=True
        )
        assert enabled is True and draft == Path("out.md")
