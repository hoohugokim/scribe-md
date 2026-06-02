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


# ---------------------------------------------------------------------------
# _resolve_common_options  (the shared file/url/live resolution block)
# ---------------------------------------------------------------------------


def _resolve_common(cfg, **overrides):
    """Call _resolve_common_options with all CLI inputs defaulting to None."""
    kwargs = dict(
        model=None, language=None, timestamps=None, timestamp_mode=None,
        paragraph_gap=None, overlap_seconds=None, vault=None, daily_note=False,
        frontmatter=None, clean=None, summary_model=None, diarize_flag=None,
        hf_token=None, num_speakers=None,
    )
    kwargs.update(overrides)
    return cli._resolve_common_options(cfg, **kwargs)


class TestResolveCommonOptions:
    def test_cli_overrides_and_config_fallbacks(self):
        cfg = ScribeMdConfig(model="large-v3", paragraph_gap=2.0, overlap_seconds=5)
        opts = _resolve_common(cfg, model="small")
        assert opts.model == "small"          # CLI wins
        assert opts.paragraph_gap == 2.0      # config fallback
        assert opts.overlap_seconds == 5
        assert opts.daily_note_folder == "Daily Notes"

    def test_empty_config_language_is_auto(self):
        opts = _resolve_common(ScribeMdConfig(language=""))
        assert opts.language is None

    def test_config_language_used(self):
        opts = _resolve_common(ScribeMdConfig(language="ko"))
        assert opts.language == "ko"

    def test_frontmatter_defaults_off_without_vault(self):
        opts = _resolve_common(ScribeMdConfig(vault=""))
        assert opts.frontmatter is False

    def test_frontmatter_defaults_on_with_vault(self):
        opts = _resolve_common(ScribeMdConfig(vault="/vault"))
        assert opts.vault == "/vault"
        assert opts.frontmatter is True

    def test_explicit_frontmatter_flag_wins(self):
        opts = _resolve_common(ScribeMdConfig(vault="/vault"), frontmatter=False)
        assert opts.frontmatter is False

    def test_timestamp_reconciliation(self):
        opts = _resolve_common(ScribeMdConfig(timestamps=True, timestamp_mode="segment"))
        assert opts.ts is True and opts.ts_mode == "segment"

    def test_no_timestamps_forces_none(self):
        opts = _resolve_common(ScribeMdConfig(), timestamps=False)
        assert opts.ts is False and opts.ts_mode == "none"

    def test_daily_note_without_vault_exits(self):
        with pytest.raises(typer.Exit):
            _resolve_common(ScribeMdConfig(vault=""), daily_note=True)

    def test_invalid_timestamp_mode_exits(self):
        with pytest.raises(typer.Exit):
            _resolve_common(ScribeMdConfig(), timestamp_mode="hourly")

    def test_diarize_and_speaker_fields(self):
        cfg = ScribeMdConfig(diarize=True, hf_token="tok", num_speakers=2)
        opts = _resolve_common(cfg)
        assert opts.diarize is True
        assert opts.hf_token == "tok"
        assert opts.num_speakers == 2
