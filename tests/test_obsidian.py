"""Tests for the obsidian module — Obsidian vault integration (Phase 4.3)."""

import re
from datetime import datetime
from pathlib import Path

from scribe_md.obsidian import (
    append_to_daily_note,
    build_frontmatter,
    format_duration,
    resolve_vault_output,
    write_with_frontmatter,
)


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(45.0) == "0:45"

    def test_minutes_and_seconds(self):
        assert format_duration(332.0) == "5:32"

    def test_hours(self):
        assert format_duration(3661.0) == "1:01:01"

    def test_zero(self):
        assert format_duration(0.0) == "0:00"

    def test_exact_minute(self):
        assert format_duration(60.0) == "1:00"

    def test_exact_hour(self):
        assert format_duration(3600.0) == "1:00:00"

    def test_fractional_seconds_truncated(self):
        assert format_duration(65.9) == "1:05"


# ---------------------------------------------------------------------------
# build_frontmatter
# ---------------------------------------------------------------------------


class TestBuildFrontmatter:
    def test_basic_metadata(self):
        meta = {
            "date": "2026-03-13",
            "source": "file: test.wav",
            "duration": "5:32",
            "language": "ko",
            "model": "large-v3",
            "tags": ["transcription"],
        }
        result = build_frontmatter(meta)
        assert result.startswith("---\n")
        assert result.endswith("---\n")
        assert 'date: "2026-03-13"' in result
        assert 'language: "ko"' in result
        assert 'model: "large-v3"' in result
        assert 'duration: "5:32"' in result
        assert 'tags: ["transcription"]' in result

    def test_source_with_colon_is_quoted(self):
        meta = {"source": "YouTube: My Video Title"}
        result = build_frontmatter(meta)
        assert '"YouTube: My Video Title"' in result

    def test_source_with_embedded_quotes_is_escaped(self):
        meta = {"source": 'YouTube: Some "quoted" title'}
        result = build_frontmatter(meta)
        assert r'source: "YouTube: Some \"quoted\" title"' in result

    def test_source_with_backslash_is_escaped(self):
        meta = {"source": r"file: C:\path\to\thing"}
        result = build_frontmatter(meta)
        assert r'source: "file: C:\\path\\to\\thing"' in result

    def test_empty_metadata(self):
        result = build_frontmatter({})
        assert result == "---\n---\n"

    def test_missing_optional_fields(self):
        meta = {"date": "2026-03-13", "model": "large-v3"}
        result = build_frontmatter(meta)
        assert 'date: "2026-03-13"' in result
        assert 'model: "large-v3"' in result
        assert "source" not in result
        assert "language" not in result
        assert "duration" not in result

    def test_empty_string_fields_omitted(self):
        meta = {"date": "2026-03-13", "language": "", "model": "large-v3"}
        result = build_frontmatter(meta)
        assert "language" not in result

    def test_multiple_tags(self):
        meta = {"tags": ["transcription", "meeting"]}
        result = build_frontmatter(meta)
        assert 'tags: ["transcription", "meeting"]' in result


# ---------------------------------------------------------------------------
# write_with_frontmatter
# ---------------------------------------------------------------------------


class TestWriteWithFrontmatter:
    def test_writes_file_with_frontmatter(self, tmp_path):
        output = tmp_path / "test.md"
        meta = {"date": "2026-03-13", "model": "large-v3"}
        write_with_frontmatter(output, "Hello world.", meta)

        content = output.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert 'date: "2026-03-13"' in content
        assert "Hello world." in content

    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "sub" / "dir" / "test.md"
        meta = {"date": "2026-03-13"}
        write_with_frontmatter(output, "Content.", meta)

        assert output.exists()
        assert "Content." in output.read_text(encoding="utf-8")

    def test_frontmatter_separated_from_body(self, tmp_path):
        output = tmp_path / "test.md"
        meta = {"date": "2026-03-13"}
        write_with_frontmatter(output, "Body text.", meta)

        content = output.read_text(encoding="utf-8")
        # There should be a blank line between frontmatter closing --- and body
        assert "---\n\nBody text." in content


# ---------------------------------------------------------------------------
# append_to_daily_note
# ---------------------------------------------------------------------------


class TestAppendToDailyNote:
    def test_creates_new_daily_note(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()

        meta = {"source": "file: test.wav"}
        path = append_to_daily_note(vault, "Daily Notes", "Transcribed text.", meta)

        today = datetime.now().strftime("%Y-%m-%d")
        assert path.name == f"{today}.md"
        assert path.parent.name == "Daily Notes"

        content = path.read_text(encoding="utf-8")
        assert "## Transcription" in content
        assert "file: test.wav" in content
        assert "Transcribed text." in content

    def test_appends_to_existing_daily_note(self, tmp_path):
        vault = tmp_path / "vault"
        daily_dir = vault / "Daily Notes"
        daily_dir.mkdir(parents=True)

        today = datetime.now().strftime("%Y-%m-%d")
        note_path = daily_dir / f"{today}.md"
        note_path.write_text("# Today\n\nExisting content.\n", encoding="utf-8")

        meta = {"source": "live: system audio"}
        path = append_to_daily_note(vault, "Daily Notes", "New transcription.", meta)

        assert path == note_path
        content = path.read_text(encoding="utf-8")
        # Original content preserved
        assert "Existing content." in content
        # New transcription appended
        assert "New transcription." in content
        assert "## Transcription" in content

    def test_section_header_includes_time(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()

        meta = {}
        path = append_to_daily_note(vault, "Daily Notes", "Text.", meta)

        content = path.read_text(encoding="utf-8")
        # Should have time in HH:MM format
        assert re.search(r"## Transcription \(\d{2}:\d{2}\)", content)

    def test_creates_daily_folder(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()

        meta = {}
        path = append_to_daily_note(vault, "My Notes", "Text.", meta)

        assert (vault / "My Notes").is_dir()
        assert path.parent == vault / "My Notes"


# ---------------------------------------------------------------------------
# resolve_vault_output
# ---------------------------------------------------------------------------


class TestResolveVaultOutput:
    def test_resolves_within_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()

        result = resolve_vault_output(vault, "output.md")
        assert result == vault / "output.md"

    def test_expands_user_home(self):
        result = resolve_vault_output(Path("~/Documents/Vault"), "test.md")
        assert "~" not in str(result)
        assert result.name == "test.md"

    def test_resolves_to_absolute(self, tmp_path):
        result = resolve_vault_output(tmp_path / "vault", "file.md")
        assert result.is_absolute()
