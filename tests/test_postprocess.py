"""Tests for the postprocess module — LLM post-processing (Phase 4.4)."""

from scribe_md.postprocess import (
    clean_transcription,
    _remove_consecutive_duplicates,
    _remove_hallucination_lines,
    _remove_hallucination_phrases_inline,
    _normalize_whitespace,
    summarize_with_llm,
)
import pytest


# ---------------------------------------------------------------------------
# clean_transcription (integration of all rule-based steps)
# ---------------------------------------------------------------------------


class TestCleanTranscription:
    def test_empty_string(self):
        assert clean_transcription("") == ""

    def test_no_artifacts(self):
        text = "This is a clean transcription. It has no issues."
        result = clean_transcription(text)
        assert "This is a clean transcription." in result
        assert "It has no issues." in result

    def test_removes_thank_you_for_watching(self):
        text = "The lecture covers quantum mechanics.\nThank you for watching.\nThe end."
        result = clean_transcription(text)
        assert "quantum mechanics" in result
        assert "Thank you for watching" not in result

    def test_removes_subscribe(self):
        text = "Some content here.\nSubscribe\nMore content."
        result = clean_transcription(text)
        assert "Some content here." in result
        assert "Subscribe" not in result
        assert "More content." in result

    def test_removes_korean_artifacts(self):
        text = "이것은 좋은 강의입니다.\n자막 제공자\n끝입니다."
        result = clean_transcription(text)
        assert "좋은 강의" in result
        assert "자막 제공자" not in result

    def test_removes_duplicate_sentences(self):
        text = "Hello world. Hello world. This is different."
        result = clean_transcription(text)
        # Should have only one "Hello world"
        assert result.count("Hello world") == 1
        assert "This is different." in result

    def test_normalizes_whitespace(self):
        text = "Too   many   spaces.\n\n\n\nToo many newlines."
        result = clean_transcription(text)
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_combined_artifacts(self):
        text = (
            "Introduction to the topic.\n"
            "Introduction to the topic.\n"
            "Thank you for watching.\n"
            "The main content follows.\n"
            "\n\n\n\n"
            "Subscribe\n"
            "Conclusion of the lecture."
        )
        result = clean_transcription(text)
        assert "Introduction to the topic" in result
        assert result.count("Introduction to the topic") == 1
        assert "Thank you for watching" not in result
        assert "Subscribe" not in result
        assert "The main content follows." in result
        assert "Conclusion of the lecture." in result
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# _remove_consecutive_duplicates
# ---------------------------------------------------------------------------


class TestRemoveConsecutiveDuplicates:
    def test_no_duplicates(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _remove_consecutive_duplicates(text)
        assert "First sentence." in result
        assert "Second sentence." in result
        assert "Third sentence." in result

    def test_consecutive_duplicates_removed(self):
        text = "Hello world. Hello world. Next sentence."
        result = _remove_consecutive_duplicates(text)
        assert result.count("Hello world") == 1
        assert "Next sentence." in result

    def test_case_insensitive_dedup(self):
        text = "Hello World. hello world. Different."
        result = _remove_consecutive_duplicates(text)
        assert result.count("ello") == 1  # Only one version kept
        assert "Different." in result

    def test_non_consecutive_duplicates_kept(self):
        text = "Apple. Banana. Apple."
        result = _remove_consecutive_duplicates(text)
        assert result.count("Apple") == 2

    def test_triple_duplicate(self):
        text = "Repeated. Repeated. Repeated. Done."
        result = _remove_consecutive_duplicates(text)
        assert result.count("Repeated") == 1
        assert "Done." in result

    def test_empty_string(self):
        assert _remove_consecutive_duplicates("") == ""

    def test_single_sentence(self):
        text = "Just one sentence."
        result = _remove_consecutive_duplicates(text)
        assert result == "Just one sentence."


# ---------------------------------------------------------------------------
# _remove_hallucination_lines
# ---------------------------------------------------------------------------


class TestRemoveHallucinationLines:
    def test_removes_thank_you_line(self):
        text = "Real content.\nThank you for watching.\nMore content."
        result = _remove_hallucination_lines(text)
        assert "Real content." in result
        assert "Thank you for watching" not in result
        assert "More content." in result

    def test_removes_subscribe_line(self):
        text = "Content.\nSubscribe.\nMore."
        result = _remove_hallucination_lines(text)
        assert "Subscribe" not in result

    def test_removes_korean_subtitle_credit(self):
        text = "한국어 내용.\n자막 제공자\n더 많은 내용."
        result = _remove_hallucination_lines(text)
        assert "자막 제공자" not in result

    def test_preserves_partial_match(self):
        text = "I want to subscribe to this service.\nEnd."
        result = _remove_hallucination_lines(text)
        # "subscribe to this service" is not an exact line match
        assert "subscribe to this service" in result

    def test_case_insensitive_removal(self):
        text = "Content.\nthank you for watching\nEnd."
        result = _remove_hallucination_lines(text)
        assert "thank you for watching" not in result

    def test_removes_line_with_trailing_punctuation(self):
        text = "Content.\nThank you for watching!\nEnd."
        result = _remove_hallucination_lines(text)
        assert "Thank you for watching" not in result

    def test_removes_bare_you(self):
        # "you" is a known Whisper hallucination in silence
        text = "Content.\nyou\nEnd."
        result = _remove_hallucination_lines(text)
        assert "\nyou\n" not in result

    def test_preserves_normal_lines(self):
        text = "This is normal text.\nAnother normal line.\nEnd."
        result = _remove_hallucination_lines(text)
        assert result == text


# ---------------------------------------------------------------------------
# _normalize_whitespace
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_collapses_multiple_blank_lines(self):
        text = "A.\n\n\n\nB."
        result = _normalize_whitespace(text)
        assert result == "A.\n\nB."

    def test_collapses_excessive_spaces(self):
        text = "Too   many    spaces   here."
        result = _normalize_whitespace(text)
        assert result == "Too many spaces here."

    def test_strips_leading_trailing(self):
        text = "  \n\nContent.\n\n  "
        result = _normalize_whitespace(text)
        assert result == "Content."

    def test_preserves_single_blank_line(self):
        text = "A.\n\nB."
        result = _normalize_whitespace(text)
        assert result == "A.\n\nB."

    def test_removes_trailing_spaces_on_lines(self):
        text = "Line one.   \nLine two.  "
        result = _normalize_whitespace(text)
        assert "   \n" not in result


# ---------------------------------------------------------------------------
# summarize_with_llm — import error handling
# ---------------------------------------------------------------------------


class TestSummarizeWithLlm:
    def test_raises_import_error_when_mlx_lm_missing(self, monkeypatch):
        """When mlx_lm is not installed, summarize_with_llm should raise
        ImportError with a helpful message."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mlx_lm":
                raise ImportError("No module named 'mlx_lm'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pip install mlx-lm"):
            summarize_with_llm("Some transcription text.")
