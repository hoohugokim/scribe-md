"""Tests for the diarize module — speaker diarization (Phase 4.1)."""

import pytest

from scribe_md.diarize import assign_speakers, _normalize_speaker_labels
from scribe_md.merger import merge_segments


# ---------------------------------------------------------------------------
# Helper to build segment / turn dicts concisely
# ---------------------------------------------------------------------------


def _seg(start: float, end: float, text: str, **kwargs) -> dict:
    d = {"start": start, "end": end, "text": text}
    d.update(kwargs)
    return d


def _turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


# ---------------------------------------------------------------------------
# _normalize_speaker_labels
# ---------------------------------------------------------------------------


class TestNormalizeSpeakerLabels:
    def test_single_speaker(self):
        turns = [_turn(0, 5, "SPEAKER_00"), _turn(5, 10, "SPEAKER_00")]
        result = _normalize_speaker_labels(turns)
        assert all(t["speaker"] == "Speaker 1" for t in result)

    def test_two_speakers_ordered_by_appearance(self):
        turns = [
            _turn(0, 5, "SPEAKER_01"),
            _turn(5, 10, "SPEAKER_00"),
            _turn(10, 15, "SPEAKER_01"),
        ]
        result = _normalize_speaker_labels(turns)
        assert result[0]["speaker"] == "Speaker 1"  # SPEAKER_01 first
        assert result[1]["speaker"] == "Speaker 2"  # SPEAKER_00 second
        assert result[2]["speaker"] == "Speaker 1"

    def test_empty_turns(self):
        assert _normalize_speaker_labels([]) == []

    def test_preserves_timestamps(self):
        turns = [_turn(1.5, 3.0, "SPEAKER_00")]
        result = _normalize_speaker_labels(turns)
        assert result[0]["start"] == 1.5
        assert result[0]["end"] == 3.0


# ---------------------------------------------------------------------------
# assign_speakers
# ---------------------------------------------------------------------------


class TestAssignSpeakers:
    def test_basic_assignment(self):
        segments = [
            _seg(0.0, 5.0, " Hello world."),
            _seg(5.0, 10.0, " How are you?"),
        ]
        turns = [
            _turn(0.0, 6.0, "Speaker 1"),
            _turn(6.0, 12.0, "Speaker 2"),
        ]
        result = assign_speakers(segments, turns)
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"
        # Original fields preserved
        assert result[0]["text"] == " Hello world."

    def test_overlap_picks_best_match(self):
        """When a segment overlaps multiple turns, pick the one with more overlap."""
        segments = [_seg(4.0, 8.0, " Overlapping.")]
        turns = [
            _turn(0.0, 5.0, "Speaker 1"),   # 1s overlap (4-5)
            _turn(5.0, 10.0, "Speaker 2"),   # 3s overlap (5-8)
        ]
        result = assign_speakers(segments, turns)
        assert result[0]["speaker"] == "Speaker 2"

    def test_no_matching_turn(self):
        segments = [_seg(20.0, 25.0, " No match.")]
        turns = [_turn(0.0, 10.0, "Speaker 1")]
        result = assign_speakers(segments, turns)
        assert result[0]["speaker"] == "Unknown"

    def test_empty_segments(self):
        assert assign_speakers([], [_turn(0, 5, "Speaker 1")]) == []

    def test_empty_turns(self):
        segments = [_seg(0.0, 5.0, " Hello.")]
        result = assign_speakers(segments, [])
        assert result[0]["speaker"] == "Unknown"

    def test_time_offset(self):
        """For chunked transcription, segment times need offset."""
        segments = [_seg(0.0, 5.0, " Chunk two.")]
        turns = [
            _turn(0.0, 10.0, "Speaker 1"),
            _turn(10.0, 20.0, "Speaker 2"),
        ]
        # With offset=15, segment global time is 15-20 -> Speaker 2
        result = assign_speakers(segments, turns, time_offset=15.0)
        assert result[0]["speaker"] == "Speaker 2"

    def test_time_offset_zero(self):
        segments = [_seg(0.0, 5.0, " First chunk.")]
        turns = [_turn(0.0, 10.0, "Speaker 1")]
        result = assign_speakers(segments, turns, time_offset=0.0)
        assert result[0]["speaker"] == "Speaker 1"


# ---------------------------------------------------------------------------
# Merger integration with speaker labels
# ---------------------------------------------------------------------------


class TestMergerSpeakerLabels:
    """Test that merge_segments renders speaker labels correctly."""

    def test_speaker_labels_none_mode(self):
        segments = [
            _seg(0.0, 2.0, " Hello.", speaker="Speaker 1"),
            _seg(2.0, 4.0, " Hi there.", speaker="Speaker 2"),
            _seg(4.0, 6.0, " How are you?", speaker="Speaker 2"),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none", paragraph_gap=100.0,
        )
        assert "**Speaker 1:**" in result
        assert "**Speaker 2:**" in result

    def test_speaker_change_forces_paragraph(self):
        segments = [
            _seg(0.0, 2.0, " Hello.", speaker="Speaker 1"),
            _seg(2.1, 4.0, " Hi.", speaker="Speaker 2"),  # small gap, but speaker change
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none", paragraph_gap=100.0,  # huge gap threshold
        )
        # Speaker change should force a paragraph break even with huge gap threshold
        parts = result.strip().split("\n\n")
        assert len(parts) == 2

    def test_no_speaker_key_backward_compat(self):
        """Segments without 'speaker' key should render as before."""
        segments = [
            _seg(0.0, 2.0, " Hello."),
            _seg(2.0, 4.0, " World."),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none",
        )
        assert "Speaker" not in result
        assert "Hello." in result

    def test_speaker_labels_segment_mode(self):
        segments = [
            _seg(0.0, 2.0, " Hello.", speaker="Speaker 1"),
            _seg(5.0, 7.0, " Hi.", speaker="Speaker 2"),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="segment",
        )
        assert "**Speaker 1:** Hello." in result
        assert "**Speaker 2:** Hi." in result

    def test_speaker_labels_paragraph_mode(self):
        segments = [
            _seg(0.0, 2.0, " Hello.", speaker="Speaker 1"),
            _seg(2.0, 3.0, " More.", speaker="Speaker 1"),
            _seg(5.0, 7.0, " Hi.", speaker="Speaker 2"),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="paragraph",
        )
        assert "**Speaker 1:**" in result
        assert "**Speaker 2:**" in result

    def test_same_speaker_no_repeated_label(self):
        """Same speaker across paragraph break should not re-emit label."""
        segments = [
            _seg(0.0, 2.0, " First.", speaker="Speaker 1"),
            _seg(5.0, 7.0, " Second.", speaker="Speaker 1"),  # gap break, same speaker
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none", paragraph_gap=2.0,
        )
        # Speaker 1 label should appear only once (at the start)
        assert result.count("**Speaker 1:**") == 1

    def test_speaker_labels_minute_mode(self):
        segments = [
            _seg(0.0, 2.0, " Hello.", speaker="Speaker 1"),
            _seg(60.0, 62.0, " Minute one.", speaker="Speaker 2"),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="minute",
        )
        assert "**Speaker 1:**" in result
        assert "**Speaker 2:**" in result


# ---------------------------------------------------------------------------
# DiarizationError
# ---------------------------------------------------------------------------


class TestDiarizationError:
    def test_missing_token_raises(self):
        """diarize_audio without a token should raise DiarizationError."""
        from scribe_md.diarize import diarize_audio, DiarizationError
        from pathlib import Path
        import os

        # Ensure HF_TOKEN is not set for this test
        old_token = os.environ.pop("HF_TOKEN", None)
        try:
            with pytest.raises(DiarizationError, match="HuggingFace token"):
                diarize_audio(Path("dummy.wav"), hf_token="")
        finally:
            if old_token is not None:
                os.environ["HF_TOKEN"] = old_token
