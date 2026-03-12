"""Tests for the merger module — intelligent formatting (Phase 4.2)."""

from scribe_md.merger import merge_segments, _find_sentence_boundary


# ---------------------------------------------------------------------------
# Helper to build segment dicts concisely
# ---------------------------------------------------------------------------

def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


# ---------------------------------------------------------------------------
# Backward compatibility: original behaviour (segment timestamps, no paragraphs)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """merge_segments with default new params should match the old behaviour
    when all segments are within a single chunk and gaps are small."""

    def test_single_chunk_with_timestamps(self):
        segments = [
            _seg(0.0, 2.0, " Hello world."),
            _seg(2.0, 4.0, " How are you?"),
        ]
        result = merge_segments([segments], chunk_duration=0, overlap=0)
        # In segment mode with small gap (< 2.0s default), they land in the
        # same paragraph, joined by \n\n (segment mode keeps each on own line).
        # Note: Whisper segments have leading spaces in their text.
        assert "[00:00:00]" in result
        assert "Hello world." in result
        assert "[00:00:02]" in result
        assert "How are you?" in result

    def test_no_timestamps(self):
        segments = [_seg(0.0, 2.0, " Hello.")]
        result = merge_segments([segments], chunk_duration=0, overlap=0, timestamps=False)
        assert "[00:00:00]" not in result
        assert "Hello." in result


# ---------------------------------------------------------------------------
# Paragraph detection
# ---------------------------------------------------------------------------


class TestParagraphDetection:
    def test_gap_triggers_paragraph_break(self):
        segments = [
            _seg(0.0, 2.0, " First paragraph."),
            _seg(5.0, 7.0, " Second paragraph."),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none",
        )
        # The 3-second gap (5.0 - 2.0) exceeds default 2.0s threshold
        parts = result.strip().split("\n\n")
        assert len(parts) == 2
        assert "First paragraph." in parts[0]
        assert "Second paragraph." in parts[1]

    def test_small_gap_no_paragraph_break(self):
        segments = [
            _seg(0.0, 2.0, " Part one."),
            _seg(2.5, 4.0, " Part two."),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none",
        )
        parts = result.strip().split("\n\n")
        # 0.5s gap is below threshold, so they stay in the same paragraph
        assert len(parts) == 1
        assert "Part one." in parts[0]
        assert "Part two." in parts[0]

    def test_custom_paragraph_gap(self):
        segments = [
            _seg(0.0, 2.0, " A."),
            _seg(3.5, 5.0, " B."),
        ]
        # With gap=5.0, the 1.5s gap should NOT trigger a break
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none", paragraph_gap=5.0,
        )
        parts = result.strip().split("\n\n")
        assert len(parts) == 1

        # With gap=1.0, the 1.5s gap SHOULD trigger a break
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="none", paragraph_gap=1.0,
        )
        parts = result.strip().split("\n\n")
        assert len(parts) == 2


# ---------------------------------------------------------------------------
# Timestamp modes
# ---------------------------------------------------------------------------


class TestTimestampModes:
    SEGMENTS = [
        _seg(0.0, 2.0, " Hello."),
        _seg(2.0, 4.0, " World."),
        _seg(10.0, 12.0, " New paragraph."),
    ]

    def test_segment_mode(self):
        result = merge_segments(
            [self.SEGMENTS], chunk_duration=0, overlap=0,
            timestamp_mode="segment",
        )
        # Each segment gets its own timestamp
        assert result.count("[00:00:") >= 2
        assert "[00:00:10]" in result

    def test_paragraph_mode(self):
        result = merge_segments(
            [self.SEGMENTS], chunk_duration=0, overlap=0,
            timestamp_mode="paragraph",
        )
        # Two paragraphs (gap between 4.0 and 10.0)
        paras = result.strip().split("\n\n")
        assert len(paras) == 2
        # First paragraph starts with timestamp, second too
        assert paras[0].startswith("[00:00:00]")
        assert paras[1].startswith("[00:00:10]")
        # "World." should NOT have its own timestamp in paragraph mode
        assert "[00:00:02]" not in result

    def test_minute_mode(self):
        segments = [
            _seg(0.0, 2.0, " Intro."),
            _seg(2.0, 4.0, " Still minute zero."),
            _seg(60.0, 62.0, " Minute one."),
            _seg(62.0, 64.0, " Still minute one."),
            _seg(120.0, 122.0, " Minute two."),
        ]
        result = merge_segments(
            [segments], chunk_duration=0, overlap=0,
            timestamp_mode="minute", paragraph_gap=100.0,  # no paragraph breaks
        )
        # Should have timestamps at 0:00, 1:00, 2:00 only
        assert "[00:00:00]" in result
        assert "[00:01:00]" in result
        assert "[00:02:00]" in result
        # Should NOT have timestamps at intermediate times
        assert "[00:00:02]" not in result
        assert "[00:01:02]" not in result

    def test_none_mode(self):
        result = merge_segments(
            [self.SEGMENTS], chunk_duration=0, overlap=0,
            timestamp_mode="none",
        )
        assert "[" not in result
        assert "Hello." in result

    def test_no_timestamps_flag_overrides_mode(self):
        result = merge_segments(
            [self.SEGMENTS], chunk_duration=0, overlap=0,
            timestamps=False, timestamp_mode="segment",
        )
        # --no-timestamps should force "none" even if mode is "segment"
        assert "[" not in result


# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------


class TestSentenceBoundary:
    def test_finds_sentence_ending_in_overlap(self):
        segments = [
            _seg(0.0, 1.5, " End of sentence."),
            _seg(1.5, 3.0, " Start of next"),
            _seg(3.0, 6.0, " sentence continues."),
        ]
        boundary = _find_sentence_boundary(segments, overlap=4.0)
        # Prefers the *latest* sentence boundary in the overlap region.
        # Seg at 3.0 starts before overlap=4.0 and ends with ".", so
        # boundary is 6.0 (end of that segment).
        assert boundary == 6.0

    def test_finds_only_sentence_ending_before_overlap(self):
        segments = [
            _seg(0.0, 1.5, " End of sentence."),
            _seg(1.5, 3.0, " Start of next"),
            _seg(3.0, 6.0, " no punctuation here"),
        ]
        boundary = _find_sentence_boundary(segments, overlap=4.0)
        # Only the first segment ends with a period, so boundary is 1.5
        assert boundary == 1.5

    def test_falls_back_to_overlap_when_no_sentence_end(self):
        segments = [
            _seg(0.0, 1.5, " No punctuation here"),
            _seg(1.5, 3.0, " Still no punctuation"),
            _seg(3.0, 6.0, " Final text."),
        ]
        boundary = _find_sentence_boundary(segments, overlap=4.0)
        # No sentence end before overlap=4.0 (the period at seg ending 6.0
        # starts at 3.0 which is < 4.0, so it IS in the overlap region)
        # Actually seg starting at 3.0 < overlap=4.0, and it ends with "."
        assert boundary == 6.0

    def test_prefers_latest_sentence_boundary(self):
        segments = [
            _seg(0.0, 1.0, " First sentence."),
            _seg(1.0, 2.0, " Second sentence."),
            _seg(2.0, 4.0, " Third no end"),
            _seg(4.0, 6.0, " Outside overlap."),
        ]
        boundary = _find_sentence_boundary(segments, overlap=3.0)
        # Should pick the latest: second sentence ends at 2.0
        assert boundary == 2.0


# ---------------------------------------------------------------------------
# Multi-chunk merge
# ---------------------------------------------------------------------------


class TestMultiChunkMerge:
    def test_two_chunks_basic(self):
        chunk0 = [
            _seg(0.0, 5.0, " Chunk zero."),
            _seg(5.0, 10.0, " End of chunk zero."),
        ]
        chunk1 = [
            _seg(0.0, 5.0, " Overlap text."),  # in overlap region
            _seg(5.0, 10.0, " Chunk one content."),
        ]
        result = merge_segments(
            [chunk0, chunk1], chunk_duration=10, overlap=5,
            timestamp_mode="none", paragraph_gap=100.0,
        )
        assert "Chunk zero." in result
        assert "End of chunk zero." in result
        assert "Chunk one content." in result
        # Overlap text should be skipped
        assert "Overlap text." not in result

    def test_empty_segments(self):
        result = merge_segments([], chunk_duration=0, overlap=0)
        assert result == ""

    def test_all_empty_chunks(self):
        result = merge_segments([[], []], chunk_duration=10, overlap=5)
        assert result == ""
