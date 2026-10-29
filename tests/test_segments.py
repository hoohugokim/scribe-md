"""Tests for ASR segment normalization helpers."""

from scribe_md.backends.segments import segments_from_text, segments_from_timed


def test_segments_from_text_wraps_duration():
    result = segments_from_text("hello world", duration=12.5)
    assert result == {
        "segments": [
            {
                "start": 0.0,
                "end": 12.5,
                "text": "hello world",
                "no_speech_prob": 0.0,
            }
        ]
    }


def test_segments_from_text_empty():
    assert segments_from_text("  ") == {"segments": []}


def test_segments_from_timed_drops_empty_and_orders_fields():
    result = segments_from_timed(
        [
            {"start": 0.0, "end": 1.0, "text": " hi "},
            {"start": 1.0, "end": 2.0, "text": ""},
            {"start": 2.0, "end": 1.5, "text": "backwards"},
        ]
    )
    assert result["segments"] == [
        {"start": 0.0, "end": 1.0, "text": "hi", "no_speech_prob": 0.0},
        {"start": 2.0, "end": 2.0, "text": "backwards", "no_speech_prob": 0.0},
    ]
