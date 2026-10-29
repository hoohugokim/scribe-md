"""Normalize heterogeneous ASR outputs into mlx-whisper-shaped segments."""

from __future__ import annotations


def segments_from_text(
    text: str,
    *,
    duration: float = 0.0,
    no_speech_prob: float = 0.0,
) -> dict:
    """Wrap plain transcript text as a single segment covering *duration*."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {"segments": []}
    end = max(float(duration), 0.0)
    return {
        "segments": [
            {
                "start": 0.0,
                "end": end,
                "text": cleaned,
                "no_speech_prob": no_speech_prob,
            }
        ]
    }


def segments_from_timed(
    items: list[dict],
    *,
    text_key: str = "text",
    start_key: str = "start",
    end_key: str = "end",
    no_speech_prob: float = 0.0,
) -> dict:
    """Map timed dicts into the standard segment list, dropping empty text."""
    segments = []
    for item in items:
        text = str(item.get(text_key, "")).strip()
        if not text:
            continue
        start = float(item.get(start_key, 0.0) or 0.0)
        end = float(item.get(end_key, start) or start)
        if end < start:
            end = start
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "no_speech_prob": float(item.get("no_speech_prob", no_speech_prob)),
            }
        )
    return {"segments": segments}
