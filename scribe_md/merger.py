"""Overlap-aware chunk merging for long-form transcription."""

from .utils import format_timestamp


def merge_segments(
    chunk_segments: list[list[dict]],
    chunk_duration: float,
    overlap: float,
    timestamps: bool = True,
) -> str:
    """Merge segments from multiple chunks into a single Markdown string.

    Each chunk's segments have timestamps relative to the chunk's audio.
    For chunk N>0, the audio includes `overlap` seconds from the previous
    chunk's tail. We skip those overlapping segments and offset the rest
    to produce a continuous timeline.
    """
    all_segments: list[dict] = []

    for idx, segments in enumerate(chunk_segments):
        offset = 0.0 if idx == 0 else idx * chunk_duration - overlap

        for seg in segments:
            # Skip segments in the overlap region of non-first chunks
            if idx > 0 and seg["start"] < overlap:
                continue
            all_segments.append({
                "start": offset + seg["start"],
                "end": offset + seg["end"],
                "text": seg["text"],
            })

    if timestamps:
        lines = [f"{format_timestamp(s['start'])} {s['text']}" for s in all_segments]
    else:
        lines = [s["text"] for s in all_segments]

    return "\n\n".join(lines) + "\n"
