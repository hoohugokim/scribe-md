"""Overlap-aware chunk merging for long-form transcription."""

import re

from .utils import format_timestamp

# Sentence-ending punctuation (including common Unicode variants)
_SENTENCE_END_RE = re.compile(r'[.!?。！？][\s"\'»）)]*$')


def _find_sentence_boundary(segments: list[dict], overlap: float) -> float:
    """Find the best split point in the overlap region, preferring sentence ends.

    Returns the timestamp at which to start accepting segments from the new
    chunk.  Falls back to the raw *overlap* value if no sentence boundary is
    found in the overlap region.
    """
    best = None
    for seg in segments:
        if seg["start"] >= overlap:
            break
        text = seg["text"].strip()
        if _SENTENCE_END_RE.search(text):
            # The segment *after* this one is a clean place to resume
            best = seg["end"]
    return best if best is not None else overlap


def merge_segments(
    chunk_segments: list[list[dict]],
    chunk_duration: float,
    overlap: float,
    timestamps: bool = True,
    *,
    timestamp_mode: str = "segment",
    paragraph_gap: float = 2.0,
) -> str:
    """Merge segments from multiple chunks into a single Markdown string.

    Each chunk's segments have timestamps relative to the chunk's audio.
    For chunk N>0, the audio includes *overlap* seconds from the previous
    chunk's tail.  We skip those overlapping segments and offset the rest
    to produce a continuous timeline.

    Parameters
    ----------
    chunk_segments : list[list[dict]]
        Per-chunk segment lists, each with ``start``, ``end``, ``text`` keys.
    chunk_duration : float
        Duration (seconds) of each audio chunk.
    overlap : float
        Overlap (seconds) between consecutive chunks.
    timestamps : bool
        Legacy flag. When *False*, equivalent to ``timestamp_mode="none"``.
    timestamp_mode : str
        Controls timestamp granularity:

        ``"segment"``
            Timestamp before every segment (default, matches original
            behaviour).
        ``"paragraph"``
            Timestamp only at the start of each paragraph.
        ``"minute"``
            Timestamp at the start of each new minute.
        ``"none"``
            No timestamps at all.
    paragraph_gap : float
        Minimum silence gap (seconds) between consecutive segments that
        triggers a paragraph break (double newline).  Default ``2.0``.
    """

    # ── 1. Flatten chunks into a single timeline ──────────────────────────
    all_segments: list[dict] = []

    for idx, segments in enumerate(chunk_segments):
        offset = 0.0 if idx == 0 else idx * chunk_duration - overlap

        if idx > 0 and overlap > 0:
            split_at = _find_sentence_boundary(segments, overlap)
        else:
            split_at = 0.0

        for seg in segments:
            if idx > 0 and seg["start"] < split_at:
                continue
            all_segments.append({
                "start": offset + seg["start"],
                "end": offset + seg["end"],
                "text": seg["text"],
            })

    if not all_segments:
        return ""

    # ── 2. Resolve effective timestamp mode ────────────────────────────────
    if not timestamps:
        timestamp_mode = "none"

    # ── 3. Group segments into paragraphs ──────────────────────────────────
    paragraphs: list[list[dict]] = []
    current_para: list[dict] = [all_segments[0]]

    for prev_seg, cur_seg in zip(all_segments, all_segments[1:]):
        gap = cur_seg["start"] - prev_seg["end"]
        if gap >= paragraph_gap:
            paragraphs.append(current_para)
            current_para = [cur_seg]
        else:
            current_para.append(cur_seg)
    paragraphs.append(current_para)

    # ── 4. Render paragraphs to Markdown ──────────────────────────────────
    rendered_paragraphs: list[str] = []
    last_emitted_minute = -1

    for para in paragraphs:
        if timestamp_mode == "segment":
            # Each segment on its own line with a timestamp (original style).
            lines = [
                f"{format_timestamp(s['start'])} {s['text']}" for s in para
            ]
            rendered_paragraphs.append("\n\n".join(lines))

        elif timestamp_mode == "paragraph":
            # Timestamp only at the beginning of the paragraph; segment texts
            # are joined as flowing prose.
            ts = format_timestamp(para[0]["start"])
            body = " ".join(s["text"].strip() for s in para)
            rendered_paragraphs.append(f"{ts} {body}")

        elif timestamp_mode == "minute":
            # Timestamp whenever a new minute begins; otherwise flowing text.
            parts: list[str] = []
            for seg in para:
                seg_minute = int(seg["start"] // 60)
                text = seg["text"].strip()
                if seg_minute != last_emitted_minute:
                    parts.append(f"{format_timestamp(seg['start'])} {text}")
                    last_emitted_minute = seg_minute
                else:
                    parts.append(text)
            rendered_paragraphs.append(" ".join(parts))

        else:
            # "none" — no timestamps, pure flowing text.
            body = " ".join(s["text"].strip() for s in para)
            rendered_paragraphs.append(body)

    return "\n\n".join(rendered_paragraphs) + "\n"
