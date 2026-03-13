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
            flat = {
                "start": offset + seg["start"],
                "end": offset + seg["end"],
                "text": seg["text"],
            }
            if "speaker" in seg:
                flat["speaker"] = seg["speaker"]
            all_segments.append(flat)

    if not all_segments:
        return ""

    # ── 2. Resolve effective timestamp mode ────────────────────────────────
    if not timestamps:
        timestamp_mode = "none"

    # ── 3. Group segments into paragraphs ──────────────────────────────────
    # A paragraph break is triggered by either:
    #   - a silence gap >= paragraph_gap, or
    #   - a speaker change (when diarization labels are present).
    paragraphs: list[list[dict]] = []
    current_para: list[dict] = [all_segments[0]]

    for prev_seg, cur_seg in zip(all_segments, all_segments[1:]):
        gap = cur_seg["start"] - prev_seg["end"]
        speaker_changed = (
            "speaker" in cur_seg
            and "speaker" in prev_seg
            and cur_seg["speaker"] != prev_seg["speaker"]
        )
        if gap >= paragraph_gap or speaker_changed:
            paragraphs.append(current_para)
            current_para = [cur_seg]
        else:
            current_para.append(cur_seg)
    paragraphs.append(current_para)

    # ── 4. Render paragraphs to Markdown ──────────────────────────────────
    rendered_paragraphs: list[str] = []
    last_emitted_minute = -1
    last_speaker: str | None = None

    for para in paragraphs:
        # Determine if this paragraph starts with a new speaker
        speaker_label = ""
        para_speaker = para[0].get("speaker")
        if para_speaker and para_speaker != last_speaker:
            speaker_label = f"**{para_speaker}:** "
            last_speaker = para_speaker

        if timestamp_mode == "segment":
            # Each segment on its own line with a timestamp (original style).
            lines: list[str] = []
            for i, s in enumerate(para):
                prefix = speaker_label if i == 0 else ""
                lines.append(
                    f"{format_timestamp(s['start'])} {prefix}{s['text'].strip()}"
                )
            rendered_paragraphs.append("\n\n".join(lines))

        elif timestamp_mode == "paragraph":
            # Timestamp only at the beginning of the paragraph; segment texts
            # are joined as flowing prose.
            ts = format_timestamp(para[0]["start"])
            body = " ".join(s["text"].strip() for s in para)
            rendered_paragraphs.append(f"{ts} {speaker_label}{body}")

        elif timestamp_mode == "minute":
            # Timestamp whenever a new minute begins; otherwise flowing text.
            parts: list[str] = []
            for i, seg in enumerate(para):
                seg_minute = int(seg["start"] // 60)
                text = seg["text"].strip()
                prefix = speaker_label if i == 0 else ""
                if seg_minute != last_emitted_minute:
                    parts.append(
                        f"{format_timestamp(seg['start'])} {prefix}{text}"
                    )
                    last_emitted_minute = seg_minute
                else:
                    parts.append(f"{prefix}{text}")
            rendered_paragraphs.append(" ".join(parts))

        else:
            # "none" — no timestamps, pure flowing text.
            body = " ".join(s["text"].strip() for s in para)
            rendered_paragraphs.append(f"{speaker_label}{body}")

    return "\n\n".join(rendered_paragraphs) + "\n"
