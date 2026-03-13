"""Speaker diarization for scribe-md (Phase 4.1).

Provides optional speaker identification using ``pyannote-audio``.
The dependency is heavy (PyTorch + pyannote pipeline) and is lazily
imported — users who never pass ``--diarize`` never need to install it.

On Apple Silicon, pyannote runs on CPU (MPS support is incomplete).
Expect roughly real-time processing (~30 min audio ≈ 15-30 min).
"""

from __future__ import annotations

from pathlib import Path


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""


def diarize_audio(
    audio_path: Path,
    *,
    hf_token: str = "",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """Run speaker diarization on an audio file.

    Returns a list of turns: ``[{"start": float, "end": float, "speaker": str}]``
    with speaker labels normalized to ``Speaker 1``, ``Speaker 2``, etc.

    Requires ``pyannote-audio`` and a HuggingFace token (the model is gated).

    Parameters
    ----------
    audio_path : Path
        Path to a WAV file (16 kHz mono recommended).
    hf_token : str
        HuggingFace API token.  Falls back to ``HF_TOKEN`` env var.
    num_speakers : int or None
        Exact number of speakers (if known).
    min_speakers, max_speakers : int or None
        Speaker count range hints.
    """
    import os

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise DiarizationError(
            "A HuggingFace token is required for speaker diarization.\n"
            "Provide it via --hf-token, config file, or HF_TOKEN env var.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "and accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyannote-audio is required for speaker diarization but is not installed.\n"
            "Install it with: pip install pyannote-audio"
        )

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    except Exception as e:
        raise DiarizationError(f"Failed to load diarization model: {e}")

    # Build kwargs for the pipeline
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        diarization = pipeline(str(audio_path), **kwargs)
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}")

    # Convert pyannote output to a list of turn dicts
    raw_turns: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    return _normalize_speaker_labels(raw_turns)


def _normalize_speaker_labels(turns: list[dict]) -> list[dict]:
    """Rename pyannote labels (``SPEAKER_00``) to ``Speaker 1``, etc.

    Labels are assigned in order of first appearance, so the first person
    to speak is always ``Speaker 1``.
    """
    label_map: dict[str, str] = {}
    counter = 0

    for turn in turns:
        raw = turn["speaker"]
        if raw not in label_map:
            counter += 1
            label_map[raw] = f"Speaker {counter}"

    return [
        {"start": t["start"], "end": t["end"], "speaker": label_map[t["speaker"]]}
        for t in turns
    ]


def assign_speakers(
    segments: list[dict],
    turns: list[dict],
    *,
    time_offset: float = 0.0,
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    Each segment is matched to the diarization turn with the greatest
    temporal overlap.  If no turn overlaps a segment, the speaker field
    is set to ``"Unknown"``.

    Parameters
    ----------
    segments : list[dict]
        Whisper segments with ``start``, ``end``, ``text`` keys.
        Times are relative to the chunk.
    turns : list[dict]
        Diarization turns with ``start``, ``end``, ``speaker`` keys.
        Times are on the global (full-audio) timeline.
    time_offset : float
        Offset to add to segment times to align with the global timeline.
        For chunk N with chunk_duration D and overlap O, this is
        ``N * D - O`` (same formula used in the merger).
    """
    result: list[dict] = []
    for seg in segments:
        global_start = seg["start"] + time_offset
        global_end = seg["end"] + time_offset

        best_speaker = "Unknown"
        best_overlap = 0.0

        for turn in turns:
            # Compute overlap between segment and turn
            overlap_start = max(global_start, turn["start"])
            overlap_end = min(global_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        result.append({
            **seg,
            "speaker": best_speaker,
        })

    return result
