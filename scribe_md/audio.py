"""FFmpeg helpers for audio conversion and splitting."""

import subprocess
from pathlib import Path

from .utils import log


def convert_to_16k_mono(input_path: Path, output_path: Path) -> Path:
    """Convert any audio file to 16kHz mono WAV for Whisper."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(output_path), "-loglevel", "error",
        ],
        check=True,
    )
    return output_path


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def is_silent(audio_path: Path, threshold_db: float = -50) -> bool:
    """Check if audio is effectively silent using ffmpeg volumedetect.

    Returns True if mean volume is below threshold_db (default -50 dBFS).
    Silent audio causes Whisper to hallucinate (e.g. "자막제공자").
    """
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(audio_path),
            "-af", "volumedetect",
            "-f", "null", "/dev/null",
        ],
        capture_output=True, text=True,
    )
    for line in result.stderr.split("\n"):
        if "mean_volume:" in line:
            try:
                vol_str = line.split("mean_volume:")[1].strip().split()[0]
                mean_vol = float(vol_str)
                return mean_vol < threshold_db
            except (IndexError, ValueError):
                pass
    # If we can't determine volume, assume not silent
    return False


def split_audio(
    input_path: Path,
    output_dir: Path,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[Path]:
    """Split audio into overlapping chunks for long-form transcription.

    Each chunk (except the first) starts `overlap_seconds` before its nominal
    boundary so the merge step can deduplicate the overlap region.
    """
    duration = get_duration(input_path)
    chunks: list[Path] = []
    start = 0.0
    idx = 0

    while start < duration:
        chunk_path = output_dir / f"chunk_{idx:03d}.wav"

        # First chunk starts at 0; subsequent chunks start earlier by overlap
        actual_start = max(0, start - overlap_seconds) if idx > 0 else 0
        actual_duration = chunk_seconds + (overlap_seconds if idx > 0 else 0)

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ss", str(actual_start),
                "-t", str(actual_duration),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                str(chunk_path), "-loglevel", "error",
            ],
            check=True,
        )
        chunks.append(chunk_path)
        log(f"  Split chunk {idx}: {actual_start:.1f}s - {actual_start + actual_duration:.1f}s")

        start += chunk_seconds
        idx += 1

    return chunks
