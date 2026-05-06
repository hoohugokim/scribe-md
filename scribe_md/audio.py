"""FFmpeg helpers for audio conversion and splitting."""

import shutil
import subprocess
from pathlib import Path

from .utils import log


class AudioConversionError(RuntimeError):
    """Raised when ffmpeg conversion fails."""


class DiskFullError(OSError):
    """Raised when a write fails due to insufficient disk space."""


def _check_disk_space(path: Path, min_bytes: int = 10 * 1024 * 1024) -> None:
    """Raise DiskFullError if the volume containing *path* has less than *min_bytes* free."""
    try:
        usage = shutil.disk_usage(path.parent if path.parent.exists() else path)
        if usage.free < min_bytes:
            free_mb = usage.free / (1024 * 1024)
            raise DiskFullError(
                f"Insufficient disk space: {free_mb:.1f} MB free. "
                "Free up space and try again."
            )
    except OSError:
        pass  # If we can't check, proceed and let the write fail naturally


def convert_to_16k_mono(input_path: Path, output_path: Path) -> Path:
    """Convert any audio file to 16kHz mono WAV for Whisper."""
    _check_disk_space(output_path)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                str(output_path), "-loglevel", "error",
            ],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise AudioConversionError(
            "ffmpeg not found. Install it with: brew install ffmpeg"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise AudioConversionError(
            f"ffmpeg conversion failed (exit {result.returncode}): {stderr}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioConversionError(
            f"ffmpeg produced no output for {input_path.name}. "
            "The input file may be corrupt or in an unsupported format."
        )

    return output_path


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise AudioConversionError(
            "ffprobe not found. Install it with: brew install ffmpeg"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise AudioConversionError(
            f"ffprobe failed (exit {result.returncode}): {stderr}"
        )

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise AudioConversionError(
            f"ffprobe returned invalid duration for {audio_path.name}: "
            f"{result.stdout.strip()!r}"
        )


def is_silent(audio_path: Path, threshold_db: float = -50) -> bool:
    """Check if audio is effectively silent using ffmpeg volumedetect.

    Returns True if mean volume is below threshold_db (default -50 dBFS).
    Silent audio causes Whisper to hallucinate (e.g. "자막제공자").
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(audio_path),
                "-af", "volumedetect",
                "-f", "null", "/dev/null",
            ],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise AudioConversionError(
            "ffmpeg not found. Install it with: brew install ffmpeg"
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

        _check_disk_space(chunk_path)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-ss", str(actual_start),
                    "-t", str(actual_duration),
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                    str(chunk_path), "-loglevel", "error",
                ],
                capture_output=True, text=True,
            )
        except FileNotFoundError:
            raise AudioConversionError(
                "ffmpeg not found. Install it with: brew install ffmpeg"
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise AudioConversionError(
                f"ffmpeg split failed on chunk {idx} (exit {result.returncode}): {stderr}"
            )

        chunks.append(chunk_path)
        log(f"  Split chunk {idx}: {actual_start:.1f}s - {actual_start + actual_duration:.1f}s")

        start += chunk_seconds
        idx += 1

    return chunks
