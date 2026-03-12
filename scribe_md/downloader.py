"""YouTube audio download via yt-dlp."""

import json
import subprocess
from pathlib import Path

from .utils import log, sanitize_filename


def get_video_info(url: str) -> dict:
    """Get video metadata without downloading."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def is_playlist(url: str) -> bool:
    """Check if URL is a playlist (has multiple entries)."""
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--dump-json", url],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.strip().split("\n") if l]
    return len(lines) > 1


def get_playlist_entries(url: str) -> list[dict]:
    """Get metadata for all videos in a playlist."""
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--dump-json", url],
        capture_output=True, text=True, check=True,
    )
    return [json.loads(line) for line in result.stdout.strip().split("\n") if line]


def download_audio(url: str, output_dir: Path) -> tuple[Path, str]:
    """Download audio from a URL, returning (audio_path, title).

    Downloads the best audio, then converts to WAV via ffmpeg post-processor.
    The caller is responsible for converting to 16kHz mono if needed.
    """
    # Get title first for naming
    info = get_video_info(url)
    title = info.get("title", "untitled")
    safe_name = sanitize_filename(title)
    output_template = str(output_dir / safe_name)

    log(f"Downloading: {title}")
    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "-o", f"{output_template}.%(ext)s",
            "--no-playlist",
            url,
        ],
        check=True,
    )

    # yt-dlp outputs to {template}.wav
    audio_path = Path(f"{output_template}.wav")
    if not audio_path.exists():
        # Fallback: look for any audio file with that stem
        candidates = list(output_dir.glob(f"{safe_name}.*"))
        if candidates:
            audio_path = candidates[0]
        else:
            raise FileNotFoundError(f"Downloaded audio not found at {audio_path}")

    return audio_path, title
