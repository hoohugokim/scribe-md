"""YouTube audio download via yt-dlp."""

import json
import subprocess
from pathlib import Path

from .utils import log, sanitize_filename


class DownloadError(RuntimeError):
    """Raised when yt-dlp cannot fetch metadata or audio."""


def _run_ytdlp(args: list[str], action: str, *, capture_output: bool = True):
    """Run yt-dlp and convert expected subprocess failures to DownloadError."""
    try:
        return subprocess.run(
            args,
            capture_output=capture_output,
            text=True,
            check=True,
        )
    except FileNotFoundError as e:
        raise DownloadError("yt-dlp not found. Install it and try again.") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        detail = f": {stderr}" if stderr else f" (exit {e.returncode})"
        raise DownloadError(f"yt-dlp failed while {action}{detail}") from e


def _loads_json(text: str, action: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise DownloadError(f"yt-dlp returned invalid JSON while {action}: {e}") from e


def get_video_info(url: str) -> dict:
    """Get video metadata without downloading."""
    result = _run_ytdlp(
        ["yt-dlp", "--dump-json", "--no-download", url],
        "reading video metadata",
    )
    return _loads_json(result.stdout, "reading video metadata")


def is_playlist(url: str) -> bool:
    """Check if URL is a playlist (has multiple entries)."""
    result = _run_ytdlp(
        ["yt-dlp", "--flat-playlist", "--dump-json", url],
        "checking playlist entries",
    )
    lines = [l for l in result.stdout.strip().split("\n") if l]
    return len(lines) > 1


def get_playlist_entries(url: str) -> list[dict]:
    """Get metadata for all videos in a playlist."""
    result = _run_ytdlp(
        ["yt-dlp", "--flat-playlist", "--dump-json", url],
        "reading playlist entries",
    )
    return [
        _loads_json(line, "reading playlist entries")
        for line in result.stdout.strip().split("\n")
        if line
    ]


def download_audio(
    url: str, output_dir: Path, title: str | None = None
) -> tuple[Path, str]:
    """Download audio from a URL, returning (audio_path, title).

    Downloads the best audio, then converts to WAV via ffmpeg post-processor.
    The caller is responsible for converting to 16kHz mono if needed.

    If *title* is already known (e.g. from a playlist entry), it is used for
    naming directly, avoiding a redundant ``--dump-json`` metadata request.
    """
    # Only hit the network for metadata when the title isn't already known.
    if not title:
        title = get_video_info(url).get("title", "untitled")
    safe_name = sanitize_filename(title)
    output_template = str(output_dir / safe_name)

    log(f"Downloading: {title}")
    _run_ytdlp(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "-o", f"{output_template}.%(ext)s",
            "--no-playlist",
            url,
        ],
        "downloading audio",
        capture_output=False,
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
