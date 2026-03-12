"""Shared utilities for scribe-md."""

import re
import sys


def format_timestamp(seconds: float) -> str:
    """Format seconds as [HH:MM:SS]."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def sanitize_filename(title: str) -> str:
    """Sanitize a video title for use as a filename."""
    # Remove characters invalid in filenames
    clean = re.sub(r'[<>:"/\\|?*]', "", title)
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    # Truncate to reasonable length
    if len(clean) > 200:
        clean = clean[:200].rsplit(" ", 1)[0]
    return clean or "untitled"


def log(msg: str) -> None:
    """Print a message to stderr."""
    print(msg, file=sys.stderr)
