"""OS detection and platform-aware user hints.

Named ``platform_support`` rather than ``platform`` to avoid shadowing the
standard-library module.
"""

import sys


def is_macos() -> bool:
    """True on macOS."""
    return sys.platform == "darwin"


def is_linux() -> bool:
    """True on any Linux distribution."""
    return sys.platform.startswith("linux")


def ffmpeg_install_hint() -> str:
    """Return a platform-appropriate ffmpeg install instruction."""
    if is_linux():
        return "Install it with: sudo apt install ffmpeg"
    return "Install it with: brew install ffmpeg"
