"""Swift audio capture binary management."""

import subprocess
from pathlib import Path

from .utils import log

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = PROJECT_ROOT / "capture"
CAPTURE_BIN = CAPTURE_DIR / ".build" / "release" / "appaudio-capture"


def ensure_capture_binary() -> Path:
    """Build the Swift capture binary if it doesn't exist. Returns the binary path."""
    if CAPTURE_BIN.exists():
        return CAPTURE_BIN

    # Check for Swift toolchain
    try:
        subprocess.run(
            ["swift", "--version"],
            capture_output=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "Swift toolchain not found. Install Xcode Command Line Tools:\n"
            "  xcode-select --install"
        )

    log("Building audio capture tool (first run only)...")
    subprocess.run(
        ["swift", "build", "-c", "release"],
        cwd=str(CAPTURE_DIR),
        check=True,
    )

    if not CAPTURE_BIN.exists():
        raise RuntimeError(f"Build succeeded but binary not found at {CAPTURE_BIN}")

    return CAPTURE_BIN


def list_apps() -> list[dict]:
    """List running apps visible to ScreenCaptureKit.

    Returns a list of dicts with 'name' and 'bundle_id' keys, sorted by name.
    """
    binary = ensure_capture_binary()
    result = subprocess.run(
        [str(binary), "--list-apps"],
        capture_output=True, text=True, check=True,
    )
    apps = []
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            name, bundle_id = line.split("\t", 1)
            apps.append({"name": name, "bundle_id": bundle_id})
    return apps


def run_capture(
    output_path: Path,
    duration: float | None = None,
    chunk_seconds: float = 0,
    overlap_seconds: float = 5,
    app: str | None = None,
) -> subprocess.Popen:
    """Launch the capture binary, returning the Popen handle.

    In chunked mode, chunk file paths are emitted to stdout (one per line).
    Status messages go to stderr.

    If `app` is specified, captures audio only from that app (name substring match).
    """
    binary = ensure_capture_binary()
    args = [str(binary), "--output", str(output_path)]

    if duration is not None:
        args.extend(["--duration", str(duration)])
    if chunk_seconds > 0:
        args.extend(["--chunk-seconds", str(chunk_seconds)])
        args.extend(["--overlap-seconds", str(overlap_seconds)])
    if app is not None:
        args.extend(["--app", app])

    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=None)
