"""Swift audio capture binary management."""

import subprocess
from pathlib import Path

from .utils import log

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = PROJECT_ROOT / "capture"
CAPTURE_BIN = CAPTURE_DIR / ".build" / "release" / "appaudio-capture"

# Screen Recording permission hint
_PERMISSION_MSG = (
    "Screen Recording permission is required.\n"
    "Grant access in: System Settings > Privacy & Security > Screen Recording\n"
    "Then restart your terminal and try again."
)


class CaptureError(RuntimeError):
    """Raised when the audio capture binary fails."""


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
        raise CaptureError(
            "Swift toolchain not found. Install Xcode Command Line Tools:\n"
            "  xcode-select --install"
        )

    log("Building audio capture tool (first run only)...")
    try:
        result = subprocess.run(
            ["swift", "build", "-c", "release"],
            cwd=str(CAPTURE_DIR),
            capture_output=True, text=True,
        )
    except OSError as e:
        raise CaptureError(f"Failed to build capture tool: {e}")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise CaptureError(f"Swift build failed (exit {result.returncode}): {stderr}")

    if not CAPTURE_BIN.exists():
        raise CaptureError(f"Build succeeded but binary not found at {CAPTURE_BIN}")

    return CAPTURE_BIN


def _check_capture_permission(binary: Path, timeout: float = 10.0) -> None:
    """Run a quick --list-apps probe to verify Screen Recording permission.

    ScreenCaptureKit may silently hang or return empty results when the
    permission has not been granted. We detect that and give the user a
    clear message.
    """
    try:
        result = subprocess.run(
            [str(binary), "--list-apps"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise CaptureError(
            f"Screen Recording permission check timed out after {timeout:.0f}s.\n"
            + _PERMISSION_MSG
        )
    except OSError as e:
        raise CaptureError(f"Failed to run capture binary: {e}")

    # The Swift binary writes "Error:" to stderr on permission failure
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "error" in stderr.lower():
            raise CaptureError(
                f"Capture binary failed: {stderr}\n" + _PERMISSION_MSG
            )
        raise CaptureError(f"Capture binary failed (exit {result.returncode}): {stderr}")


def list_apps() -> list[dict]:
    """List running apps visible to ScreenCaptureKit.

    Returns a list of dicts with 'name' and 'bundle_id' keys, sorted by name.
    """
    binary = ensure_capture_binary()
    try:
        result = subprocess.run(
            [str(binary), "--list-apps"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise CaptureError(
            "Listing apps timed out. " + _PERMISSION_MSG
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "error" in stderr.lower():
            raise CaptureError(
                f"Cannot list apps: {stderr}\n" + _PERMISSION_MSG
            )
        raise CaptureError(f"list-apps failed (exit {result.returncode}): {stderr}")

    apps = []
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            name, bundle_id = line.split("\t", 1)
            apps.append({"name": name, "bundle_id": bundle_id})
    return apps


def terminate_capture(proc: subprocess.Popen) -> None:
    """Stop and reap a capture subprocess; safe if it already exited.

    The live pipelines launch the Swift recorder concurrently with Python.
    If the pipeline fails mid-run (e.g. DiskFullError) the recorder must be
    reaped before the temp directory it writes into is cleaned up — otherwise
    it keeps running orphaned and races the cleanup. Called from a ``finally``.
    """
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass


def run_capture(
    output_path: Path,
    duration: float | None = None,
    chunk_seconds: float = 0,
    overlap_seconds: float = 5,
    app: str | list[str] | None = None,
) -> subprocess.Popen:
    """Launch the capture binary, returning the Popen handle.

    In chunked mode, chunk file paths are emitted to stdout (one per line).
    Status messages go to stderr.

    If `app` is specified (string or list of strings), captures audio only from
    those app(s) (name substring match).

    Raises CaptureError if Screen Recording permission is not granted.
    """
    binary = ensure_capture_binary()

    # Verify Screen Recording permission before starting a long capture
    _check_capture_permission(binary)

    args = [str(binary), "--output", str(output_path)]

    if duration is not None:
        args.extend(["--duration", str(duration)])
    if chunk_seconds > 0:
        args.extend(["--chunk-seconds", str(chunk_seconds)])
        args.extend(["--overlap-seconds", str(overlap_seconds)])
    if app is not None:
        if isinstance(app, list):
            for a in app:
                args.extend(["--app", a])
        else:
            args.extend(["--app", app])

    try:
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=None)
    except OSError as e:
        raise CaptureError(f"Failed to start capture: {e}")
