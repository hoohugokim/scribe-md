"""CUDA device discovery and ``--gpus`` spec resolution.

Pure parsing (``resolve_gpu_spec``) is separated from the ``nvidia-smi``
subprocess (``discover_cuda_devices``) so the grammar is unit-testable
without hardware.
"""

from __future__ import annotations

import re
import subprocess


class GpuSpecError(ValueError):
    """Raised when a --gpus value cannot be satisfied by available devices."""


_GPU_LINE = re.compile(r"^GPU (\d+):", re.MULTILINE)


def discover_cuda_devices() -> list[int]:
    """Return CUDA device indices from ``nvidia-smi -L`` (empty if unavailable)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [int(m) for m in _GPU_LINE.findall(result.stdout)]


def resolve_gpu_spec(spec: str | None, available: list[int]) -> list[int]:
    """Map a --gpus value onto *available* device ids.

    Grammar: ``None``/``"1"`` -> first device only; ``"auto"`` -> all;
    integer ``"N"`` -> first N; list ``"0,1"`` -> those explicit ids.
    Raises ``GpuSpecError`` if the request cannot be met.
    """
    spec = (spec or "").strip().lower()

    # Sequential default needs no real device list (single-device / non-CUDA
    # callers handle the empty case themselves).
    if spec in ("", "1"):
        return available[:1] if available else [0]

    if not available:
        raise GpuSpecError(
            "--gpus requested but no CUDA devices were found (nvidia-smi). "
            "Use the cuda pixi env on an NVIDIA machine, or drop --gpus."
        )

    if spec == "auto":
        return list(available)

    if "," in spec:
        ids = [int(x) for x in spec.split(",") if x.strip()]
        missing = [i for i in ids if i not in available]
        if missing:
            raise GpuSpecError(
                f"--gpus {spec!r}: device(s) {missing} not in available {available}."
            )
        return ids

    if spec.isdigit():
        n = int(spec)
        if n < 1:
            raise GpuSpecError("--gpus must be >= 1.")
        if n > len(available):
            raise GpuSpecError(
                f"--gpus {n} requested but only {len(available)} device(s) "
                f"available: {available}."
            )
        return available[:n]

    raise GpuSpecError(
        f"--gpus {spec!r} not understood; use 'auto', an integer, or a list "
        "like '0,1'."
    )
