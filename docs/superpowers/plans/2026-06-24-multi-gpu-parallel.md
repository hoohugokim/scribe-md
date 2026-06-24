# Multi-GPU Parallel Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one `scribe-md` invocation accept many inputs (files/URLs) and transcribe their chunks concurrently across multiple NVIDIA GPUs, replacing per-GPU batch scripts.

**Architecture:** A new `gpu.py` discovers CUDA devices and parses the `--gpus` grammar. A new `scheduler.py` runs a bounded producer/consumer: each chunk is an isolated `whisper-cli` subprocess pinned to a GPU via `CUDA_VISIBLE_DEVICES`, pulled from a device pool by a thread pool sized to the GPU count. The CLI gains variadic inputs + `--from-file` + `--gpus`; results are reassembled per source, in chunk order, and written one `.md` per source. Multi-input works on every platform; parallelism engages only for the CUDA whisper.cpp backend (else sequential fallback).

**Tech Stack:** Python ≥3.12, Typer, `concurrent.futures.ThreadPoolExecutor`, `queue.Queue`, whisper.cpp subprocess backend, pytest (fully mocked).

## Global Constraints

- **Python ≥ 3.12** (`pyproject.toml`), pixi-managed; stdlib only (no new deps).
- **Multi-GPU is CUDA-only for v1.** Vulkan/MLX → sequential fallback with a one-line notice. Deferred backends are documented in README, not implemented.
- **Tests are fully hermetic:** no real GPU, network, ffmpeg, or whisper binary — mock `nvidia-smi`, `subprocess.run`, and the backend. Mirror existing test style (`tests/test_*.py`, `CliRunner`, `monkeypatch`).
- **All human output via `console` (`Console(stderr=True)`) or `utils.log`.** Errors collapse to `typer.Exit(1)` in command handlers; library code raises `TranscriptionError`.
- Follow existing patterns; DRY, YAGNI, TDD; commit after each task.
- Branch: `feat/multi-gpu-parallel` (already based on main, which includes the chunk-failure fix).

---

### Task 1: GPU discovery & `--gpus` grammar (`scribe_md/gpu.py`)

**Files:**
- Create: `scribe_md/gpu.py`
- Test: `tests/test_gpu.py`

**Interfaces:**
- Produces: `discover_cuda_devices() -> list[int]`; `resolve_gpu_spec(spec: str | None, available: list[int]) -> list[int]` (raises `GpuSpecError`); `class GpuSpecError(ValueError)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gpu.py
import pytest
from scribe_md import gpu


def test_resolve_none_or_one_means_single_device():
    assert gpu.resolve_gpu_spec(None, [0, 1, 2]) == [0]
    assert gpu.resolve_gpu_spec("1", [0, 1, 2]) == [0]


def test_resolve_auto_uses_all_available():
    assert gpu.resolve_gpu_spec("auto", [0, 1, 2]) == [0, 1, 2]


def test_resolve_integer_takes_first_n():
    assert gpu.resolve_gpu_spec("2", [0, 1, 2]) == [0, 1]


def test_resolve_explicit_list_validates_membership():
    assert gpu.resolve_gpu_spec("0,2", [0, 1, 2]) == [0, 2]
    with pytest.raises(gpu.GpuSpecError):
        gpu.resolve_gpu_spec("0,5", [0, 1, 2])


def test_resolve_more_than_available_is_clamped_with_error():
    with pytest.raises(gpu.GpuSpecError):
        gpu.resolve_gpu_spec("4", [0, 1])


def test_resolve_no_devices_available_raises():
    with pytest.raises(gpu.GpuSpecError):
        gpu.resolve_gpu_spec("auto", [])


def test_discover_parses_nvidia_smi(monkeypatch):
    sample = "GPU 0: NVIDIA RTX 3090 (UUID: GPU-aaa)\nGPU 1: NVIDIA RTX 3090 (UUID: GPU-bbb)\n"

    class R:
        returncode = 0
        stdout = sample

    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: R())
    assert gpu.discover_cuda_devices() == [0, 1]


def test_discover_returns_empty_when_nvidia_smi_absent(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    assert gpu.discover_cuda_devices() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_gpu.py -q`
Expected: FAIL — `ModuleNotFoundError: scribe_md.gpu`.

- [ ] **Step 3: Implement `scribe_md/gpu.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_gpu.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add scribe_md/gpu.py tests/test_gpu.py
git commit -m "feat(gpu): CUDA device discovery and --gpus spec resolution"
```

---

### Task 2: Backend `device` passthrough

**Files:**
- Modify: `scribe_md/backends/base.py` (protocol signature)
- Modify: `scribe_md/backends/mlx.py` (accept + ignore `device`)
- Modify: `scribe_md/backends/whispercpp.py:274-295` (`transcribe` sets `CUDA_VISIBLE_DEVICES`)
- Modify: `scribe_md/transcriber.py:29-61` (`transcribe_audio` passthrough)
- Test: `tests/test_whispercpp.py` (new test), `tests/test_backends.py` (new test)

**Interfaces:**
- Produces: `Backend.transcribe(audio_path, *, model, language, device: str | None = None)`; `transcriber.transcribe_audio(audio_path, model=..., language=..., device=None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_whispercpp.py  (add)
from pathlib import Path
from scribe_md.backends import whispercpp


def test_transcribe_pins_cuda_visible_devices(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, capture_output, text, env=None):
        captured["env"] = env

        class R:
            returncode = 0
            stderr = ""
        # write the expected JSON so transcribe() succeeds
        out_prefix = Path(cmd[cmd.index("-of") + 1])
        out_prefix.with_suffix(".json").write_text('{"transcription": []}')
        return R()

    monkeypatch.setattr(whispercpp, "ensure_whisper_binary", lambda: Path("/bin/whisper-cli"))
    monkeypatch.setattr(whispercpp, "_ensure_model_file", lambda m: Path("/m/ggml-tiny.bin"))
    monkeypatch.setattr(whispercpp.subprocess, "run", fake_run)

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00" * 100)
    whispercpp.WhisperCppBackend().transcribe(wav, model="tiny", language="ko", device="1")

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_transcribe_without_device_leaves_env_default(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, capture_output, text, env=None):
        captured["env"] = env

        class R:
            returncode = 0
            stderr = ""
        out_prefix = Path(cmd[cmd.index("-of") + 1])
        out_prefix.with_suffix(".json").write_text('{"transcription": []}')
        return R()

    monkeypatch.setattr(whispercpp, "ensure_whisper_binary", lambda: Path("/bin/whisper-cli"))
    monkeypatch.setattr(whispercpp, "_ensure_model_file", lambda m: Path("/m/ggml-tiny.bin"))
    monkeypatch.setattr(whispercpp.subprocess, "run", fake_run)

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00" * 100)
    whispercpp.WhisperCppBackend().transcribe(wav, model="tiny", language="ko")

    assert captured["env"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_whispercpp.py -k cuda_visible -q`
Expected: FAIL — `transcribe() got an unexpected keyword argument 'device'`.

- [ ] **Step 3: Implement the passthrough**

`scribe_md/backends/base.py` — update the protocol method signature:

```python
    def transcribe(
        self, audio_path: Path, *, model: str, language: str | None,
        device: str | None = None,
    ) -> dict:
        """Transcribe a 16 kHz mono WAV, returning a result dict.

        ``device`` optionally pins a specific accelerator (e.g. a CUDA device
        index); backends that cannot target a device ignore it.
        """
        ...
```

`scribe_md/backends/mlx.py` — accept and ignore:

```python
    def transcribe(
        self, audio_path: Path, *, model: str, language: str | None,
        device: str | None = None,
    ) -> dict:
        # device is ignored: Apple Silicon is a single unified-memory device.
        import mlx_whisper

        kwargs = {"path_or_hf_repo": self.resolve_model(model)}
        if language:
            kwargs["language"] = language
        return mlx_whisper.transcribe(str(audio_path), **kwargs)
```

`scribe_md/backends/whispercpp.py` — change `transcribe` (lines 274-295) to set env:

```python
    def transcribe(
        self, audio_path: Path, *, model: str, language: str | None,
        device: str | None = None,
    ) -> dict:
        binary = ensure_whisper_binary()
        model_path = _ensure_model_file(model)
        env = None
        if device is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
        with tempfile.TemporaryDirectory() as td:
            out_prefix = Path(td) / "out"
            cmd = _build_command(binary, model_path, audio_path, out_prefix, language)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            except OSError as e:
                raise WhisperCppError(f"Failed to run whisper-cli: {e}")
            if result.returncode != 0:
                raise WhisperCppError(
                    f"whisper.cpp failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            json_path = out_prefix.with_suffix(".json")
            if not json_path.exists():
                raise WhisperCppError(
                    f"whisper.cpp produced no JSON output at {json_path}."
                )
            data = json.loads(json_path.read_text(encoding="utf-8"))
        return parse_whispercpp_json(data)
```

`scribe_md/transcriber.py` — add `device` to `transcribe_audio` and pass it through:

```python
def transcribe_audio(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str | None = None,
) -> dict:
    """Validate audio_path and transcribe it via the active backend."""
    # ... existing validation unchanged ...
    from .backends import get_backend

    backend = get_backend()
    log(f"Transcribing {audio_path.name} via {backend.describe()}...")
    try:
        return backend.transcribe(
            audio_path, model=model, language=language, device=device
        )
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(
            f"Transcription failed for {audio_path.name}: {e}"
        ) from e
```

- [ ] **Step 4: Run to verify pass + no regressions**

Run: `pixi run pytest tests/test_whispercpp.py tests/test_backends.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scribe_md/backends/base.py scribe_md/backends/mlx.py \
        scribe_md/backends/whispercpp.py scribe_md/transcriber.py \
        tests/test_whispercpp.py
git commit -m "feat(backends): optional device pinning via CUDA_VISIBLE_DEVICES"
```

---

### Task 3: Shared `transcribe_chunk` helper

**Files:**
- Create: `scribe_md/scheduler.py` (helper only for now)
- Modify: `scribe_md/cli.py:486-498` (`_transcribe_chunk` delegates to scheduler)
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Produces: `scheduler.transcribe_chunk(chunk_path: Path, model: str, language: str | None, device: str | None = None) -> list[dict]`.
- Consumes: `transcriber.transcribe_audio(..., device=...)` (Task 2), `audio.is_silent`, `transcriber.extract_segments`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
from pathlib import Path
from scribe_md import scheduler


def test_transcribe_chunk_returns_empty_for_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler.audio, "is_silent", lambda p: True)
    assert scheduler.transcribe_chunk(tmp_path / "c.wav", "tiny", "ko") == []


def test_transcribe_chunk_passes_device_through(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(scheduler.audio, "is_silent", lambda p: False)

    def fake_transcribe(path, *, model, language, device=None):
        seen["device"] = device
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "no_speech_prob": 0.0}]}

    monkeypatch.setattr(scheduler.transcriber, "transcribe_audio", fake_transcribe)
    out = scheduler.transcribe_chunk(tmp_path / "c.wav", "tiny", "ko", device="2")
    assert seen["device"] == "2"
    assert out == [{"start": 0.0, "end": 1.0, "text": "hi"}]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_scheduler.py -q`
Expected: FAIL — `ModuleNotFoundError: scribe_md.scheduler`.

- [ ] **Step 3: Create `scribe_md/scheduler.py` with the helper**

```python
"""Multi-GPU parallel transcription scheduler.

Owns concurrency, GPU assignment, per-source ordering, and bounded resource
use. Decoupled from CLI/Obsidian specifics via prepare/finalize callbacks.
"""

from __future__ import annotations

from pathlib import Path

from . import audio, transcriber
from .utils import log


def transcribe_chunk(
    chunk_path: Path,
    model: str,
    language: str | None,
    device: str | None = None,
) -> list[dict]:
    """Transcribe one chunk, returning its segments ([] if silent/no speech)."""
    if audio.is_silent(chunk_path):
        return []
    result = transcriber.transcribe_audio(
        chunk_path, model=model, language=language, device=device
    )
    return transcriber.extract_segments(result)
```

- [ ] **Step 4: Point cli `_transcribe_chunk` at the shared helper**

In `scribe_md/cli.py`, add `from . import scheduler` to the imports (line 13 group), then replace `_transcribe_chunk` (lines 486-498) with a thin delegator so there is one implementation:

```python
def _transcribe_chunk(
    chunk_path: Path,
    model: str,
    language: str | None,
) -> list[dict]:
    """Transcribe a single chunk file, returning its segments.

    Returns an empty list if the chunk is silent or has no speech.
    """
    return scheduler.transcribe_chunk(chunk_path, model, language)
```

- [ ] **Step 5: Run to verify pass + no regressions**

Run: `pixi run pytest tests/test_scheduler.py tests/test_cli_degradation.py -q`
Expected: PASS (existing chunk-failure tests still green).

- [ ] **Step 6: Commit**

```bash
git add scribe_md/scheduler.py scribe_md/cli.py tests/test_scheduler.py
git commit -m "refactor: share transcribe_chunk via scheduler (no cli<->scheduler cycle)"
```

---

### Task 4: Parallel scheduler core (`scheduler.transcribe_in_parallel`)

**Files:**
- Modify: `scribe_md/scheduler.py` (add dataclasses + orchestration)
- Test: `tests/test_scheduler.py` (add)

**Interfaces:**
- Produces:
  - `@dataclass PreparedSource(key: str, chunk_paths: list[Path], cleanup: Callable[[], None], payload: object = None)`
  - `@dataclass RunSummary(succeeded: list[str], skipped: list[tuple[str, str]])` with `all_failed` property
  - `transcribe_in_parallel(sources, *, gpu_ids: list[int], model: str, language: str | None, prepare: Callable[[object], PreparedSource], finalize: Callable[[PreparedSource, list[list[dict]]], None], max_inflight: int) -> RunSummary`
- Consumes: `transcribe_chunk` (Task 3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scheduler.py  (add)
import threading
from pathlib import Path
from scribe_md import scheduler
from scribe_md.scheduler import PreparedSource


def _prep(n_chunks):
    def prepare(source):
        return PreparedSource(
            key=str(source),
            chunk_paths=[Path(f"{source}-{i}.wav") for i in range(n_chunks)],
            cleanup=lambda: None,
        )
    return prepare


def test_parallel_uses_all_gpus_and_preserves_order(monkeypatch):
    seen_devices = set()
    lock = threading.Lock()

    def fake_chunk(path, model, language, device=None):
        with lock:
            seen_devices.add(device)
        idx = int(path.stem.split("-")[-1])
        return [{"start": float(idx), "end": idx + 1.0, "text": f"seg{idx}"}]

    monkeypatch.setattr(scheduler, "transcribe_chunk", fake_chunk)

    written = {}

    def finalize(prepared, ordered):
        written[prepared.key] = [segs[0]["text"] for segs in ordered]

    summary = scheduler.transcribe_in_parallel(
        ["A"], gpu_ids=[0, 1], model="tiny", language="ko",
        prepare=_prep(4), finalize=finalize, max_inflight=2,
    )

    assert seen_devices == {"0", "1"}
    assert written["A"] == ["seg0", "seg1", "seg2", "seg3"]  # in order
    assert summary.succeeded == ["A"]


def test_parallel_skips_a_fully_failed_source_but_continues(monkeypatch):
    def fake_chunk(path, model, language, device=None):
        if path.stem.startswith("BAD"):
            raise RuntimeError("boom")
        return [{"start": 0.0, "end": 1.0, "text": "ok"}]

    monkeypatch.setattr(scheduler, "transcribe_chunk", fake_chunk)
    written = []
    summary = scheduler.transcribe_in_parallel(
        ["GOOD", "BAD"], gpu_ids=[0], model="tiny", language=None,
        prepare=_prep(2), finalize=lambda p, o: written.append(p.key),
        max_inflight=2,
    )
    assert written == ["GOOD"]
    assert [k for k, _ in summary.skipped] == ["BAD"]
    assert not summary.all_failed


def test_parallel_all_sources_failed_sets_all_failed(monkeypatch):
    def boom(path, model, language, device=None):
        raise RuntimeError("x")

    monkeypatch.setattr(scheduler, "transcribe_chunk", boom)
    summary = scheduler.transcribe_in_parallel(
        ["A", "B"], gpu_ids=[0], model="tiny", language=None,
        prepare=_prep(1), finalize=lambda p, o: None, max_inflight=2,
    )
    assert summary.succeeded == []
    assert summary.all_failed
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_scheduler.py -k parallel -q`
Expected: FAIL — `AttributeError: module 'scribe_md.scheduler' has no attribute 'transcribe_in_parallel'`.

- [ ] **Step 3: Implement the scheduler core**

Append to `scribe_md/scheduler.py` (and add imports `threading`, `from concurrent.futures import ThreadPoolExecutor`, `from dataclasses import dataclass, field`, `from queue import Queue`, `from typing import Callable`):

```python
@dataclass
class PreparedSource:
    """A source ready to transcribe: ordered chunks + how to finish/clean it."""

    key: str
    chunk_paths: list[Path]
    cleanup: Callable[[], None]
    payload: object = None


@dataclass
class RunSummary:
    succeeded: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (key, reason)

    @property
    def all_failed(self) -> bool:
        return bool(self.skipped) and not self.succeeded


@dataclass
class _Job:
    prepared: PreparedSource
    remaining: int
    results: dict[int, list[dict]] = field(default_factory=dict)
    failures: int = 0
    last_error: BaseException | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def transcribe_in_parallel(
    sources: list,
    *,
    gpu_ids: list[int],
    model: str,
    language: str | None,
    prepare: Callable[[object], PreparedSource],
    finalize: Callable[[PreparedSource, list[list[dict]]], None],
    max_inflight: int,
) -> RunSummary:
    """Transcribe many sources concurrently across *gpu_ids*.

    Each chunk runs on one checked-out GPU. Sources are prepared lazily and
    bounded by *max_inflight* (caps temp-disk for big URL batches). Each
    finished source is finalized (merge/write) on a single finalizer thread,
    so writes never overlap and workers return to transcribing immediately.
    """
    devices: Queue = Queue()
    for gid in gpu_ids:
        devices.put(str(gid))
    inflight = threading.Semaphore(max_inflight)
    done_queue: Queue = Queue()
    summary = RunSummary()
    SENTINEL = object()

    def finalizer() -> None:
        while True:
            job = done_queue.get()
            if job is SENTINEL:
                return
            n = len(job.prepared.chunk_paths)
            try:
                if n and job.failures == n:
                    summary.skipped.append(
                        (job.prepared.key, f"all {n} chunk(s) failed: {job.last_error}")
                    )
                    log(f"[skip] {job.prepared.key}: all chunks failed ({job.last_error})")
                else:
                    ordered = [job.results[i] for i in range(n)]
                    finalize(job.prepared, ordered)
                    summary.succeeded.append(job.prepared.key)
                    if job.failures:
                        log(
                            f"  [{job.prepared.key}] warning: {job.failures}/{n} "
                            "chunk(s) failed; transcript incomplete"
                        )
            finally:
                job.prepared.cleanup()
                inflight.release()

    finalizer_thread = threading.Thread(target=finalizer, daemon=True)
    finalizer_thread.start()

    def run_chunk(job: _Job, idx: int, chunk_path: Path) -> None:
        device = devices.get()
        err: BaseException | None = None
        try:
            segments = transcribe_chunk(chunk_path, model, language, device=device)
        except Exception as e:  # noqa: BLE001 — record, keep batch alive
            log(f"  [{job.prepared.key}] chunk {idx} failed: {e}")
            segments, err = [], e
        finally:
            devices.put(device)
        with job.lock:
            job.results[idx] = segments
            if err is not None:
                job.failures += 1
                job.last_error = err
            job.remaining -= 1
            done = job.remaining == 0
        if done:
            done_queue.put(job)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        for source in sources:
            inflight.acquire()
            try:
                prepared = prepare(source)
            except Exception as e:  # noqa: BLE001 — prepare failure skips one source
                inflight.release()
                summary.skipped.append((str(source), f"prepare failed: {e}"))
                log(f"[skip] {source}: {e}")
                continue
            if not prepared.chunk_paths:
                prepared.cleanup()
                inflight.release()
                summary.skipped.append((prepared.key, "no audio chunks"))
                continue
            job = _Job(prepared=prepared, remaining=len(prepared.chunk_paths))
            for idx, chunk_path in enumerate(prepared.chunk_paths):
                executor.submit(run_chunk, job, idx, chunk_path)
        # executor exit waits for every submitted chunk task to finish.

    done_queue.put(SENTINEL)
    finalizer_thread.join()
    return summary
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_scheduler.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scribe_md/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): bounded multi-GPU producer/consumer pipeline"
```

---

### Task 5: Config `[gpu].gpus`

**Files:**
- Modify: `scribe_md/config.py` (dataclass field, `_apply_toml`, `config_as_toml`, `DEFAULT_CONFIG_TOML`)
- Test: `tests/test_config.py` (add)

**Interfaces:**
- Produces: `ScribeMdConfig.gpus: str = ""`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  (add)
import tomllib
from scribe_md.config import ScribeMdConfig, _apply_toml, config_as_toml


def test_gpu_section_parsed():
    cfg = ScribeMdConfig()
    _apply_toml(cfg, {"gpu": {"gpus": "auto"}}, "test")
    assert cfg.gpus == "auto"


def test_gpus_default_empty_and_round_trips_through_toml():
    cfg = ScribeMdConfig()
    assert cfg.gpus == ""
    rendered = config_as_toml(cfg)
    assert "[gpu]" in rendered
    tomllib.loads(rendered)  # must remain valid TOML
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_config.py -k gpu -q`
Expected: FAIL — `AttributeError: 'ScribeMdConfig' object has no attribute 'gpus'`.

- [ ] **Step 3: Implement**

In `scribe_md/config.py`:

Add the field to `ScribeMdConfig` (after the `[live]` block, before `_sources`):

```python
    # [gpu]
    gpus: str = ""  # "", "1", "auto", "N", or "0,1" — same grammar as --gpus
```

In `_apply_toml`, after the `live` handling block:

```python
    gpu = data.get("gpu", {})
    if "gpus" in gpu:
        cfg.gpus = str(gpu["gpus"])
```

In `config_as_toml`, append before the final `]`:

```python
        "",
        "[gpu]",
        f"gpus = {_toml_str(cfg.gpus)}",
```

In `DEFAULT_CONFIG_TOML`, append a section:

```toml

[gpu]
gpus = ""                 # "", "1", "auto", "N", or "0,1" (same as --gpus)
```

- [ ] **Step 4: Run to verify pass + full config tests**

Run: `pixi run pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scribe_md/config.py tests/test_config.py
git commit -m "feat(config): [gpu].gpus setting"
```

---

### Task 6: CLI input collection (`--from-file`, variadic args, `-o` validation)

**Files:**
- Modify: `scribe_md/cli.py` (new `_collect_inputs` helper + `_FromFile` option type)
- Test: `tests/test_cli_inputs.py`

**Interfaces:**
- Produces: `_collect_inputs(positional: list[str], from_file: Path | None) -> list[str]` (raises `typer.Exit(1)` on empty); `_validate_single_output(inputs: list, output: Path | None) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_inputs.py
import pytest
import typer
from scribe_md.cli import _collect_inputs, _validate_single_output


def test_collect_from_positional_only():
    assert _collect_inputs(["a.mp4", "b.mp4"], None) == ["a.mp4", "b.mp4"]


def test_collect_from_file_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("url1\n\n# comment\n  url2  \n")
    assert _collect_inputs([], f) == ["url1", "url2"]


def test_collect_merges_positional_and_file(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("url2\n")
    assert _collect_inputs(["url1"], f) == ["url1", "url2"]


def test_collect_empty_raises_exit():
    with pytest.raises(typer.Exit):
        _collect_inputs([], None)


def test_output_with_multiple_inputs_raises():
    from pathlib import Path
    with pytest.raises(typer.Exit):
        _validate_single_output(["a", "b"], Path("out.md"))
    # single input + -o is fine
    _validate_single_output(["a"], Path("out.md"))
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_cli_inputs.py -q`
Expected: FAIL — `ImportError: cannot import name '_collect_inputs'`.

- [ ] **Step 3: Implement the helpers in `scribe_md/cli.py`**

Add near the other option types (after line 65):

```python
_FromFile = Annotated[Optional[Path], typer.Option(
    "--from-file", help="Read inputs (one per line; '#' comments allowed) from a file",
)]
_Gpus = Annotated[Optional[str], typer.Option(
    "--gpus", help="GPUs for parallel transcription: 'auto', N, or '0,1' (CUDA only)",
)]
```

Add the helpers near the Obsidian helpers (after `_validate_daily_note`):

```python
def _collect_inputs(positional: list[str], from_file: Path | None) -> list[str]:
    """Merge positional inputs with a --from-file list; fail fast if empty."""
    inputs = list(positional or [])
    if from_file is not None:
        for line in from_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                inputs.append(line)
    if not inputs:
        console.print("[red]Error:[/red] no inputs given (positional or --from-file).")
        raise typer.Exit(1)
    return inputs


def _validate_single_output(inputs: list, output: Path | None) -> None:
    """`-o/--output` names one file, so reject it with multiple inputs."""
    if output is not None and len(inputs) > 1:
        console.print(
            "[red]Error:[/red] --output/-o works with a single input only; "
            "with multiple inputs, outputs are written to the output directory."
        )
        raise typer.Exit(1)
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_cli_inputs.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scribe_md/cli.py tests/test_cli_inputs.py
git commit -m "feat(cli): --from-file + multi-input collection helpers"
```

---

### Task 7: Wire multi-input + `--gpus` into `file`/`url`

This task makes `file` and `url` accept lists and dispatch through a single batch runner that chooses parallel (CUDA, >1 device) or the existing sequential path.

**Files:**
- Modify: `scribe_md/cli.py` (variadic args on `file`/`url`; new `_resolve_gpu_ids`, `_run_batch`, `prepare`/`finalize` closures)
- Test: `tests/test_cli_batch.py`

**Interfaces:**
- Consumes: `gpu.discover_cuda_devices`, `gpu.resolve_gpu_spec` (Task 1); `scheduler.transcribe_in_parallel`, `PreparedSource` (Task 4); existing `audio.*`, `merger.merge_segments`, `_apply_postprocessing`, `_write_obsidian_output`, `_build_obsidian_metadata`, `_run_diarization`, `diarize.assign_speakers`.
- Produces: `_resolve_gpu_ids(spec: str | None) -> list[int]` (returns `[]` when parallelism is unavailable/declined → caller runs sequential).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_batch.py
import pytest
from typer.testing import CliRunner
from scribe_md import cli
from scribe_md.cli import app, _resolve_gpu_ids

runner = CliRunner()


def test_resolve_gpu_ids_sequential_default(monkeypatch):
    monkeypatch.setattr(cli.gpu, "discover_cuda_devices", lambda: [0, 1])
    # default (None / "1") => not parallel
    assert _resolve_gpu_ids(None) == []
    assert _resolve_gpu_ids("1") == []


def test_resolve_gpu_ids_auto(monkeypatch):
    monkeypatch.setattr(cli.gpu, "discover_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(cli, "_backend_is_cuda", lambda: True)
    assert _resolve_gpu_ids("auto") == [0, 1]


def test_resolve_gpu_ids_falls_back_when_not_cuda(monkeypatch):
    monkeypatch.setattr(cli.gpu, "discover_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(cli, "_backend_is_cuda", lambda: False)
    # non-CUDA backend: warn + sequential ([] means "run sequentially")
    assert _resolve_gpu_ids("auto") == []


def test_file_rejects_output_with_multiple_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("scribe_md.cli.load_config", lambda: cli.ScribeMdConfig())
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"\x00" * 100)
    b.write_bytes(b"\x00" * 100)
    result = runner.invoke(app, ["file", str(a), str(b), "-o", "out.md"])
    assert result.exit_code == 1
    assert "single input only" in result.output
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_cli_batch.py -q`
Expected: FAIL — `ImportError: cannot import name '_resolve_gpu_ids'` / missing `gpu` import.

- [ ] **Step 3: Implement**

In `scribe_md/cli.py` imports (line 13 group) add: `from . import gpu` and `import shutil`, and `from .scheduler import PreparedSource`.

Add the backend/GPU helpers (near `_collect_inputs`):

```python
def _backend_is_cuda() -> bool:
    """True only when the active backend is whisper.cpp built for CUDA."""
    from .backends import get_backend
    from .backends.whispercpp import _read_built_accel, detect_accel

    backend = get_backend()
    if backend.name != "whispercpp":
        return False
    return (_read_built_accel() or detect_accel()) == "cuda"


def _resolve_gpu_ids(spec: str | None) -> list[int]:
    """Resolve --gpus to device ids, or [] to mean 'run sequentially'.

    Returns [] for the default/single case and for non-CUDA backends (with a
    one-line notice), so callers treat [] as the existing sequential path.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "1"):
        return []
    if not _backend_is_cuda():
        console.print(
            "[yellow]Note:[/yellow] --gpus needs the CUDA whisper.cpp backend; "
            "running sequentially on this platform."
        )
        return []
    try:
        ids = gpu.resolve_gpu_spec(spec, gpu.discover_cuda_devices())
    except gpu.GpuSpecError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    return ids if len(ids) > 1 else []
```

Change the `file` command signature: replace `audio_file: Path = typer.Argument(...)` with

```python
    audio_files: list[Path] = typer.Argument(None, help="Audio file(s) (WAV, MP3, ...)"),
    from_file: _FromFile = None,
    gpus: _Gpus = None,
```

and at the top of the `file` body, replace the single-file existence checks with:

```python
    _guard_summarize_on_linux(summarize)
    inputs = [Path(p) for p in _collect_inputs([str(p) for p in (audio_files or [])], from_file)]
    _validate_single_output(inputs, output)
    for p in inputs:
        if not p.exists():
            console.print(f"[red]Error:[/red] {p} not found")
            raise typer.Exit(1)
        if p.stat().st_size == 0:
            console.print(f"[red]Error:[/red] {p} is empty (0 bytes)")
            raise typer.Exit(1)
    cfg = load_config()
    opts = _resolve_common_options(cfg, model=model, language=language, ...)  # unchanged kwargs
    r_chunk_seconds = _resolve(chunk_seconds, cfg.chunk_seconds)
    r_incremental = _resolve(incremental, cfg.incremental)
    gpu_ids = _resolve_gpu_ids(_resolve(gpus, cfg.gpus))
    if gpu_ids and r_incremental:
        log("Incremental output disabled under multi-GPU parallelism.")
        r_incremental = False
    _run_batch(
        inputs, kind="file", single_output=output, cfg=cfg, opts=opts,
        chunk_seconds=r_chunk_seconds, overlap_seconds=opts.overlap_seconds,
        incremental=r_incremental, daily_note=daily_note, summarize=summarize,
        gpu_ids=gpu_ids,
    )
```

Apply the analogous change to `url` (variadic `urls: list[str]`, `from_file`, `gpus`; expand each URL via `downloader.get_playlist_entries` into `(entry_url, title)` sources before `_run_batch(..., kind="url", ...)`). The existing per-video `try/except ... Skipping` loop is retained for the **sequential** path; the parallel path's per-source skip is handled by `RunSummary`.

Add `_run_batch` (single dispatch point). For `gpu_ids == []` it loops the existing per-source code paths unchanged (`_transcribe_chunked`/`_transcribe_single` for files; `_transcribe_url` for urls), preserving today's behavior and `--incremental`. For `gpu_ids` with >1 device it builds `prepare`/`finalize` closures over the existing helpers and calls `scheduler.transcribe_in_parallel`:

```python
def _run_batch(inputs, *, kind, single_output, cfg, opts, chunk_seconds,
               overlap_seconds, incremental, daily_note, summarize, gpu_ids):
    if not gpu_ids:
        _run_batch_sequential(inputs, kind=kind, single_output=single_output,
                              cfg=cfg, opts=opts, chunk_seconds=chunk_seconds,
                              overlap_seconds=overlap_seconds, incremental=incremental,
                              daily_note=daily_note, summarize=summarize)
        return

    log(f"Transcribing {len(inputs)} source(s) across GPUs {gpu_ids}...")
    # Warm up once so workers don't race to build the binary / download the model.
    from .backends import get_backend
    from .backends import whispercpp
    if get_backend().name == "whispercpp":
        whispercpp.ensure_whisper_binary()
        whispercpp._ensure_model_file(opts.model)

    def prepare(source) -> PreparedSource:
        tmpdir = Path(tempfile.mkdtemp(prefix="scribe-md-"))
        if kind == "url":
            entry_url, title = source
            raw, title = downloader.download_audio(entry_url, tmpdir, title=title)
            src_label = f"YouTube: {title}"
            out = _output_path_for(None, single_output, cfg.output_directory, title=title)
        else:
            title = source.stem
            raw = source
            src_label = f"file: {source.name}"
            out = _output_path_for(source, single_output, cfg.output_directory)
        converted = tmpdir / "converted.wav"
        log(f"Converting {title} to 16kHz mono...")
        audio.convert_to_16k_mono(raw, converted)
        duration = audio.get_duration(converted)
        if _should_chunk(duration, chunk_seconds):
            chunks = audio.split_audio(converted, tmpdir, chunk_seconds, overlap_seconds)
        else:
            chunks = [converted]
        turns = _run_diarization(converted, hf_token=opts.hf_token,
                                 num_speakers=opts.num_speakers) if opts.diarize else None
        payload = {"out": out, "duration": duration, "turns": turns, "source": src_label}
        return PreparedSource(
            key=out.name, chunk_paths=chunks,
            cleanup=lambda: shutil.rmtree(tmpdir, ignore_errors=True),
            payload=payload,
        )

    def finalize(prepared, ordered):
        p = prepared.payload
        if p["turns"] is not None:
            for idx, segs in enumerate(ordered):
                offset = 0.0 if idx == 0 else idx * chunk_seconds - overlap_seconds
                ordered[idx] = diarize.assign_speakers(segs, p["turns"], time_offset=offset)
        text = merger.merge_segments(
            ordered, chunk_duration=chunk_seconds, overlap=overlap_seconds,
            timestamps=opts.ts, timestamp_mode=opts.ts_mode, paragraph_gap=opts.paragraph_gap,
        )
        text = _apply_postprocessing(text, clean=opts.clean, summarize=summarize,
                                     summary_model=opts.summary_model)
        metadata = _build_obsidian_metadata(source=p["source"], duration=p["duration"],
                                            language=opts.language, model=opts.model)
        _write_obsidian_output(text, p["out"], opts.vault, daily_note, opts.frontmatter,
                               metadata, opts.daily_note_folder)

    summary = scheduler.transcribe_in_parallel(
        inputs, gpu_ids=gpu_ids, model=opts.model, language=opts.language,
        prepare=prepare, finalize=finalize, max_inflight=max(2, len(gpu_ids)),
    )
    log(f"Done: {len(summary.succeeded)} written, {len(summary.skipped)} skipped.")
    if summary.all_failed:
        console.print("[red]Error:[/red] all sources failed to transcribe.")
        raise typer.Exit(1)
```

Add the small output-path helper:

```python
def _output_path_for(src: Path | None, single_output: Path | None,
                     output_directory: str, *, title: str | None = None) -> Path:
    if single_output is not None:
        return single_output
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(title) if title is not None else src.stem
    return out_dir / f"{stem}.md"
```

`_run_batch_sequential` wraps the existing logic: for `kind == "file"` it runs the current `try/except`-wrapped convert→chunk/single→write block (now in a loop over `inputs`); for `kind == "url"` it runs the existing playlist/single `_transcribe_url` loop. Keep the existing per-command `except` handlers around `_run_batch` so sequential errors still map to `typer.Exit(1)`.

- [ ] **Step 4: Run to verify pass + full suite**

Run: `pixi run pytest tests/test_cli_batch.py -q && pixi run pytest -q`
Expected: PASS; full suite green.

- [ ] **Step 5: Manual smoke test (document only — needs a real GPU box)**

```bash
# On the Linux/NVIDIA machine, in the cuda env:
env SCRIBE_MD_WHISPER_ACCEL=cuda pixi run -e cuda \
  scribe-md file Lecture{5..7}.mp4 --gpus auto -l ko --clean
# Expect: "Transcribing 3 source(s) across GPUs [0, 1]..." and 3 .md files.
```

- [ ] **Step 6: Commit**

```bash
git add scribe_md/cli.py tests/test_cli_batch.py
git commit -m "feat(cli): multi-input + --gpus parallel transcription"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md` (new "Multi-GPU / batch transcription" section + deferred-backends note)
- Modify: `docs/LINUX.md` (cross-reference + `--gpus auto` example)
- Modify: `CHANGELOG.md` (`[Unreleased]` → `### Added`)

- [ ] **Step 1: README — add a section** after "Chunked Transcription":

````markdown
## Multi-GPU / Batch Transcription

Pass multiple inputs in one command; each produces its own `.md`:

```bash
scribe-md file Lecture{5..15}.mp4 -l ko --clean        # many files
scribe-md url URL1 URL2 URL3 -l ko                     # many URLs
scribe-md url --from-file urls.txt -l ko               # a list file
```

On a multi-GPU NVIDIA machine, `--gpus` transcribes chunks from all inputs
concurrently across devices:

```bash
scribe-md file Lecture{5..15}.mp4 --gpus auto -l ko    # all CUDA GPUs
scribe-md url --from-file urls.txt --gpus 0,1 -l ko    # specific devices
```

`--gpus`: `auto` (all CUDA devices), an integer `N` (first N), or a list
(`0,1`). Settable as `[gpu] gpus` in config. Pin `SCRIBE_MD_WHISPER_ACCEL=cuda`
and use the `cuda` pixi env so the CUDA backend is selected (see
[docs/LINUX.md](docs/LINUX.md)).

> **Scope:** multi-GPU parallelism is **CUDA-only** today. On Apple Silicon
> (single unified-memory device) and the Linux **Vulkan** backend, multi-input
> still works but runs sequentially. Vulkan/MLX multi-GPU is possible future
> work — open an issue if you need it. `--incremental` is disabled under
> multi-GPU (chunks finish out of order).
````

- [ ] **Step 2: docs/LINUX.md — add a short note** referencing `--gpus auto`, pinning `SCRIBE_MD_WHISPER_ACCEL=cuda`, and the `cuda` pixi env.

- [ ] **Step 3: CHANGELOG — under `## [Unreleased]`:**

```markdown
### Added
- Multi-input transcription: `file` and `url` accept several inputs in one run
  (plus `--from-file`), each written to its own `.md`.
- `--gpus` (and `[gpu].gpus` config) transcribes chunks across multiple NVIDIA
  GPUs in parallel (CUDA only; Vulkan/MLX run sequentially). `--incremental` is
  disabled under multi-GPU.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/LINUX.md CHANGELOG.md
git commit -m "docs: multi-GPU / batch transcription usage and scope"
```

---

## Self-Review

**1. Spec coverage:**
- Multi-input CLI + `--from-file` → Task 6. ✓
- `--gpus` grammar + config → Tasks 1, 5, 7. ✓
- Unified queue / scheduler + ordering + bounded inflight → Task 4. ✓
- Backend `device` passthrough → Task 2. ✓
- Per-chunk helper move (no cli↔scheduler cycle) → Task 3. ✓
- Warm-up before fan-out → Task 7 (`_run_batch`). ✓
- Failure handling (skip fully-failed source, all-fail non-zero) → Task 4 + Task 7. ✓
- `--incremental` disabled under parallel → Task 7. ✓
- Non-CUDA fallback → Task 7 (`_resolve_gpu_ids`). ✓
- CUDA-only scope + Vulkan/MLX deferred noted in README → Task 8. ✓
- Hermetic tests → every task mocks GPU/subprocess/backend. ✓
- `-o` single-input rule → Task 6. ✓

**2. Placeholder scan:** No TBD/TODO. The one prose-only step is Task 7/Step 5 (manual smoke test), explicitly marked "document only — needs a real GPU box" because it cannot run in the hermetic suite; the executable behavior is covered by Task 7 unit tests.

**3. Type consistency:** `transcribe_chunk(chunk_path, model, language, device=None)` is identical in Tasks 3, 4. `PreparedSource(key, chunk_paths, cleanup, payload)` defined in Task 4 and constructed in Task 7. `transcribe_in_parallel(...)` kwargs match between Task 4 definition and Task 7 call. `device: str | None` is consistent across base/mlx/whispercpp/transcriber (Task 2) and the scheduler (Tasks 3-4). `_resolve_gpu_ids` returns `[]` for sequential everywhere it's used.
