"""Multi-GPU parallel transcription scheduler.

Owns concurrency, GPU assignment, per-source ordering, and bounded resource
use. Decoupled from CLI/Obsidian specifics via prepare/finalize callbacks.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Callable

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
                except Exception as e:  # noqa: BLE001 — finalize failure skips one source
                    log(f"[skip] {job.prepared.key}: finalize failed: {e}")
                    summary.skipped.append((job.prepared.key, f"finalize failed: {e}"))
            finally:
                job.prepared.cleanup()
                inflight.release()

    finalizer_thread = threading.Thread(target=finalizer, daemon=True)
    finalizer_thread.start()

    def run_chunk(job: _Job, idx: int, chunk_path: Path) -> None:
        device = devices.get()
        segments: list[dict] = []
        err: BaseException | None = None
        try:
            segments = transcribe_chunk(chunk_path, model, language, device=device)
        except BaseException as e:  # noqa: BLE001 — record, keep batch alive
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
