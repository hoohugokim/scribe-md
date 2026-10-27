# Multi-GPU parallel transcription for scribe-md

**Date:** 2026-06-24
**Branch:** `feat/multi-gpu-parallel`
**Status:** Design — approved, pending spec review

## Goal

Let one `scribe-md` invocation accept **many inputs** (files and/or YouTube
URLs) and transcribe them as a **single unified work queue spread across
multiple NVIDIA GPUs**, so long videos and large batches both keep every GPU
busy. This replaces hand-written per-GPU batch shell scripts.

Two capabilities, layered:

1. **Multi-input CLI (all platforms).** `file` and `url` take a list of inputs
   (plus `--from-file`). Each input still produces its own `.md`. This works on
   macOS and Linux regardless of GPU count — it just runs sequentially when
   parallelism isn't available.
2. **Multi-GPU parallelism (CUDA only, opt-in).** With `--gpus`, chunks from all
   inputs are distributed across the chosen CUDA devices and transcribed
   concurrently.

## Scope

**In scope:**
- Variadic inputs for `file` (`list[Path]`) and `url` (`list[str]`), plus
  `--from-file PATH` (newline-separated; blank lines and `#` comments ignored).
- `--gpus` option: `1`/unset = today's sequential behavior; `auto` = all CUDA
  devices; `N` = first N; `0,1,3` = explicit device ids. Also settable in config
  (`[gpu] gpus = "..."`).
- A unified producer/consumer scheduler that fans chunks across GPUs, preserves
  per-source chunk order, and writes one `.md` per source.
- Per-subprocess GPU pinning via `CUDA_VISIBLE_DEVICES`.
- One-time warm-up (binary build + model download) before fan-out.
- Failure handling consistent with the chunk-failure fix on
  `fix/chunk-failure-not-silent` (a failed chunk is not "silence"; a fully
  failed source is skipped; an all-failed run exits non-zero).
- README note documenting CUDA-only multi-GPU and the deferred backends.

**Out of scope (deferred future work):**
- **Vulkan multi-GPU** (AMD / non-CUDA NVIDIA). Mechanism would be
  `GGML_VK_VISIBLE_DEVICES`; deferred until requested. Noted in README.
- **MLX multi-GPU** — Apple Silicon is a single unified-memory device; not
  applicable. Noted in README.
- Multi-**machine** distribution.
- Parallelizing diarization or summarization (CPU/Apple-only paths).

## Background: why this is a small change

The whisper.cpp backend already runs each chunk as an **isolated `whisper-cli`
subprocess** (`whispercpp.py:274`) — the module docstring states this isolation
is deliberate so a GPU crash cannot take down Python. Chunks are independent
units. So GPU parallelism reduces to: run N of those subprocesses at once, each
with `CUDA_VISIBLE_DEVICES` set to a different device. The single CUDA-built
binary supports every device; the env var selects which one a process uses.

The current "chunks must be sequential" rule (`cli.py:522`) is an MLX / Apple
Metal command-buffer limitation, **not** a whisper.cpp one, and does not apply
to the subprocess backend.

## CLI surface

```bash
# Many local files (shell expands the glob/brace to variadic args)
scribe-md file Lecture{5..15}.mp4 --gpus auto -l ko --clean

# Many URLs (each may itself be a playlist, which expands further)
scribe-md url URL1 URL2 URL3 --gpus 0,1 -l ko --clean

# Big lists from a file
scribe-md url --from-file urls.txt --gpus auto -l ko --clean
```

Rules:
- **At least one input** must come from positional args or `--from-file`
  (combinable); otherwise a clear error.
- **`-o/--output` is valid only with exactly one resolved input.** With multiple
  inputs, outputs are written per-source into the output directory, named after
  the source title — exactly as playlist transcription already behaves.
- `--gpus` grammar (resolved CLI > config `[gpu].gpus` > default `1`):
  - unset / `1` → sequential (current behavior, all platforms)
  - `auto` → every discovered CUDA device
  - integer `N` → first `N` discovered devices
  - list `0,1,3` → those explicit device ids
- If `--gpus` requests >1 device but the active backend is **not** CUDA
  whisper.cpp (macOS/MLX, Linux Vulkan/CPU), scribe-md prints a one-line notice
  and **falls back to sequential** rather than erroring.

## GPU discovery & validation (`scribe_md/gpu.py`)

A small, isolated module:
- `discover_cuda_devices() -> list[int]` — parse `nvidia-smi -L` (lines like
  `GPU 0: NVIDIA ... (UUID: ...)`); empty list if `nvidia-smi` is absent or
  fails.
- `resolve_gpu_spec(spec: str | None, available: list[int]) -> list[int]` — pure
  function mapping the `--gpus` grammar onto available devices; validates that
  explicit ids exist and that the count is ≥1; raises a clear error otherwise.

Pure parsing is separated from the `nvidia-smi` subprocess so it is unit-testable
without hardware.

## Architecture (`scribe_md/scheduler.py`)

A **bounded producer/consumer pipeline**. `cli.py` stays the thin orchestrator
and owns "what a source is" and "what to do with its result"; the scheduler owns
concurrency, GPU assignment, ordering, and resource bounding.

Interface (callbacks keep the scheduler decoupled from download/obsidian
details):

```
transcribe_in_parallel(
    sources: list[Source],          # opaque to the scheduler
    *,
    gpu_ids: list[int],
    model: str,
    language: str | None,
    prepare: Callable[[Source], PreparedSource],   # download/convert/split (+diarize)
    finalize: Callable[[PreparedSource, list[list[dict]]], None], # merge→postprocess→write
    max_inflight: int,
) -> RunSummary
```

The scheduler performs the per-chunk transcribe itself (so the GPU device it
checked out is applied at the call site); `model`/`language` are run-wide, so
they are plain parameters rather than threaded through a callback.

- `PreparedSource` carries: ordered chunk paths, a per-source temp-dir handle,
  output metadata, and (optional) diarization turns.
- **Producer:** iterates `sources`; for each, calls `prepare(source)` (download /
  convert to 16k mono / split into a per-source temp dir / optional diarize) and
  submits that source's chunks as tasks. A **semaphore bounds in-flight sources**
  to `max_inflight = max(2, len(gpu_ids))` so a big URL batch does not download
  everything to disk at once.
- **Consumers:** `ThreadPoolExecutor(max_workers=len(gpu_ids))`. A thread-safe
  `Queue` seeded with `gpu_ids` is the device pool: each chunk task checks out a
  device id, transcribes with it — `transcriber.transcribe_audio(chunk,
  model=model, language=language, device=str(id))` + `extract_segments`, with
  the same silence pre-check as today's `_transcribe_chunk` — then returns the
  id to the pool. Threads suffice because the work is in a subprocess (the GIL
  is released while waiting). Faster GPUs naturally pull more chunks (load
  balancing).
- **Ordering:** results are stored per source keyed by chunk index; when a
  source's last chunk completes, a completion step sorts by index, calls
  `finalize`, cleans the source's temp dir, and releases its semaphore slot.
- **Single-GPU / non-CUDA path:** when `len(gpu_ids) <= 1` (or fallback), the
  scheduler runs sources sequentially through the existing
  `_transcribe_chunked` / `_transcribe_single` path — no behavior change.

### Data flow

```
inputs (args + --from-file)
  → expand (playlist URLs → videos)            [producer]
  → per source: prepare() download/convert/split (+diarize)
  → enqueue (source, chunk_index, chunk_path)  → GPU pool (N workers, device-pinned)
  → collect results per source, in index order
  → finalize(): merge_segments → clean/summarize → write one .md
```

## Backend API change

Add an optional device argument to the backend contract:

```
Backend.transcribe(audio_path, *, model, language, device: str | None = None)
```

- `WhisperCppBackend`: when `device` is set, run the subprocess with
  `env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}` (the only call site that
  needs `env=`). `device=None` → unchanged behavior.
- `MLXBackend`: ignores `device` (single unified-memory device).
- `transcriber.transcribe_audio(...)` gains a matching `device=None` parameter
  that it passes straight through to `backend.transcribe(device=...)`.

This is the one cross-cutting interface change; everything else is additive.

## Warm-up (avoid build/download races)

Before fan-out, the scheduler calls `ensure_whisper_binary()` once and
`_ensure_model_file(model)` once (for the whisper.cpp backend). This prevents N
workers from racing to build the binary or download the GGML weights
simultaneously on first use. The atomic-publish download already makes a race
safe, but warming up avoids N redundant downloads and N cmake builds.

## Failure handling

Builds on the `fix/chunk-failure-not-silent` change (assumed merged or rebased
in):
- A chunk that raises is recorded as a **failure**, not mislabeled "silent".
- If **every chunk of a source** fails, that source is **skipped** with a clear
  error and the run continues (playlist-style resilience).
- If **every source** fails entirely, the command exits **non-zero**.
- Per-source result/failure accumulation is guarded by a lock (multiple workers
  write concurrently).
- `RunSummary` reports per-source success/skip/failure counts for the final log
  line and exit code.

## Incremental output

`--incremental` is **incompatible with effective parallelism > 1**: out-of-order
completion across interleaved sources makes a single streamed file meaningless.
Under multi-GPU, incremental is disabled with a one-line warning. Single-GPU
keeps today's incremental behavior unchanged.

## Configuration

New `[gpu]` section in `config.toml`:

```toml
[gpu]
gpus = ""          # "", "1", "auto", "2", or "0,1" — same grammar as --gpus
```

Resolution: `--gpus` (CLI) > `[gpu].gpus` (config) > `""`/`1` (sequential
default). Parsed/validated by `resolve_gpu_spec`.

## Documentation

- **README:** new "Multi-GPU / batch transcription" section showing the multi-
  input and `--gpus` usage; an explicit note that multi-GPU parallelism is
  **CUDA-only for now**, and that **Vulkan and MLX multi-GPU are possible future
  work** (open an issue if needed).
- **docs/LINUX.md:** brief cross-reference and the `--gpus auto` example;
  reiterate pinning `SCRIBE_MD_WHISPER_ACCEL=cuda` and using the `cuda` pixi env.
- **CHANGELOG:** `Added` entries for multi-input, `--from-file`, and `--gpus`.

## Testing (fully hermetic — no GPU/network, matching the suite)

- `gpu.py`: `resolve_gpu_spec` grammar (auto/N/list/invalid id/zero); device
  discovery with a mocked `nvidia-smi -L` output and with `nvidia-smi` absent.
- Scheduler with a **mock `transcribe_chunk`** that records the device it
  received and can simulate latency/failure:
  - all chosen GPU ids are actually used;
  - per-source output order is preserved despite out-of-order completion;
  - `max_inflight` is respected (never more than the bound prepared at once);
  - warm-up callback runs exactly once;
  - all-fail → non-zero summary; partial-fail → source skipped, run continues.
- CLI: `--from-file` parsing (comments/blanks); `-o` + multiple inputs → error;
  at-least-one-input validation; `--gpus >1` on a non-CUDA backend → sequential
  fallback notice; `--incremental` + multi-GPU → disabled warning.
- Backend: `WhisperCppBackend.transcribe(device=...)` injects
  `CUDA_VISIBLE_DEVICES` into the subprocess `env` (assert via a mocked
  `subprocess.run`); `device=None` leaves env untouched; `MLXBackend` ignores it.

## Module boundaries (new/changed)

- **New `scribe_md/gpu.py`** — device discovery + `--gpus` grammar (pure +
  one subprocess). ~60 lines.
- **New `scribe_md/scheduler.py`** — producer/consumer orchestration. ~150 lines.
  The per-chunk transcribe helper (silence check + device-pinned transcribe +
  `extract_segments`, today's `cli._transcribe_chunk`) moves here (or to
  `transcriber.py`) so the scheduler never imports `cli` (avoids a cycle).
- **`scribe_md/backends/*`** — add `device` param to the protocol + both backends.
- **`scribe_md/cli.py`** — variadic args, `--from-file`, `--gpus`, config
  resolution, and wiring `prepare`/`transcribe_chunk`/`finalize` closures to the
  scheduler. Net `cli.py` growth kept small by delegating to `scheduler.py`.
- **`scribe_md/config.py`** — `[gpu].gpus` field.

## Risks / open considerations

- **VRAM:** each concurrent worker loads the model on its GPU independently
  (`large-v3` ≈ 3–4 GB). Documented; per-GPU concurrency is 1, so peak is one
  model per GPU.
- **Diarization** runs per source in `prepare` (CPU); with `--diarize` on a big
  batch the producer can become the bottleneck. Acceptable for v1; noted.
- **Granularity:** parallel speedup needs chunks ≥ GPUs. A short single file
  (few chunks) sees limited benefit; documented hint to lower `--chunk-seconds`
  for a single long file on many GPUs.
```
