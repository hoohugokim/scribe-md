<!-- statutor: plane=state | policy=state (doctor-checked) | writer=orchestrator | stable IDs, one line per task -->
# TASKS

- [ ] T-0001 Run the CUDA `--gpus` smoke in `docs/superpowers/plans/2026-06-24-multi-gpu-parallel.md` Task 7 step 5 on a Linux NVIDIA machine (`SCRIBE_MD_WHISPER_ACCEL=cuda`, `pixi run -e cuda`, three inputs, one `.md` each)
- [ ] T-0002 Extract one transcription job from `scribe_md/cli.py` that the CLI and a future TUI both call, emitting progress events and domain errors instead of Rich output and `typer.Exit`
