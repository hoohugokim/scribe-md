<!-- statutor: plane=plan | writer=human | agents read ONLY the section below the marker -->
# ROADMAP

## Current milestone <!-- agent-visible -->
A TUI on the existing pipeline (D-0003). Done means file, url, and live run through one job interface (T-0002), the CLI and the TUI share it, progress is not scraped from stderr, and `pixi run pytest -q` passes. Do not add a second transcription pipeline.

## Later (human context, agents ignore)
- T-0001 CUDA smoke stays open and does not block the TUI. It needs a Linux NVIDIA machine.
- Vulkan multi-GPU (`GGML_VK_VISIBLE_DEVICES`) stays deferred until someone asks for it.
- MLX multi-GPU does not apply. Apple Silicon is one unified-memory device.
