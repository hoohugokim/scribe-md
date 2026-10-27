<!-- statutor: plane=plan | writer=human | agents read ONLY the section below the marker -->
# ROADMAP

## Current milestone <!-- agent-visible -->
No product milestone is committed. main is the integration line (D-0002). The TUI is an experiment on `explore/tui` (D-0004), not this milestone.

## Later (human context, agents ignore)
- T-0001 CUDA smoke stays open. It needs a Linux NVIDIA machine.
- Vulkan multi-GPU (`GGML_VK_VISIBLE_DEVICES`) stays deferred until someone asks for it.
- MLX multi-GPU does not apply. Apple Silicon is one unified-memory device.
- Inside the TUI experiment, one shared job still applies (D-0003). Widgets do not call ffmpeg, whisper, or yt-dlp.
