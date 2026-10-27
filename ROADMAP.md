<!-- statutor: plane=plan | writer=human | agents read ONLY the section below the marker -->
# ROADMAP

## Current milestone <!-- agent-visible -->
v0.3.0 is on main: multi-input and CUDA `--gpus` are merged, and `pixi run pytest -q` passed (238). This milestone is done when T-0001, the NVIDIA CUDA smoke, has been run and recorded, or a later decision defers it.

## Later (human context, agents ignore)
- Vulkan multi-GPU (`GGML_VK_VISIBLE_DEVICES`) stays deferred until someone asks for it.
- MLX multi-GPU does not apply. Apple Silicon is one unified-memory device.
