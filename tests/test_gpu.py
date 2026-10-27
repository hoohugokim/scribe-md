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
