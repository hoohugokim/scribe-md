import pytest

from scribe_md.backends import whispercpp as w


SAMPLE_JSON = {
    "transcription": [
        {"offsets": {"from": 0, "to": 2000}, "text": " Hello world"},
        {"offsets": {"from": 2000, "to": 4500}, "text": " second segment "},
    ]
}


def test_parse_json_converts_ms_to_seconds_and_strips_text():
    result = w.parse_whispercpp_json(SAMPLE_JSON)
    segs = result["segments"]
    assert segs[0] == {"start": 0.0, "end": 2.0, "text": "Hello world", "no_speech_prob": 0.0}
    assert segs[1] == {"start": 2.0, "end": 4.5, "text": "second segment", "no_speech_prob": 0.0}


def test_parse_json_empty_transcription():
    assert w.parse_whispercpp_json({"transcription": []}) == {"segments": []}


def test_resolve_model_maps_presets_to_ggml():
    assert w.resolve_model_filename("small") == "ggml-small.bin"
    assert w.resolve_model_filename("large-v3") == "ggml-large-v3.bin"
    assert w.resolve_model_filename("large") == "ggml-large-v3.bin"


def test_resolve_model_passthrough_for_unknown():
    assert w.resolve_model_filename("ggml-custom.bin") == "ggml-custom.bin"


def test_accel_cmake_flags():
    assert w.accel_cmake_flags("vulkan") == ["-DGGML_VULKAN=1"]
    assert w.accel_cmake_flags("cuda") == ["-DGGML_CUDA=1"]
    assert w.accel_cmake_flags("cpu") == []


def test_detect_accel_env_override(monkeypatch):
    monkeypatch.setenv("SCRIBE_MD_WHISPER_ACCEL", "cpu")
    assert w.detect_accel() == "cpu"


def test_detect_accel_prefers_cuda_when_toolkit_present(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/" + name
                        if name in ("nvcc", "nvidia-smi") else None)
    assert w.detect_accel() == "cuda"


def test_detect_accel_falls_back_to_vulkan(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/vulkaninfo"
                        if name == "vulkaninfo" else None)
    assert w.detect_accel() == "vulkan"


def test_detect_accel_cpu_when_nothing_present(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: None)
    assert w.detect_accel() == "cpu"


def test_build_command_shape(tmp_path):
    cmd = w._build_command("/bin/whisper-cli", tmp_path / "m.bin",
                           tmp_path / "in.wav", tmp_path / "out", "en")
    assert cmd[0] == "/bin/whisper-cli"
    assert "-oj" in cmd and "--no-prints" in cmd
    assert cmd[cmd.index("-l") + 1] == "en"


def test_build_command_auto_language(tmp_path):
    cmd = w._build_command("/bin/whisper-cli", tmp_path / "m.bin",
                           tmp_path / "in.wav", tmp_path / "out", None)
    assert cmd[cmd.index("-l") + 1] == "auto"
