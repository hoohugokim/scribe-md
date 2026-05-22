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


from pathlib import Path


def test_ensure_binary_returns_existing(monkeypatch, tmp_path):
    fake_bin = tmp_path / "whisper-cli"
    fake_bin.write_text("")
    monkeypatch.setattr(w, "WHISPER_BIN", fake_bin)
    assert w.ensure_whisper_binary() == fake_bin


def test_ensure_binary_errors_when_source_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "WHISPER_BIN", tmp_path / "absent" / "whisper-cli")
    monkeypatch.setattr(w, "VENDOR_DIR", tmp_path / "empty-vendor")
    with pytest.raises(w.WhisperCppError) as excinfo:
        w.ensure_whisper_binary()
    assert "submodule" in str(excinfo.value)


def test_ensure_model_uses_existing_file(monkeypatch, tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "ggml-small.bin").write_text("weights")
    monkeypatch.setattr(w, "MODEL_CACHE", cache)
    called = {"download": False}
    monkeypatch.setattr(w.urllib.request, "urlretrieve",
                        lambda *a, **k: called.__setitem__("download", True))
    path = w._ensure_model_file("small")
    assert path == cache / "ggml-small.bin"
    assert called["download"] is False


def test_transcribe_runs_cli_and_parses(monkeypatch, tmp_path):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"\x00" * 100)
    monkeypatch.setattr(w, "ensure_whisper_binary", lambda: Path("/bin/whisper-cli"))
    monkeypatch.setattr(w, "_ensure_model_file", lambda m: tmp_path / "m.bin")

    def fake_run(cmd, **kwargs):
        out_prefix = Path(cmd[cmd.index("-of") + 1])
        out_prefix.with_suffix(".json").write_text(
            '{"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": " hi"}]}'
        )
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(w.subprocess, "run", fake_run)
    result = w.WhisperCppBackend().transcribe(wav, model="small", language="en")
    assert result["segments"][0]["text"] == "hi"
    assert result["segments"][0]["end"] == 1.0


def test_transcribe_raises_on_nonzero_exit(monkeypatch, tmp_path):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"\x00" * 100)
    monkeypatch.setattr(w, "ensure_whisper_binary", lambda: Path("/bin/whisper-cli"))
    monkeypatch.setattr(w, "_ensure_model_file", lambda m: tmp_path / "m.bin")

    def fake_run(cmd, **kwargs):
        class R:
            returncode = 1
            stderr = "vulkan device lost"
        return R()

    monkeypatch.setattr(w.subprocess, "run", fake_run)
    with pytest.raises(w.WhisperCppError):
        w.WhisperCppBackend().transcribe(wav, model="small", language=None)


def test_transcribe_wraps_oserror_from_subprocess(monkeypatch, tmp_path):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"\x00" * 100)
    monkeypatch.setattr(w, "ensure_whisper_binary", lambda: Path("/bin/whisper-cli"))
    monkeypatch.setattr(w, "_ensure_model_file", lambda m: tmp_path / "m.bin")

    def boom(cmd, **kwargs):
        raise OSError("not executable")

    monkeypatch.setattr(w.subprocess, "run", boom)
    with pytest.raises(w.WhisperCppError):
        w.WhisperCppBackend().transcribe(wav, model="small", language="en")
