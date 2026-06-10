import os

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


def _stub_probe(monkeypatch, returncode):
    """Make every GPU probe subprocess return *returncode*."""
    class R:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: R(returncode))


def test_detect_accel_prefers_cuda_when_toolkit_present_and_device_works(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/" + name
                        if name in ("nvcc", "nvidia-smi") else None)
    _stub_probe(monkeypatch, 0)
    assert w.detect_accel() == "cuda"


def test_detect_accel_falls_back_to_vulkan_when_device_works(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/vulkaninfo"
                        if name == "vulkaninfo" else None)
    _stub_probe(monkeypatch, 0)
    assert w.detect_accel() == "vulkan"


def test_detect_accel_cpu_when_nothing_present(monkeypatch):
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: None)
    assert w.detect_accel() == "cpu"


def test_detect_accel_cuda_requires_working_probe(monkeypatch):
    # nvcc + nvidia-smi installed, but no usable GPU/driver (probe exits non-zero).
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/" + name
                        if name in ("nvcc", "nvidia-smi") else None)
    _stub_probe(monkeypatch, 1)
    assert w.detect_accel() == "cpu"


def test_detect_accel_vulkan_requires_working_probe(monkeypatch):
    # vulkan-tools installed by pixi, but headless box has no usable device.
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/vulkaninfo"
                        if name == "vulkaninfo" else None)
    _stub_probe(monkeypatch, 1)
    assert w.detect_accel() == "cpu"


def test_detect_accel_vulkan_cpu_when_probe_unavailable(monkeypatch):
    # vulkaninfo on PATH but raising (e.g. missing shared lib) must not crash.
    monkeypatch.delenv("SCRIBE_MD_WHISPER_ACCEL", raising=False)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/vulkaninfo"
                        if name == "vulkaninfo" else None)

    def boom(*a, **k):
        raise OSError("cannot exec")

    monkeypatch.setattr(w.subprocess, "run", boom)
    assert w.detect_accel() == "cpu"


def test_describe_reports_built_accel(monkeypatch, tmp_path):
    # describe() must report what the binary was actually built with, not a
    # freshly re-detected accel (which could differ and re-runs the probe).
    marker = tmp_path / ".scribe_accel"
    marker.write_text("vulkan")
    monkeypatch.setattr(w, "ACCEL_MARKER", marker)
    assert w.WhisperCppBackend().describe() == "whisper.cpp (vulkan)"


def test_describe_falls_back_to_detect_without_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "ACCEL_MARKER", tmp_path / "absent")
    monkeypatch.setattr(w, "detect_accel", lambda: "cpu")
    assert w.WhisperCppBackend().describe() == "whisper.cpp (cpu)"


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


# ---------------------------------------------------------------------------
# Accelerator marker: rebuild when the cached binary's accel no longer matches
# ---------------------------------------------------------------------------


def _setup_build_dirs(monkeypatch, tmp_path):
    """Point the module's build paths at a throwaway tmp build tree."""
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    binary = build / "bin" / "whisper-cli"
    marker = build / ".scribe_accel"
    monkeypatch.setattr(w, "BUILD_DIR", build)
    monkeypatch.setattr(w, "WHISPER_BIN", binary)
    monkeypatch.setattr(w, "ACCEL_MARKER", marker)
    return build, binary, marker


def _no_build_run(monkeypatch):
    """Record subprocess invocations and fail the test if a build is launched."""
    calls = []
    monkeypatch.setattr(w.subprocess, "run",
                        lambda *a, **k: calls.append(a) or _fail_if_called())
    return calls


def _fail_if_called():
    raise AssertionError("subprocess.run should not have been called")


def test_ensure_binary_keeps_binary_when_marker_matches(monkeypatch, tmp_path):
    _, binary, marker = _setup_build_dirs(monkeypatch, tmp_path)
    binary.write_text("bin")
    marker.write_text("vulkan")
    monkeypatch.setattr(w, "detect_accel", lambda: "vulkan")
    _no_build_run(monkeypatch)
    assert w.ensure_whisper_binary() == binary


def test_ensure_binary_keeps_legacy_binary_without_marker(monkeypatch, tmp_path):
    # A binary built by an older version (or `pixi run build-whisper`) has no
    # marker; we must not force a surprise rebuild on every run.
    _, binary, marker = _setup_build_dirs(monkeypatch, tmp_path)
    binary.write_text("bin")
    monkeypatch.setattr(w, "detect_accel", lambda: "cuda")
    _no_build_run(monkeypatch)
    assert w.ensure_whisper_binary() == binary


def test_ensure_binary_rebuilds_when_marker_differs(monkeypatch, tmp_path):
    build, binary, marker = _setup_build_dirs(monkeypatch, tmp_path)
    binary.write_text("old")
    marker.write_text("cpu")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "CMakeLists.txt").write_text("")
    monkeypatch.setattr(w, "VENDOR_DIR", vendor)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/cmake")
    monkeypatch.setattr(w, "detect_accel", lambda: "vulkan")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("new")

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr(w.subprocess, "run", fake_run)

    result = w.ensure_whisper_binary()

    assert result == binary
    assert binary.read_text() == "new"          # a fresh build happened
    assert calls, "expected cmake to be invoked"
    assert marker.read_text() == "vulkan"        # marker updated to new accel


def test_build_records_accel_marker(monkeypatch, tmp_path):
    build, binary, marker = _setup_build_dirs(monkeypatch, tmp_path)
    # No binary yet -> first build.
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "CMakeLists.txt").write_text("")
    monkeypatch.setattr(w, "VENDOR_DIR", vendor)
    monkeypatch.setattr(w.shutil, "which", lambda name: "/usr/bin/cmake")
    monkeypatch.setattr(w, "detect_accel", lambda: "cpu")

    def fake_run(cmd, **kwargs):
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("built")

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr(w.subprocess, "run", fake_run)

    w.ensure_whisper_binary()
    assert marker.read_text() == "cpu"


# ---------------------------------------------------------------------------
# Model download integrity: reject HTML/truncated bodies, atomic unique temp
# ---------------------------------------------------------------------------


def test_ensure_model_rejects_html_body(monkeypatch, tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    monkeypatch.setattr(w, "MODEL_CACHE", cache)

    def fake(url, filename):
        Path(filename).write_text("<html>503 from a proxy</html>")
        return (str(filename), {"Content-Type": "text/html; charset=utf-8"})

    monkeypatch.setattr(w.urllib.request, "urlretrieve", fake)
    with pytest.raises(w.WhisperCppError):
        w._ensure_model_file("small")
    # A poisoned cache entry must NOT be left behind.
    assert not (cache / "ggml-small.bin").exists()
    assert list(cache.glob("*.tmp")) == []


def test_ensure_model_rejects_truncated_body(monkeypatch, tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    monkeypatch.setattr(w, "MODEL_CACHE", cache)

    def fake(url, filename):
        Path(filename).write_bytes(b"\x00" * 1024)  # 1 KiB, far below any real model
        return (str(filename), {"Content-Type": "application/octet-stream"})

    monkeypatch.setattr(w.urllib.request, "urlretrieve", fake)
    with pytest.raises(w.WhisperCppError):
        w._ensure_model_file("small")
    assert not (cache / "ggml-small.bin").exists()


def test_ensure_model_uses_unique_temp_and_renames(monkeypatch, tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    monkeypatch.setattr(w, "MODEL_CACHE", cache)
    seen = {}

    def fake(url, filename):
        seen["tmp"] = Path(filename)
        Path(filename).write_bytes(b"\x00" * (w._MIN_MODEL_BYTES + 10))
        return (str(filename), {"Content-Type": "application/octet-stream"})

    monkeypatch.setattr(w.urllib.request, "urlretrieve", fake)
    path = w._ensure_model_file("small")

    assert path == cache / "ggml-small.bin"
    assert path.stat().st_size > w._MIN_MODEL_BYTES
    # A process-unique temp name (not the shared "<fname>.tmp"), cleaned up.
    assert seen["tmp"] != cache / "ggml-small.bin.tmp"
    assert str(os.getpid()) in seen["tmp"].name
    assert list(cache.glob("*.tmp")) == []


def test_ensure_model_rejects_path_traversal(monkeypatch, tmp_path):
    # A malicious/typo'd model name from an auto-discovered .scribe-md.toml must
    # not let the download escape the model cache.
    cache = tmp_path / "models"
    cache.mkdir()
    monkeypatch.setattr(w, "MODEL_CACHE", cache)

    def must_not_download(*a, **k):
        raise AssertionError("download must not be attempted for a traversal path")

    monkeypatch.setattr(w.urllib.request, "urlretrieve", must_not_download)
    with pytest.raises(w.WhisperCppError):
        w._ensure_model_file("../../../../etc/cron.d/evil")
