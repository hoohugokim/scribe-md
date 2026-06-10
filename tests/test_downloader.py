"""Tests for yt-dlp error handling."""

import subprocess

import pytest

from scribe_md import downloader
from scribe_md.downloader import DownloadError


def test_get_video_info_wraps_yt_dlp_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr="private video",
        )

    monkeypatch.setattr(downloader.subprocess, "run", fail)

    with pytest.raises(DownloadError, match="private video"):
        downloader.get_video_info("https://example.invalid/video")


def test_get_playlist_entries_wraps_invalid_json(monkeypatch):
    result = subprocess.CompletedProcess(
        args=["yt-dlp"],
        returncode=0,
        stdout="{bad json}\n",
    )
    monkeypatch.setattr(downloader.subprocess, "run", lambda *a, **k: result)

    with pytest.raises(DownloadError, match="invalid JSON"):
        downloader.get_playlist_entries("https://example.invalid/playlist")


from pathlib import Path


def _fake_ytdlp_download(monkeypatch):
    """Stub the yt-dlp audio download to just create the expected .wav."""
    def fake_run(args, action, *, capture_output=True):
        out = args[args.index("-o") + 1]
        Path(out.replace("%(ext)s", "wav")).write_bytes(b"\x00" * 16)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(downloader, "_run_ytdlp", fake_run)


def test_download_audio_uses_given_title_without_metadata_call(monkeypatch, tmp_path):
    # When the caller already knows the title (e.g. from a playlist entry),
    # download_audio must NOT make an extra get_video_info network call.
    def boom(url):
        raise AssertionError("get_video_info must not be called when title is given")

    monkeypatch.setattr(downloader, "get_video_info", boom)
    _fake_ytdlp_download(monkeypatch)

    path, title = downloader.download_audio("http://x/v", tmp_path, title="My Talk")
    assert title == "My Talk"
    assert path == tmp_path / "My Talk.wav"
    assert path.exists()


def test_download_audio_fetches_title_when_absent(monkeypatch, tmp_path):
    calls = {"info": 0}

    def fake_info(url):
        calls["info"] += 1
        return {"title": "Fetched Title"}

    monkeypatch.setattr(downloader, "get_video_info", fake_info)
    _fake_ytdlp_download(monkeypatch)

    path, title = downloader.download_audio("http://x/v", tmp_path)
    assert calls["info"] == 1
    assert title == "Fetched Title"
    assert path == tmp_path / "Fetched Title.wav"
