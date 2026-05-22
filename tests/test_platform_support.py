"""Tests for the platform_support module — OS detection and install hints."""

import scribe_md.platform_support as ps


def test_is_macos_true_on_darwin(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    assert ps.is_macos() is True
    assert ps.is_linux() is False


def test_is_linux_true_on_linux(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert ps.is_linux() is True
    assert ps.is_macos() is False


def test_ffmpeg_hint_is_apt_on_linux(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert "apt" in ps.ffmpeg_install_hint()


def test_ffmpeg_hint_is_brew_on_macos(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    assert "brew" in ps.ffmpeg_install_hint()
