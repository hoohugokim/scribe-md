"""Tests for capture subprocess lifecycle management."""

import io
import subprocess

import pytest

from scribe_md import capture


class FakeProc:
    """A stand-in for subprocess.Popen for testing reaping logic."""

    def __init__(self, alive=True, wait_raises=0):
        self._alive = alive
        self._wait_raises = wait_raises  # how many wait() calls should time out
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self.stdout = io.BytesIO(b"")

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls <= self._wait_raises:
            raise subprocess.TimeoutExpired(cmd="capture", timeout=timeout)
        self._alive = False
        return 0


def test_terminate_capture_terminates_live_proc():
    proc = FakeProc(alive=True)
    capture.terminate_capture(proc)
    assert proc.terminated is True
    assert proc.killed is False
    assert proc.stdout.closed is True


def test_terminate_capture_kills_when_terminate_times_out():
    proc = FakeProc(alive=True, wait_raises=1)  # first wait() times out
    capture.terminate_capture(proc)
    assert proc.terminated is True
    assert proc.killed is True
    assert proc.stdout.closed is True


def test_terminate_capture_noop_when_already_exited():
    proc = FakeProc(alive=False)
    capture.terminate_capture(proc)
    assert proc.terminated is False
    assert proc.killed is False
    # Even an already-dead process gets its stdout pipe closed.
    assert proc.stdout.closed is True


def test_terminate_capture_tolerates_missing_stdout():
    proc = FakeProc(alive=True)
    proc.stdout = None
    capture.terminate_capture(proc)  # must not raise
    assert proc.terminated is True
