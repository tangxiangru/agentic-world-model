from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
REDACTOR = REPO / "rollout" / "redact_claude_stream.py"


def _load_redactor():
    spec = importlib.util.spec_from_file_location("awm_stream_redactor", REDACTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_is_private_and_byte_identical_to_forwarded_redacted_stream(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "scientist-stream.jsonl"
    source = (
        '{"type":"result","token":"secret-value","text":"safe λ"}\n'
        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456\n"
    )
    result = subprocess.run(
        [sys.executable, str(REDACTOR), "--capture", str(capture)],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_bytes() == result.stdout.encode()
    assert "secret-value" not in result.stdout
    assert "abcdefghijklmnopqrstuvwxyz" not in result.stdout
    metadata = capture.stat()
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1


@pytest.mark.parametrize("existing_kind", ["file", "symlink"])
def test_capture_refuses_preexisting_path(
    tmp_path: Path, existing_kind: str
) -> None:
    capture = tmp_path / "scientist-stream.jsonl"
    protected = tmp_path / "protected"
    protected.write_text("unchanged")
    if existing_kind == "file":
        capture.write_text("existing")
    else:
        capture.symlink_to(protected)

    result = subprocess.run(
        [sys.executable, str(REDACTOR), "--capture", str(capture)],
        input="safe\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    if existing_kind == "file":
        assert capture.read_text() == "existing"
    else:
        assert capture.is_symlink()
        assert protected.read_text() == "unchanged"


def test_write_all_survives_nonblocking_pipe_backpressure() -> None:
    redactor = _load_redactor()
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    prefix = bytearray()
    fill = b"p" * 4096
    while True:
        try:
            prefix.extend(fill[: os.write(write_fd, fill)])
        except BlockingIOError:
            break

    payload = (b"durable-stream-" * 8192) + b"end"
    received = bytearray()

    def drain() -> None:
        time.sleep(0.05)
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                return
            received.extend(chunk)

    reader = threading.Thread(target=drain)
    reader.start()
    try:
        redactor._write_all(write_fd, payload)
    finally:
        os.close(write_fd)
    reader.join(timeout=5)
    os.close(read_fd)
    assert not reader.is_alive()
    assert received == prefix + payload


def test_capture_restores_shared_ptb_pipe_before_later_shell_output(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "scientist-stream.jsonl"
    read_fd, write_fd = os.pipe()
    # subprocess stdout is dup2'd from write_fd, so both descriptors refer to
    # one open-file description just like the PTB shell's `2>&1` pipe.
    os.set_blocking(write_fd, False)
    process = subprocess.Popen(
        [sys.executable, str(REDACTOR), "--capture", str(capture)],
        stdin=subprocess.PIPE,
        stdout=write_fd,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _stdout, stderr = process.communicate("safe trajectory λ\n", timeout=10)
        assert process.returncode == 0, stderr
        assert os.get_blocking(write_fd)
        later_shell_output = b"post-Claude attestation\n"
        os.write(write_fd, later_shell_output)
    finally:
        os.close(write_fd)
    forwarded = bytearray()
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        forwarded.extend(chunk)
    os.close(read_fd)
    assert bytes(forwarded) == capture.read_bytes() + later_shell_output


def test_write_all_propagates_closed_output() -> None:
    redactor = _load_redactor()
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        with pytest.raises(BrokenPipeError):
            redactor._write_all(write_fd, b"must-not-disappear")
    finally:
        os.close(write_fd)
