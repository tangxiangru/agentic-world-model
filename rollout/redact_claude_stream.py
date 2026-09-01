#!/usr/bin/env python3
"""Redact credential material from a Claude stream without changing its events.

Claude Code's Bash tool receives a short-lived messaging token, and a scientist
can also print credential files that happen to be present in a shared cache.
The model has already seen a tool result before it is emitted on stdout, so this
filter only changes the retained/published trajectory.  It accepts JSONL stream
events, recursively redacts sensitive fields and strings, and passes malformed
diagnostic lines through the same string scrubber.  With ``--capture`` it also
creates a private durable copy while forwarding identical bytes to stdout; its
writer tolerates a shared PTB pipe being switched to nonblocking mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import stat
import sys
from pathlib import Path
from typing import Any, TextIO

REDACTED = "<redacted>"
SENSITIVE_FIELDS = {
    "access_token",
    "api_key",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "hf_token",
    "id_token",
    "oauth_token",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}
ASSIGNMENT = re.compile(
    r"(?im)(?P<prefix>(?<![A-Za-z0-9_])(?:export[ \t]+)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*"
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|AUTHORIZATION))"
    r"[ \t]*=[ \t]*)[^\s,;]+"
)
JSONISH_SECRET = re.compile(
    r"(?i)(?P<prefix>[\"']?(?:access_token|api_key|authorization|client_secret|"
    r"hf_token|id_token|oauth_token|password|passwd|private_key|refresh_token|"
    r"secret|token)[\"']?[ \t]*[:=][ \t]*[\"']?)[^\"'\r\n,}]+"
)
BEARER = re.compile(r"(?i)(?P<prefix>authorization[ \t]*:[ \t]*bearer[ \t]+)\S+")
HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")
OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
GOOGLE_ACCESS_TOKEN = re.compile(r"\bya29\.[A-Za-z0-9._~-]{16,}\b")
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
GOOGLE_API_KEY = re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")
GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
SLACK_TOKEN = re.compile(r"\bxox[a-z]-[A-Za-z0-9-]{16,}\b", re.IGNORECASE)
AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----.*?"
    r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)


def redact_text(value: str) -> str:
    value = ASSIGNMENT.sub(lambda match: match.group("prefix") + REDACTED, value)
    value = JSONISH_SECRET.sub(lambda match: match.group("prefix") + REDACTED, value)
    value = BEARER.sub(lambda match: match.group("prefix") + REDACTED, value)
    value = HF_TOKEN.sub(REDACTED, value)
    value = OPENAI_STYLE_KEY.sub(REDACTED, value)
    value = GOOGLE_ACCESS_TOKEN.sub(REDACTED, value)
    value = JWT.sub(REDACTED, value)
    value = GOOGLE_API_KEY.sub(REDACTED, value)
    value = GITHUB_TOKEN.sub(REDACTED, value)
    value = SLACK_TOKEN.sub(REDACTED, value)
    value = AWS_ACCESS_KEY.sub(REDACTED, value)
    return PRIVATE_KEY.sub(REDACTED, value)


def sensitive_field(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalised = key.strip().lower().replace("-", "_")
    return normalised in SENSITIVE_FIELDS or normalised.endswith(
        (
            "_access_token",
            "_api_key",
            "_auth_token",
            "_client_secret",
            "_messaging_token",
            "_oauth_token",
            "_password",
            "_private_key",
            "_refresh_token",
            "_secret",
            "_token",
        )
    )


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[Any, Any] = {}
        for key, child in value.items():
            if sensitive_field(key) and isinstance(child, str):
                clean[key] = REDACTED
            else:
                clean[key] = redact(child)
        return clean
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def filter_stream(source: TextIO, destination: TextIO) -> None:
    for raw_line in source:
        line = raw_line.rstrip("\n")
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            rendered = redact_text(line)
        else:
            rendered = json.dumps(redact(event), ensure_ascii=False, separators=(",", ":"))
        destination.write(rendered + "\n")
        destination.flush()


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte, tolerating a shared pipe being flipped nonblocking.

    Claude/Node may change ``O_NONBLOCK`` on an inherited stderr descriptor.
    The outer PTB wrapper merges that descriptor with this process's stdout,
    so the flag can change while the stream is running.  Retrying EAGAIN is
    therefore part of the durable trajectory contract, not an optimization.
    """

    remaining = memoryview(data)
    while remaining:
        try:
            written = os.write(fd, remaining)
        except InterruptedError:
            continue
        except BlockingIOError:
            while True:
                try:
                    _readable, writable, _exceptional = select.select([], [fd], [fd])
                except InterruptedError:
                    continue
                if writable:
                    break
                raise OSError("trajectory output closed while waiting for backpressure")
            continue
        if written <= 0:
            raise OSError("trajectory output accepted zero bytes")
        remaining = remaining[written:]


class _CaptureWriter:
    """Text sink that writes identical UTF-8 bytes to a file and stdout."""

    def __init__(self, path: Path):
        if not path.is_absolute() or not path.name:
            raise ValueError("--capture must name an absolute file path")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        parent_fd = os.open(path.parent, directory_flags)
        try:
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_CLOEXEC"):
                file_flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            self._fd = os.open(path.name, file_flags, 0o600, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        os.fchmod(self._fd, 0o600)
        metadata = os.fstat(self._fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            os.close(self._fd)
            raise OSError("capture is not a private, singly-linked regular file")
        self._closed = False

    def write(self, value: str) -> int:
        data = value.encode("utf-8")
        _write_all(self._fd, data)
        _write_all(sys.stdout.fileno(), data)
        return len(value)

    def flush(self) -> None:
        # Both destinations use unbuffered os.write calls. Durability is
        # established once at EOF so a long session does not fsync every event.
        return None

    def close(self) -> None:
        if self._closed:
            return
        try:
            os.fsync(self._fd)
        finally:
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> "_CaptureWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture",
        type=Path,
        help="atomically create this private file and mirror the redacted stream to stdout",
    )
    args = parser.parse_args(argv)
    if args.capture is None:
        filter_stream(sys.stdin, sys.stdout)
    else:
        # stdout is the same outer PTB pipe open-file description inherited by
        # the parent agent shell. Claude/Node can set O_NONBLOCK on that shared
        # description through stderr. Handle it while forwarding, then always
        # restore normal blocking shell semantics after Claude reaches EOF (or
        # capture setup/filtering fails), so later attestations are not lost to
        # EAGAIN.
        try:
            with _CaptureWriter(args.capture) as destination:
                filter_stream(sys.stdin, destination)
        finally:
            os.set_blocking(sys.stdout.fileno(), True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
