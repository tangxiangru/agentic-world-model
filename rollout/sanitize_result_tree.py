#!/usr/bin/env python3
"""Redact credential material from text files in a task or result tree.

Exit status 0 means that the tree was already clean, 3 means at least one file
was redacted, and 2 means that the sanitizer could not safely complete.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    from redact_claude_stream import REDACTED, redact_text
except ModuleNotFoundError:  # Support ``python -m rollout.sanitize_result_tree``.
    from rollout.redact_claude_stream import REDACTED, redact_text


ATTESTATION_NAME = "secret-sanitization.json"
ERROR_EXIT = 2
REDACTED_EXIT = 3
SCHEMA_VERSION = "awm-secret-sanitization-v1"

# These formats are model weights, training state, archives, or other binary
# artifacts. They are intentionally not decoded or rewritten.
KNOWN_BINARY_SUFFIXES = frozenset(
    {
        ".7z",
        ".arrow",
        ".avi",
        ".bin",
        ".bz2",
        ".ckpt",
        ".flac",
        ".ggml",
        ".gguf",
        ".gif",
        ".gz",
        ".h5",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".model",
        ".mov",
        ".mp3",
        ".mp4",
        ".npy",
        ".npz",
        ".onnx",
        ".ot",
        ".parquet",
        ".pb",
        ".png",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".tflite",
        ".webm",
        ".webp",
        ".wav",
        ".xz",
        ".zip",
        ".zst",
    }
)


class SanitizationError(RuntimeError):
    """Raised when a tree cannot be sanitized without following unsafe paths."""


@dataclass
class ScanSummary:
    files_scanned: int = 0
    bytes_scanned: int = 0
    files_redacted: int = 0
    redaction_count: int = 0
    redacted_paths: list[str] = field(default_factory=list)
    skipped: dict[str, list[str]] = field(
        default_factory=lambda: {
            "binary": [],
            "known_binary": [],
            "special": [],
            "symlink": [],
        }
    )

    def skip(self, reason: str, path: str) -> None:
        self.skipped[reason].append(path)

    def attestation(self) -> dict[str, object]:
        skipped = {
            reason: {"count": len(paths), "paths": sorted(paths)}
            for reason, paths in sorted(self.skipped.items())
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "redacted" if self.files_redacted else "clean",
            "root": ".",
            "files_scanned": self.files_scanned,
            "bytes_scanned": self.bytes_scanned,
            "files_redacted": self.files_redacted,
            "redaction_count": self.redaction_count,
            "redacted_paths": sorted(self.redacted_paths),
            "skipped": skipped,
            "redactor_source": "rollout/redact_claude_stream.py",
        }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_root(root_arg: Path) -> Path:
    try:
        root_stat = root_arg.lstat()
    except OSError as exc:
        raise SanitizationError(f"cannot inspect root path: {exc.filename}") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise SanitizationError("root path must not be a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise SanitizationError("root path must be a directory")
    return root_arg.resolve(strict=True)


def _resolve_attestation(root: Path, argument: Path | None) -> Path:
    candidate = root / ATTESTATION_NAME if argument is None else argument
    if not candidate.is_absolute():
        candidate = root / candidate

    # The parent must already exist so that resolution cannot create a path
    # through an uninspected symlink after traversal starts.
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise SanitizationError("attestation parent must be an existing directory") from exc
    resolved = parent / candidate.name
    if not _is_within(resolved, root):
        raise SanitizationError("attestation path must remain inside the sanitized root")

    try:
        existing = resolved.lstat()
    except FileNotFoundError:
        return resolved
    except OSError as exc:
        raise SanitizationError("cannot inspect attestation path") from exc
    if stat.S_ISLNK(existing.st_mode):
        raise SanitizationError("attestation path must not be a symlink")
    if not stat.S_ISREG(existing.st_mode):
        raise SanitizationError("attestation path must be a regular file")
    return resolved


def _atomic_write(path: Path, data: bytes, mode: int, expected: os.stat_result | None) -> None:
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.sanitize-",
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(data)
            output.flush()
            os.fsync(output.fileno())

        if expected is not None:
            current = path.lstat()
            if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
                raise SanitizationError("a scanned file changed type before redaction")
            expected_identity = (
                expected.st_dev,
                expected.st_ino,
                expected.st_size,
                expected.st_mtime_ns,
            )
            current_identity = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
            if current_identity != expected_identity:
                raise SanitizationError("a scanned file changed before redaction")
        elif path.exists() or path.is_symlink():
            raise SanitizationError("attestation path appeared during sanitization")

        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise SanitizationError(f"cannot safely rewrite path: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _sanitize_file(path: Path, relative: str, summary: ScanSummary) -> None:
    try:
        before_stat = path.lstat()
    except OSError as exc:
        raise SanitizationError(f"cannot inspect file: {relative}") from exc
    if stat.S_ISLNK(before_stat.st_mode):
        summary.skip("symlink", relative)
        return
    if not stat.S_ISREG(before_stat.st_mode):
        summary.skip("special", relative)
        return
    if path.suffix.lower() in KNOWN_BINARY_SUFFIXES:
        summary.skip("known_binary", relative)
        return

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SanitizationError(f"cannot read file: {relative}") from exc
    if b"\x00" in raw[:8192]:
        summary.skip("binary", relative)
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        summary.skip("binary", relative)
        return

    summary.files_scanned += 1
    summary.bytes_scanned += len(raw)
    sanitized = redact_text(text)
    if sanitized == text:
        return

    # Every replacement made by the shared redactor inserts one marker. Existing
    # markers are subtracted so the attestation contains counts, never values.
    replacements = sanitized.count(REDACTED) - text.count(REDACTED)
    summary.redaction_count += max(1, replacements)
    summary.files_redacted += 1
    summary.redacted_paths.append(relative)
    _atomic_write(
        path,
        sanitized.encode("utf-8"),
        stat.S_IMODE(before_stat.st_mode),
        before_stat,
    )


def sanitize_tree(root: Path, attestation_path: Path | None = None) -> tuple[dict[str, object], Path]:
    root = _resolve_root(root)
    attestation = _resolve_attestation(root, attestation_path)
    summary = ScanSummary()

    def walk_error(error: OSError) -> None:
        filename = Path(error.filename) if error.filename else root
        try:
            relative = _relative(filename.resolve(strict=False), root)
        except ValueError:
            relative = "."
        raise SanitizationError(f"cannot traverse directory: {relative}") from error

    for directory_name, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        directory = Path(directory_name)
        traversable: list[str] = []
        for name in sorted(directory_names):
            path = directory / name
            relative = _relative(path, root)
            try:
                entry_stat = path.lstat()
            except OSError as exc:
                raise SanitizationError(f"cannot inspect directory entry: {relative}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                summary.skip("symlink", relative)
            elif stat.S_ISDIR(entry_stat.st_mode):
                traversable.append(name)
            else:
                summary.skip("special", relative)
        directory_names[:] = traversable

        for name in sorted(file_names):
            path = directory / name
            if path == attestation:
                continue
            _sanitize_file(path, _relative(path, root), summary)

    document = summary.attestation()
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        previous = attestation.lstat()
    except FileNotFoundError:
        previous = None
    except OSError as exc:
        raise SanitizationError("cannot inspect attestation before writing") from exc
    if previous is not None and (
        stat.S_ISLNK(previous.st_mode) or not stat.S_ISREG(previous.st_mode)
    ):
        raise SanitizationError("attestation path changed type during sanitization")
    _atomic_write(attestation, encoded, 0o600, previous)
    return document, attestation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="task or result directory to sanitize")
    parser.add_argument(
        "--attestation",
        type=Path,
        help=f"attestation path inside root (default: {ATTESTATION_NAME})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        document, attestation = sanitize_tree(arguments.root, arguments.attestation)
    except SanitizationError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return ERROR_EXIT

    output = {
        "status": document["status"],
        "files_redacted": document["files_redacted"],
        "redaction_count": document["redaction_count"],
        "attestation": attestation.name,
    }
    print(json.dumps(output, sort_keys=True))
    return REDACTED_EXIT if document["files_redacted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
