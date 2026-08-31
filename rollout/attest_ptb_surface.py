#!/usr/bin/env python3
"""Write or verify the derived, site-independent PTB study surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


SCHEMA = "awm-ptb-study-surface-v1"
HEX40 = re.compile(r"[0-9a-fA-F]{40}")
FIXED_FILES = (
    "src/run_task.sh",
    "src/commit_utils/set_env_vars.sh",
    "src/commit_utils/pin_src_locally.sh",
    "src/commit_utils/attest_study_surface.py",
    "src/eval/general/prompt_fulltraj.txt",
    "src/eval/general/prompt_wm.txt",
    "src/eval/general/prompt_wm_fulltraj.txt",
    "src/eval/general/prompt_wm_smoke.txt",
    "src/eval/general/prompt_wm_fulltraj_smoke.txt",
    "src/eval/tasks/gsm8k/test_data.json",
)
AGENTS = ("hv_recipe", "hv_noop", "claude_fulltraj_noawm", "claude_wm")


class SurfaceError(RuntimeError):
    """The derived private-checkout surface is invalid or changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, awm_commit: str, ptb_commit: str) -> dict:
    if not HEX40.fullmatch(awm_commit) or not HEX40.fullmatch(ptb_commit):
        raise SurfaceError("AWM and PTB revisions must be full 40-hex commits")
    relative_files = set(FIXED_FILES)
    for agent in AGENTS:
        agent_root = root / "agents" / agent
        if agent_root.is_symlink() or not agent_root.is_dir():
            raise SurfaceError(f"study agent is missing, linked, or invalid: {agent_root}")
        for path in agent_root.rglob("*"):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise SurfaceError(f"study agent contains linked/special content: {path}")
            if path.is_file():
                relative_files.add(path.relative_to(root).as_posix())
    records = []
    for relative in sorted(relative_files):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise SurfaceError(f"study surface file is missing, linked, or invalid: {path}")
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "schema_version": SCHEMA,
        "awm_commit": awm_commit.lower(),
        "ptb_commit": ptb_commit.lower(),
        "files": records,
    }


def canonical(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--awm-commit", required=True)
    parser.add_argument("--ptb-commit", required=True)
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        expected = canonical(inventory(root, args.awm_commit, args.ptb_commit))
        if args.mode == "write":
            write_atomic(args.manifest, expected)
        else:
            if args.manifest.is_symlink() or not args.manifest.is_file():
                raise SurfaceError(f"study surface manifest is missing or linked: {args.manifest}")
            if args.manifest.read_bytes() != expected:
                raise SurfaceError("private PTB study surface differs from its setup manifest")
        print(hashlib.sha256(expected).hexdigest())
    except (OSError, SurfaceError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
