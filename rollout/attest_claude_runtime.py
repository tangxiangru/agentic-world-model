#!/usr/bin/env python3
"""Fail-closed attestations for the Claude CLI and its reported model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class AttestationError(RuntimeError):
    """The runtime did not prove the requested identity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def read_object(path: Path) -> dict[str, Any]:
    if not regular(path):
        raise AttestationError(f"required regular file is missing or linked: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"invalid JSON object in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AttestationError(f"JSON top level is not an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_study_input(path: Path, key: str, value: dict[str, Any]) -> None:
    study = read_object(path)
    if key in study:
        raise AttestationError(f"study input already contains {key!r}: {path}")
    study[key] = value
    atomic_json(path, study)


def install_and_attest_cli(
    version_path: Path, package_version: str, expected_output: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", package_version):
        raise AttestationError("Claude CLI package version is not an exact semantic version")
    if not expected_output or "\n" in expected_output or "\r" in expected_output:
        raise AttestationError("expected Claude CLI version output must be one non-empty line")
    prefix = Path.home() / ".local"
    package = f"@anthropic-ai/claude-code@{package_version}"
    try:
        installed = subprocess.run(
            [
                "npm",
                "install",
                "-g",
                "--prefix",
                str(prefix),
                "--no-fund",
                "--no-audit",
                package,
            ],
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AttestationError(f"exact Claude CLI installation failed: {exc}") from exc
    if installed.returncode != 0:
        raise AttestationError(
            f"exact Claude CLI installation exited {installed.returncode}: {package}"
        )
    binary = shutil.which("claude")
    if not binary:
        raise AttestationError("exact Claude CLI install succeeded but claude is not on PATH")
    expected_binary = (prefix / "bin" / "claude").resolve()
    if Path(binary).resolve() != expected_binary:
        raise AttestationError(
            f"resolved Claude CLI {binary} is not the exact install {expected_binary}"
        )
    try:
        version = subprocess.run(
            [binary, "--version"], text=True, capture_output=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AttestationError(f"Claude CLI version lookup failed: {exc}") from exc
    actual = version.stdout.strip()
    if version.returncode != 0:
        raise AttestationError(
            f"Claude CLI version lookup exited {version.returncode}: {version.stderr.strip()}"
        )
    if actual != expected_output:
        raise AttestationError(
            f"Claude CLI version output {actual!r} does not exactly match expected "
            f"{expected_output!r}"
        )
    version_text = "\n".join(
        (
            "binary: claude",
            "package: @anthropic-ai/claude-code",
            f"package_version: {package_version}",
            f"path: {binary}",
            f"version: {actual}",
            "update: pinned-install",
            "",
        )
    )
    atomic_text(version_path, version_text)
    attestation = {
        "actual_version_output": actual,
        "expected_version_output": expected_output,
        "package": "@anthropic-ai/claude-code",
        "package_version": package_version,
        "resolved_path": binary,
        "update": "pinned-install",
    }
    attestation["version_record_sha256"] = sha256_file(version_path)
    return attestation


def read_stream(path: Path) -> list[dict[str, Any]]:
    if not regular(path) or path.stat().st_size == 0:
        raise AttestationError(f"Claude stream is missing, empty, or linked: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AttestationError(f"Claude stream row {number} is not an object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"invalid Claude stream {path}: {exc}") from exc
    if not rows:
        raise AttestationError(f"Claude stream contains no events: {path}")
    return rows


def reported_models(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        if row.get("type") == "system" and isinstance(row.get("model"), str):
            found.add(row["model"])
        message = row.get("message")
        if isinstance(message, dict) and isinstance(message.get("model"), str):
            found.add(message["model"])
        if row.get("type") == "result":
            usage = row.get("modelUsage") or row.get("model_usage")
            if isinstance(usage, dict):
                found.update(str(model) for model in usage)
    return sorted(found)


def attest_vertex_provider(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Require Claude's own telemetry to prove Vertex-only execution."""
    sources = {
        row.get("apiKeySource")
        for row in rows
        if row.get("type") == "system" and row.get("subtype") == "init"
    }
    if sources != {"none"}:
        raise AttestationError(
            "Claude CLI reported a non-Vertex/direct key source: "
            f"{sorted(map(str, sources))}"
        )
    providers: set[str] = set()
    for result in results:
        usage = result.get("modelUsage") or result.get("model_usage")
        if not isinstance(usage, dict) or not usage:
            raise AttestationError("Claude result omitted actual provider telemetry")
        for details in usage.values():
            if not isinstance(details, dict) or not isinstance(details.get("provider"), str):
                raise AttestationError("Claude result omitted actual provider telemetry")
            providers.add(details["provider"])
    if providers != {"vertex"}:
        raise AttestationError(
            f"Claude actual provider is not exactly Vertex: {sorted(providers)}"
        )
    return sorted(str(source) for source in sources), sorted(providers)


def attest_model(
    stream_path: Path, requested_alias: str, expected_model_id: str
) -> dict[str, Any]:
    if not requested_alias or not expected_model_id:
        raise AttestationError("requested alias and expected provider model ID are required")
    if re.search(r"\s", expected_model_id):
        raise AttestationError("expected provider model ID may not contain whitespace")
    rows = read_stream(stream_path)
    results = [row for row in rows if row.get("type") == "result"]
    if not results:
        raise AttestationError("Claude stream has no result event")
    for result in results:
        if result.get("is_error") or result.get("subtype") not in (None, "success"):
            raise AttestationError("Claude stream contains a non-success result event")
    reported = reported_models(rows)
    if not reported:
        raise AttestationError("Claude stream did not report its actual model")
    if reported != [expected_model_id]:
        raise AttestationError(
            f"reported Claude models {reported!r} do not exactly match expected "
            f"provider ID {expected_model_id!r} for alias {requested_alias!r}"
        )
    api_key_sources, providers = attest_vertex_provider(rows, results)
    return {
        "api_key_sources": api_key_sources,
        "expected_model_id": expected_model_id,
        "reported_providers": providers,
        "reported_model_ids": reported,
        "requested_alias": requested_alias,
        "stream_sha256": sha256_file(stream_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cli = subparsers.add_parser(
        "install-cli", help="install and attest one exact Claude CLI package version"
    )
    cli.add_argument("--version-file", required=True, type=Path)
    cli.add_argument("--package-version", required=True)
    cli.add_argument("--expected-version-output", required=True)
    model = subparsers.add_parser("model", help="attest model IDs in stream-json")
    model.add_argument("stream", type=Path)
    model.add_argument("--requested-alias", required=True)
    model.add_argument("--expected-model-id", required=True)
    for command in (cli, model):
        command.add_argument("--record", required=True, type=Path)
        command.add_argument("--study-input", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "install-cli":
            key = "claude_cli"
            attestation = install_and_attest_cli(
                args.version_file, args.package_version, args.expected_version_output
            )
        else:
            key = "scientist_model"
            attestation = attest_model(
                args.stream, args.requested_alias, args.expected_model_id
            )
        atomic_json(args.record, attestation)
        update_study_input(args.study_input, key, attestation)
    except (AttestationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(attestation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
