"""Bounded provider check using the exact CLI extracted from the pinned image.

This is not an Apptainer or GPU acceptance test. The record explicitly names
its extracted-CLI/host-runtime scope; no scientist task or benchmark is sent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL = "claude-opus-4-8[1m]"
CANONICAL = "claude-opus-4-8"
IMAGE_SHA = "35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759"
IMAGE_CLI = "usr/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(raw, returncode):
    events = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(value, dict):
            events.append(value)
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    usage = result.get("modelUsage") or {}
    resolved, model_usage = next(iter(usage.items()), (None, {}))
    context = model_usage.get("contextWindow")
    canonical = model_usage.get("canonicalModel")
    allowed = {MODEL, CANONICAL}
    observed_tool_use = any(
        block.get("type") == "tool_use"
        for event in events
        for block in (event.get("message", {}).get("content", []) if isinstance(event.get("message"), dict) else [])
        if isinstance(block, dict)
    )
    verified = (
        returncode == 0 and bool(result) and not result.get("is_error")
        and result.get("api_error_status") in (None, 0)
        and len(usage) == 1 and resolved in allowed
        and canonical in (None, *allowed)
        and type(context) is int and context == 1_000_000
        and init.get("claude_code_version") == "2.1.219"
        and str(result.get("result", "")).strip() == "OK"
        and not observed_tool_use
    )
    return {
        "verified": verified, "requested_model": MODEL,
        "resolved_model": resolved, "canonical_model": canonical,
        "requested_context_tokens": 1_000_000, "resolved_context_tokens": context,
        "cli_version": init.get("claude_code_version"),
        "terminal_reason": result.get("terminal_reason"),
        "api_error_status": result.get("api_error_status"),
        "result_subtype": result.get("subtype"), "is_error": result.get("is_error"),
        "tool_use_observed": observed_tool_use,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bwrap", default="bwrap")
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    (output / "config").mkdir(mode=0o700)
    cli = args.cli.resolve(strict=True)
    image = args.image.resolve(strict=True)
    if sha_file(image) != IMAGE_SHA:
        raise SystemExit("pinned image SHA256 mismatch; no provider call made")
    # Compare the executable with bytes read directly from the actual image.
    digest = hashlib.sha256()
    extraction = subprocess.Popen(
        ["unsquashfs", "-o", "57344", "-cat", str(image), IMAGE_CLI],
        stdout=subprocess.PIPE,
    )
    for block in iter(lambda: extraction.stdout.read(1024 * 1024), b""):
        digest.update(block)
    extraction.stdout.close()
    if extraction.wait() or digest.hexdigest() != sha_file(cli):
        raise SystemExit("extracted CLI differs from pinned image bytes; no provider call made")

    # Vertex uses its own credentials. Do not inherit unrelated API keys,
    # custom Anthropic endpoints, tool settings or default-model overrides.
    names = ("PATH", "LANG", "LC_ALL", "TZ", "USER", "LOGNAME",
             "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_CONFIG",
             "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(CLAUDE_CODE_USE_VERTEX="1", ANTHROPIC_VERTEX_PROJECT_ID="sercan-v1",
               ANTHROPIC_VERTEX_REGION="global", VERTEX_REGION_CLAUDE_4_8_OPUS="global",
               CLAUDE_CODE_EFFORT_LEVEL="high", CUDA_VISIBLE_DEVICES="",
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    command = [
        args.bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--bind", str(output), str(output), "--chdir", str(output), "--die-with-parent",
        "--setenv", "CLAUDE_CONFIG_DIR", str(output / "config"),
        str(cli), "--print", "--verbose", "--output-format", "stream-json",
        "--model", MODEL, "--effort", "high", "--setting-sources", "",
        "--safe-mode", "--no-session-persistence", "--permission-mode", "plan",
        "--max-budget-usd", "0.10", "--tools", "", "--", "Reply with exactly OK.",
    ]
    raw_path = output / "stream.json"
    started = time.monotonic()
    timed_out = False
    with raw_path.open("xb") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    record = {
        "schema_version": 1, **summarize(raw_path.read_bytes(), returncode),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "provider": "vertex", "project": "sercan-v1", "region": "global", "effort": "high",
        "container": image.name, "container_sha256": IMAGE_SHA,
        "cli_image_path": IMAGE_CLI, "cli_sha256": digest.hexdigest(),
        "execution_mode": "exact_extracted_image_cli_host_libraries_readonly_bwrap",
        "full_container_execution": False, "slurm_job_id": None, "gpu_ids": None,
        "observed_returncode": returncode, "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "raw_trace": str(raw_path), "raw_trace_sha256": sha_file(raw_path),
        "command": command,
    }
    with (output / "record.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))
    return 0 if record["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
