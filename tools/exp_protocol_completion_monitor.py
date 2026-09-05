#!/usr/bin/env python3
"""Wait for a batch-sized set of Slurm jobs to become terminal.

This process is deliberately read-only with respect to Slurm.  It writes one
small state file on the shared data volume so a resumed Codex goal can harvest,
validate, and analyse the completed attempts through the normal operator path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_name(raw: str) -> str:
    return raw.strip().split()[0].removesuffix("+") if raw.strip() else "UNKNOWN"


def read_states(job_ids: list[str]) -> dict[str, str]:
    result = subprocess.run(
        [
            "sacct",
            "-nX",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State",
            "--parsable2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sacct failed")
    wanted = set(job_ids)
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 2:
            continue
        job_id = fields[0].split(".", 1)[0]
        if job_id in wanted and job_id not in states:
            states[job_id] = _state_name(fields[1])
    return {job_id: states.get(job_id, "UNKNOWN") for job_id in job_ids}


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="comma-separated Slurm job ids")
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=3600)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("data/ptb/monitor/exp_protocol_goal.json"),
    )
    parser.add_argument("--max-polls", type=int, default=0, help="0 means run until ready")
    args = parser.parse_args()

    jobs = list(dict.fromkeys(item.strip() for item in args.jobs.split(",") if item.strip()))
    if not jobs:
        parser.error("--jobs must contain at least one job id")
    if args.threshold < 1 or args.threshold > len(jobs):
        parser.error("--threshold must be between 1 and the number of watched jobs")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")

    polls = 0
    while True:
        polls += 1
        states = read_states(jobs)
        terminal = [job_id for job_id in jobs if states[job_id] in TERMINAL_STATES]
        ready = len(terminal) >= args.threshold
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": "ready" if ready else "watching",
            "checked_at": _now(),
            "monitor_pid": os.getpid(),
            "threshold": args.threshold,
            "watched_jobs": jobs,
            "states": states,
            "terminal_jobs": terminal,
            "terminal_count": len(terminal),
            "next_action": (
                "Run awm ptb reconcile --apply, commit receipt-backed harvests, count clean new "
                "cells, then invoke the local Claude Opus 5 max trace review if the clean window "
                "reaches eight."
                if ready
                else "Wait for the monitor event; do not poll Slurm faster than this process."
            ),
        }
        write_state(args.state_file, payload)
        if ready:
            print(json.dumps(payload, sort_keys=True), flush=True)
            return 0
        if args.max_polls and polls >= args.max_polls:
            print(json.dumps(payload, sort_keys=True), flush=True)
            return 3
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
