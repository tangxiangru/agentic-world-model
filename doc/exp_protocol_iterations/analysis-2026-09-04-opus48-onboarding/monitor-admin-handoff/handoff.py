"""One-shot handoff after the existing detector exits normally on known retirements.

Never kills the old process, queries Slurm early, releases jobs, or analyses data.
Unexpected state leaves the old ready file intact for planner attention.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

ROOT = Path("/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator")
DIRECTORY = Path(__file__).resolve().parent
STATE = ROOT / "data/ptb/monitor/exp_protocol_goal.json"
OLD_PID = 3564003
RETIRED = {"91046","91047","91048","91049","91050","91051","91052","91053",
           "91058","91059","91068","91069","91070","91071","91072","91073","91965"}
REMAINING = {str(i) for i in range(90826, 90831)} | {str(i) for i in range(92125, 92141)}

def read(path):
    return json.loads(path.read_text())

def write(name, value):
    with (DIRECTORY / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")

def check_ready(state):
    assert state["monitor_pid"] == OLD_PID and state["status"] == "ready"
    assert set(state["watched_jobs"]) == RETIRED | REMAINING
    assert set(state["terminal_jobs"]) == RETIRED and state["terminal_count"] == 17
    assert all(state["states"][job] == "CANCELLED" for job in RETIRED)
    assert state["threshold"] == 8

def check_harvest():
    audit = read(ROOT / "doc/exp_protocol_iterations/analysis-2026-09-04-opus48-onboarding/legacy-retirement-postflight.json")
    assert set(audit["retire_ids"]) == RETIRED
    seen = set()
    for row in audit["records"]:
        if not row["retire"]:
            continue
        receipt_path = Path(row["receipt"]).resolve()
        assert receipt_path.is_relative_to(ROOT / "results/ptb")
        receipt = read(receipt_path)
        jobs = [job for job in receipt["jobs"] if job["job_id"] == row["job_id"]]
        assert len(jobs) == 1
        job = jobs[0]
        status = read(receipt_path.parent / job["cell_id"] / "status.json")
        assert status["job_id"] == row["job_id"] and status["slurm_state"] == "CANCELLED"
        assert status["complete"] is False and status["eligible"] is False
        assert status["accuracy"] is None and status["result_dir"] is None
        assert any(c["job_id"] == job["job_id"] and c.get("pending_only") is True
                   and c["state_before"] == "PENDING" and c["state_after"] == "CANCELLED"
                   for c in receipt["cancellations"])
        seen.add(row["job_id"])
    assert seen == RETIRED

def main():
    check_harvest()
    descriptor = os.pidfd_open(OLD_PID)
    command = Path(f"/proc/{OLD_PID}/cmdline").read_bytes().split(b"\0")
    assert b"tools/exp_protocol_completion_monitor.py" in command
    assert command[command.index(b"--poll-seconds") + 1] == b"3600"
    assert set(command[command.index(b"--jobs") + 1].decode().split(",")) == RETIRED | REMAINING
    initial = read(STATE)
    assert initial["monitor_pid"] == OLD_PID and initial["status"] == "watching"
    assert set(initial["watched_jobs"]) == RETIRED | REMAINING
    write("waiting.json", {"old_pid": OLD_PID, "helper_pid": os.getpid(),
          "started_at": datetime.now(timezone.utc).isoformat(), "old_state": initial})
    print("WAITING_ON_VERIFIED_OLD_PID", OLD_PID, flush=True)
    waiter = select.poll()
    waiter.register(descriptor, select.POLLIN)
    waiter.poll()  # Background wait on the actual process handle, not a stale file.
    os.close(descriptor)
    previous = read(STATE)
    check_ready(previous)
    check_harvest()
    write("previous-ready-state.json", previous)
    # Do not overwrite another operator's monitor if the state owner changed.
    assert read(STATE) == previous
    log = (DIRECTORY / "new-monitor.log").open("x")
    monitor = subprocess.Popen([sys.executable, str(ROOT / "tools/exp_protocol_completion_monitor.py"),
        "--jobs", ",".join(sorted(REMAINING)), "--threshold", "8", "--poll-seconds", "3600",
        "--state-file", str(STATE)], cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if monitor.poll() is not None:
            raise RuntimeError("new detector terminated during handoff; inspect its log/state")
        current = read(STATE)
        if current.get("monitor_pid") == monitor.pid:
            assert set(current["watched_jobs"]) == REMAINING
            assert current["status"] == "watching"
            record = {"old_pid": OLD_PID, "new_pid": monitor.pid,
                      "old_terminal_count": 17, "administrative_clean_cells": 0,
                      "new_state": current, "completed_at": datetime.now(timezone.utc).isoformat()}
            write("completed.json", record)
            print(json.dumps(record, indent=2), flush=True)
            return
        time.sleep(.2)
    raise RuntimeError("new detector first tick unconfirmed; inspect the same handle, do not restart")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write("error.json", {"error_type": type(exc).__name__, "message": str(exc),
                             "at": datetime.now(timezone.utc).isoformat()})
        raise
