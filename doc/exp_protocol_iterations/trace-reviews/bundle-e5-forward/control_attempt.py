"""Independent operator of two already-locked, owned CPU fixture attempts."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path("/tmp/e5-forward-review.35fG5B")
python = "/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/bin/python"
cli = "/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/bin/awm"
card = sys.argv[1]
mode = sys.argv[2]
assert (card, mode) in {("exp-03", "SIGINT"), ("exp-04", "SIGKILL")}
output = root / "outputs" / ("sigint" if mode == "SIGINT" else "observer-killed")
command = [python, cli, "exp_protocol", "run", "--dir", str(root), card]
proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
deadline = time.monotonic() + 10
while not (output / "ready.json").exists() and proc.poll() is None and time.monotonic() < deadline:
    time.sleep(.025)
assert (output / "ready.json").exists(), "fixture did not become ready"
ready = json.loads((output / "ready.json").read_text())
attempts = list((root / "memory/attempts" / card).glob("*/process.json"))
assert len(attempts) == 1
process_file = attempts[0]
identity = json.loads(process_file.read_text())
child = ready["pid"]
assert identity["child_pid"] == child == ready["pgid"]

def owned_state():
    path = Path(f"/proc/{child}")
    if not path.exists():
        return "gone"
    stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
    assert int(stat[19]) == identity["identity"]["start_ticks"], "PID birth changed"
    if stat[0] != "Z":
        assert str(root / "interruptible.py").encode() in (path / "cmdline").read_bytes()
        assert os.getpgid(child) == child
    return stat[0]

report = {"card": card, "signal": mode, "wrapper_pid": proc.pid, "child_pid": child,
          "process_record": str(process_file), "state_before": owned_state()}
proc.send_signal(getattr(signal, mode))
stdout, stderr = proc.communicate(timeout=8)
report.update(wrapper_returncode=proc.returncode, stdout=stdout, stderr=stderr)
report["finish_exists"] = (process_file.parent / "finish.json").exists()
if mode == "SIGKILL":
    retry = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=4)
    report["same_card_retry"] = {"returncode": retry.returncode,
                                 "stdout": retry.stdout, "stderr": retry.stderr}
    report["owned_child_after_observer_death"] = owned_state()
    if owned_state() not in ("gone", "Z"):
        os.killpg(child, signal.SIGTERM)
        report["cleanup_signal"] = "SIGTERM to receipt-matched, birth-checked owned child group"
deadline = time.monotonic() + 5
while owned_state() not in ("gone", "Z") and time.monotonic() < deadline:
    time.sleep(.025)
report["child_terminal_state"] = owned_state()
assert report["child_terminal_state"] in ("gone", "Z"), "owned child remains active"
print(json.dumps(report, indent=2))
