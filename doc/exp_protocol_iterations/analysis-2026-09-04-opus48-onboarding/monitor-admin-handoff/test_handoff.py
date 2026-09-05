import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
import json
import subprocess
import sys
import threading
import time

spec = importlib.util.spec_from_file_location("admin_handoff", Path(__file__).with_name("handoff.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.fixture
def state():
    return {"monitor_pid": module.OLD_PID, "status": "ready", "threshold": 8,
            "watched_jobs": list(module.RETIRED | module.REMAINING),
            "terminal_jobs": list(module.RETIRED), "terminal_count": 17,
            "states": {job: "CANCELLED" if job in module.RETIRED else "PENDING"
                       for job in module.RETIRED | module.REMAINING}}

def test_exact_administrative_event(state):
    module.check_ready(state)

@pytest.mark.parametrize("change", ["pid", "watching", "extra_terminal", "missing_terminal", "scientist_failure"])
def test_unknown_event_is_not_consumed(state, change):
    value = deepcopy(state)
    if change == "pid": value["monitor_pid"] += 1
    elif change == "watching": value["status"] = "watching"
    elif change == "extra_terminal": value["terminal_jobs"].append("92125")
    elif change == "missing_terminal": value["terminal_jobs"].pop()
    else: value["states"]["91965"] = "FAILED"
    with pytest.raises(AssertionError): module.check_ready(value)

def test_frozen_retirements_are_harvested():
    module.check_harvest()

def test_real_process_handoff_with_only_synthetic_detectors(tmp_path, monkeypatch):
    """Exercise pidfd/lifecycle locally; neither child queries Slurm or real state."""
    root = tmp_path / "root"
    scripts = root / "tools"
    scripts.mkdir(parents=True)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    state_path = tmp_path / "state.json"
    script = scripts / "exp_protocol_completion_monitor.py"
    script.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--jobs'); p.add_argument('--threshold',type=int)
p.add_argument('--poll-seconds'); p.add_argument('--state-file',type=Path)
a=p.parse_args(); jobs=a.jobs.split(',')
retired=set(''' + repr(sorted(module.RETIRED)) + ''')
state={'monitor_pid':os.getpid(),'status':'watching','threshold':a.threshold,
       'watched_jobs':jobs,'terminal_jobs':[],'terminal_count':0,
       'states':{job:'PENDING' for job in jobs}}
def write():
 t=a.state_file.with_suffix('.tmp'); t.write_text(json.dumps(state)); t.replace(a.state_file)
write()
if len(jobs)==38:
 deadline=time.monotonic()+5
 while not (a.state_file.parent/'release-old').exists() and time.monotonic()<deadline: time.sleep(.01)
 if not (a.state_file.parent/'release-old').exists(): raise SystemExit(3)
 state.update(status='ready',terminal_jobs=sorted(retired),terminal_count=17,
              states={job:'CANCELLED' if job in retired else 'PENDING' for job in jobs})
 write()
else:
 time.sleep(10)
''')
    old = subprocess.Popen([sys.executable, "tools/exp_protocol_completion_monitor.py",
        "--jobs", ",".join(sorted(module.RETIRED | module.REMAINING)), "--threshold", "8",
        "--poll-seconds", "3600", "--state-file", str(state_path)], cwd=root)
    children = []
    real_popen = subprocess.Popen
    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "DIRECTORY", evidence)
    monkeypatch.setattr(module, "STATE", state_path)
    monkeypatch.setattr(module, "OLD_PID", old.pid)
    monkeypatch.setattr(module, "check_harvest", lambda: None)
    monkeypatch.setattr(module.subprocess, "Popen", capture)
    deadline = time.monotonic() + 5
    while not state_path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert state_path.exists()
    def release():
        until = time.monotonic()+4
        while not (evidence/"waiting.json").exists() and time.monotonic()<until:
            time.sleep(.01)
        (tmp_path/"release-old").write_text("synthetic fixture only")
    thread = threading.Thread(target=release)
    thread.start()
    try:
        module.main()
        record = json.loads((evidence/"completed.json").read_text())
        assert record["old_pid"] == old.pid and len(children) == 1
        assert record["new_pid"] == children[0].pid and children[0].poll() is None
        assert set(record["new_state"]["watched_jobs"]) == module.REMAINING
        assert json.loads((evidence/"previous-ready-state.json").read_text())["terminal_count"] == 17
    finally:
        thread.join(timeout=5)
        for child in [old, *children]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
