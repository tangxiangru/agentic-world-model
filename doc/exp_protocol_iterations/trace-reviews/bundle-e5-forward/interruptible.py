import json
import os
import signal
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

def interrupted(signum, frame):
    (out / "interrupted.json").write_text(json.dumps({"signal": signum, "pid": os.getpid()}) + "\n")
    print("owned fixture acknowledged signal", signum, flush=True)
    raise SystemExit(128 + signum)

signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
(out / "ready.json").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp()}) + "\n")
print("ready for controlled interruption", flush=True)
time.sleep(20)
(out / "complete.json").write_text('{"complete":true}\n')
