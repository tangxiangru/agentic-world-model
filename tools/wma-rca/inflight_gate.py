"""Lock-gate behaviour of cells still running, from the operator's `<cell>.inflight/solve_out.tail` peeks.

Per cell, in time order: LOCK (an `awm exp_protocol lock` / `awm wma review` call), GATE (the wait's
heartbeat count, the one-line verdict, or the failed/timeout/not-attached notice) and LAUNCH (a
backgrounded or long training/eval command). A LAUNCH between a LOCK and its GATE is a launch before
the verdict. The tail holds only the last 200 transcript lines, so this is a window, not the cell.

    python tools/wma-rca/inflight_gate.py [glob]        default results/ptb/*r02*-v2/*.inflight
"""
import json
import re
import sys
import glob
import os
root = "results/ptb"
cells = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else f"{root}/*r02*-v2/*.inflight"))
for d in cells:
    cell = d.split("/")[-2].split("high-")[1] + "/" + os.path.basename(d).replace(".inflight", "")
    lines = open(f"{d}/solve_out.tail", errors="replace").read().splitlines()
    events = []
    for ln in lines:
        m = re.match(r"\[(\S+)\] (\{.*)$", ln)
        if not m:
            continue
        ts, body = m.group(1), m.group(2)
        try:
            j = json.loads(body)
        except Exception:
            continue
        msg = j.get("message", {})
        for c in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if c.get("type") == "tool_use":
                cmd = str(c.get("input", {}).get("command", ""))
                if "exp_protocol lock" in cmd or "wma review" in cmd:
                    events.append((ts, "LOCK", cmd[:110]))
                elif re.search(r"(python\S* .*train|accelerate launch|torchrun|nohup .*train|sft|trl)", cmd) and "&" in cmd:
                    events.append((ts, "LAUNCH", cmd[:90]))
            if c.get("type") == "tool_result":
                txt = c.get("content")
                if isinstance(txt, list):
                    txt = " ".join(t.get("text", "") for t in txt if isinstance(t, dict))
                txt = str(txt)
                w = re.findall(r"(\d+\.\d) min elapsed", txt)
                v = re.search(r"verdict: (L0_runs=\S+; L1_valid=\S+; L2_effect=[^;]+; L3_worth_now=\S+)", txt)
                st = re.search(r"(recorded as a timeout|you may launch|no world-model agent is attached|review failed[^\n]*)", txt)
                if w or v or st:
                    events.append((ts, "GATE", f"waited~{w[-1] if w else '<0.5'}min " + (v.group(1) if v else "") + (" | " + st.group(1) if st else "")))
    print(f"== {cell}  ({len(lines)} lines, {lines[0][1:21] if lines else ''} .. {lines[-1][1:21] if lines else ''})")
    for e in events:
        print("  ", e[0][11:19], e[1], e[2])
