#!/usr/bin/env python3
"""Read one harvested PTB cell bundle for the exp_protocol line.

Usage: tools/exp_protocol_cell_read.py results/ptb/<batch>/<cell> [...]

Prints, per bundle: status/accuracy/placement/hours; every card with its family,
declared stop token, verdict, pitfall hours, overrides (current lock and relocked_from),
and whether the training launch in the trace came after the lock; evaluate.py use
(direct calls, and inspect logs whose size gives the sample count at ~44 KB per sample);
RL launches; pitfall signatures (zero-grad, OOM, double <bos>, stop ids, orphaned vLLM);
whether a greedy generation config was written; background launches, sleeps, tool
timeouts; the timer's first and last reading; and the last assistant turn.
The parsed trace renders a Bash call either as `$ cmd` or as a `"command": ...` block.
"""
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

TRAIN_FAMILIES = {"sft", "rft", "dpo", "grpo", "distill", "ppo"}
GSM8K_TEST_N = 1319
BYTES_PER_SAMPLE = None  # calibrated from a known full run when available


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_trace(bundle):
    p = bundle / "solve_parsed.txt.gz"
    if not p.exists():
        return []
    return gzip.open(p, "rt", errors="replace").read().splitlines()


def turn_events(lines):
    for i, l in enumerate(lines):
        m = re.match(r"(Assistant|User) — turn (\d+) \| (\S+)", l)
        if m:
            yield i, m.group(1), int(m.group(2)), ts(m.group(3))


def commands(lines):
    """(line_no, ts, command_text) for every Bash tool call.

    The parsed trace renders a call either as `$ cmd` (single line) or as a JSON-ish block
    `"command": <text…>` that runs until the `"description":` / `"timeout":` / `}` line.
    """
    cur = None
    out = []
    i = 0
    n = len(lines)
    while i < n:
        l = lines[i]
        m = re.match(r"(Assistant|User) — turn (\d+) \| (\S+)", l)
        if m:
            cur = ts(m.group(3))
            i += 1
            continue
        s = l.strip()
        if s.startswith("$ "):
            out.append((i, cur, s[2:]))
        elif s.startswith('"command":'):
            buf = [s[len('"command":'):].strip()]
            j = i + 1
            while j < n:
                t = lines[j].strip()
                if t.startswith('"description":') or t.startswith('"timeout":') or t.startswith('"run_in_background"') or t == "}" or t == "},":
                    break
                buf.append(t)
                j += 1
            out.append((i, cur, "\n".join(buf).rstrip(",")))
            i = j
            continue
        i += 1
    return out


def eval_logs(bundle, st):
    """Inspect logs written by evaluate.py: one per run. Size ≈ samples × ~4.7 KB."""
    logs = []
    for item in st.get("skipped") or []:
        p = item.get("path", "")
        if re.search(r"logs/.*_gsm8k_.*\.json$", p):
            logs.append((p, int(item.get("size") or 0)))
    tl = bundle / "task" / "logs"
    if tl.exists():
        for f in tl.glob("*_gsm8k_*.json"):
            logs.append((f"logs/{f.name}", f.stat().st_size))
    return sorted(set(logs))


def main(bundle):
    bundle = Path(bundle)
    st = json.load(open(bundle / "status.json"))
    print(f"== {st.get('batch')} / {st.get('cell')}  job {st.get('job_id')}  {st.get('slurm_state')}")
    print(f"   accuracy={st.get('accuracy')}  complete={st.get('complete')}  issues={st.get('issues')}  judge_flags={st.get('judge_flags')}")
    prov = {}
    if (bundle / "runtime_provenance.json").exists():
        prov = json.load(open(bundle / "runtime_provenance.json"))
    node = (prov.get("slurm") or {}).get("node")
    print(f"   node={node}  strict_site={'ondem' in str(node)}")
    tt = (bundle / "time_taken.txt").read_text().strip() if (bundle / "time_taken.txt").exists() else None
    print(f"   time_taken={tt}")
    jg = bundle / "judgement_general.json"
    if jg.exists():
        j = json.load(open(jg))
        print(f"   general_anomaly={j.get('general_anomaly')}: {str(j.get('justification_general_anomaly'))[:300]}")

    lines = read_trace(bundle)
    evs = list(turn_events(lines))
    if evs:
        print(f"   trace: {len(lines)} lines, turns {evs[0][2]}..{evs[-1][2]}, {evs[0][3]:%H:%M}Z -> {evs[-1][3]:%H:%M}Z")
    cmds = commands(lines)

    # --- cards & locks ---
    cards_dir = bundle / "task" / "memory" / "cards"
    cards = sorted(cards_dir.glob("exp-*.yaml")) if cards_dir.exists() else []
    print(f"-- cards: {len(cards)}")
    n_overrides = 0
    n_relock = 0
    pit_h = 0.0
    lbl_results = []
    for c in cards:
        d = yaml.safe_load(open(c)) or {}
        cid = c.stem
        setup = d.get("setup") or {}
        method = setup.get("method") or {}
        fam = method.get("family")
        verdict = (d.get("conclusion") or {}).get("verdict")
        execution = (d.get("result") or {}).get("execution")
        pits = (d.get("situation") or {}).get("pitfalls_hit") or []
        ph = sum(float(x.get("cost_h") or 0) for x in pits if isinstance(x, dict))
        pit_h += ph
        lock = cards_dir / f"{cid}.lock.json"
        locked_at = None
        ov = {}
        if lock.exists():
            L = json.load(open(lock))
            locked_at = L.get("locked_at")
            ov = dict(L.get("overrides") or {})
            for h in L.get("relocked_from") or []:
                ov.update(h.get("overrides") or {})
                n_relock += 1
            n_overrides += len(ov)
        lbl = ""
        if fam in TRAIN_FAMILIES and locked_at:
            argv = (setup.get("command") or {}).get("argv") or []
            key = next((Path(a).name for a in argv if isinstance(a, str) and a.endswith(".py")), None)
            out_dir = str(setup.get("output_dir") or "")
            out_key = Path(out_dir).name if out_dir else None
            lt = ts(locked_at)
            hit = None
            for i, t, cmd in cmds:
                if key and key in cmd and (not out_key or out_key in cmd) and "--dry-run" not in cmd and "--help" not in cmd and "pgrep" not in cmd[:40] and "tail -" not in cmd[:40]:
                    if t is not None:
                        hit = (t, cmd)
                        if t >= lt:
                            break
            if hit:
                ok = hit[0] >= lt
                lbl = f"launch {hit[0]:%H:%M:%S}Z {'AFTER' if ok else 'BEFORE'} lock {lt:%H:%M:%S}Z"
                lbl_results.append(ok)
            else:
                lbl = f"launch not found (key={key}, out={out_key})"
        print(f"   {cid}: family={fam} stop_token={method.get('stop_token')} exec={execution} verdict={verdict} pitfalls_h={ph:.2f} overrides={list(ov) or '-'} {lbl}")
    print(f"-- overrides total={n_overrides} relocks={n_relock} pitfalls_cost_h={pit_h:.2f} lock_before_launch={sum(lbl_results)}/{len(lbl_results)}")
    hook = bundle / "task" / "memory" / ".stop_hook.json"
    if hook.exists():
        print(f"-- stop hook counter: {hook.read_text().strip()[:200]}")

    # --- trace-derived facts ---
    text = "\n".join(lines)
    ev_cmds = [(t, cmd) for _, t, cmd in cmds if re.search(r"python[3]?\s+\S*evaluate\.py", cmd)]
    lims = []
    for t, cmd in ev_cmds:
        m = re.search(r"--limit\s+\"?\$?\{?(\w+)", cmd)
        lims.append(m.group(1) if m else "FULL")
    logs = eval_logs(bundle, st)
    sizes = [s for _, s in logs]
    est = sorted(round(s / 44000) for s in sizes)  # ~44 KB per sample for a trained model
    own_test = [cmd[:80].replace("\n", " ") for _, t, cmd in cmds if re.search(r"test_data\.json|test\.jsonl|split[= ]+['\"]?test", cmd) and re.search(r"gen|sample|eval|score|infer", cmd, re.I) and "contamination" not in cmd]
    print(f"-- evaluate.py direct calls: {len(ev_cmds)} limits={lims}; inspect logs: {len(logs)} est_n={est} (max {max(est) if est else 0}); own-evaluator test-set cmds: {len(own_test)}" + (f" e.g. {own_test[0]}" if own_test else ""))
    rl_launch = [cmd[:90].replace("\n", " ") for _, t, cmd in cmds if re.search(r"(grpo|dpo|ppo|rloo)", cmd, re.I) and ".py" in cmd and "--help" not in cmd and "grep" not in cmd[:30]]
    print(f"-- RL launches={len(rl_launch)}" + (f" e.g. {rl_launch[0]}" if rl_launch else ""))
    zero = len(re.findall(r"grad_norm['\"]?: 0\.0\b", text))
    oom = len(re.findall(r"CUDA out of memory|OutOfMemoryError", text))
    dbos = len(re.findall(r"\[2, 2,|double.?bos|add_special_tokens=False", text))
    eos_ign = len(re.findall(r"generation_config\.eos_token_id|stop_token_ids", text))
    orphan = len(re.findall(r"Free memory \d+\.\d+/\d+\.\d+ GiB|orphan", text))
    print(f"-- zero-grad lines={zero} OOM={oom} double-bos signs={dbos} stop_token_ids/eos-config mentions={eos_ign} orphan-vllm signs={orphan}")
    greedy_set = len(re.findall(r"do_sample[\"']?\s*[:=]\s*[Ff]alse", text))
    greedy_words = len(re.findall(r"\bgreedy\b", text, re.I))
    print(f"-- greedy: do_sample=false writes/mentions={greedy_set} 'greedy' mentions={greedy_words}")
    bg = len([1 for _, t, cmd in cmds if re.search(r"nohup|setsid|&\s*$", cmd)])
    sleeps = len(re.findall(r"\bsleep \d+", text))
    timeouts = len(re.findall(r"Command timed out", text))
    print(f"-- background launches={bg} sleeps={sleeps} tool timeouts={timeouts}")
    rem = re.findall(r"Remaining time \(hours:minutes\):\s*\n\s*(\d+:\d+)", text)
    print(f"-- timer readings: first={rem[0] if rem else None} last={rem[-1] if rem else None} (n={len(rem)})")
    last_asst = [i for i, r, n, t in evs if r == "Assistant"]
    if last_asst:
        i = last_asst[-1]
        chunk = [l.strip() for l in lines[i + 1:i + 12] if l.strip()]
        print("-- last assistant turn:", " | ".join(chunk)[:400])
    tn = re.findall(r'"status": "(killed|stopped)"', text)
    print(f"-- background tasks killed/stopped at exit: {len(tn)}")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        main(b)
        print()
