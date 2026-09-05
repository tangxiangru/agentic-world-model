#!/usr/bin/env python3
"""Where did a scientist's hours go? Time-by-category and a stage timeline from a parsed trace.

Usage: trace_timeline.py results/ptb/<batch>/<cell> [...]   (add --turns to dump every tool call with its duration)

Method: every `Assistant — turn N | ts` / `User — turn N | ts` header is a wall-clock stamp. A tool call
(Assistant turn with `Tool call — X`) runs until the next header (the User turn carrying the result), so that
gap is the tool's execution time; the gap from a User turn to the next Assistant turn is the model's own
generation (thinking/writing) time. Tool calls are classified by their text.
"""
import gzip
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CATS = [
    ("protocol", r"awm exp_protocol|memory/cards/|card\.template|exp_protocol/SKILL|pitfalls\.yaml|memory/index\.md"),
    ("train_launch", r"(python3?|torchrun|accelerate launch)[^\n]*\btrain_\w*\.py(?![^\n]*--dry-run)|(python3?|torchrun)[^\n]*\b(sft|grpo|dpo|rft)\w*\.py"),
    ("sample_eval", r"(python3?)[^\n]*(evaluate\.py|gen\w*\.py|sample\w*\.py|infer\w*\.py)|eval_all|inspect eval"),
    ("waiting_on_runs", r"\bsleep \d+|while .*sleep|tail -[nf]|pgrep|nvidia-smi|ps aux|timer\.sh|watch "),
    ("data", r"prep_\w*\.py|build_\w*\.py|load_dataset|jsonl|contamination|snapshot_download|tokeniz|datasets"),
    ("inspect_env", r"^\s*(ls|cat|head|sed|grep|find|python3? -c|pip|which|echo|df|du|free|wc)\b"),
]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def classify(tool, text):
    if tool in ("Edit", "Write", "NotebookEdit"):
        if re.search(r"memory/cards/|memory/index", text):
            return "protocol"
        if re.search(r"train|sft|grpo|dpo", text, re.I):
            return "write_train_code"
        if re.search(r"prep|data|build", text, re.I):
            return "write_data_code"
        return "write_other_code"
    if tool == "Read":
        if re.search(r"SKILL\.md|pitfalls|card\.template|example-card|memory/cards", text):
            return "protocol"
        return "read_files"
    if tool != "Bash":
        return f"tool_{tool.lower()}"
    for name, pat in CATS:
        if re.search(pat, text, re.I):
            return name
    return "bash_other"


def parse(lines):
    """Return events: (kind, turn, ts, tool, text, line_no)."""
    events = []
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"(Assistant|User) — turn (\d+) \| (\S+)", lines[i])
        if not m:
            i += 1
            continue
        role, turn, t = m.group(1), int(m.group(2)), ts(m.group(3))
        # scan the turn body for a tool call
        j = i + 1
        tool = None
        text_lines = []
        while j < n and not re.match(r"(Assistant|User) — turn ", lines[j]) and not lines[j].startswith("System event"):
            tm = re.match(r"\s*Tool call — (\w+)", lines[j])
            if tm and tool is None:
                tool = tm.group(1)
            text_lines.append(lines[j])
            j += 1
        events.append((role, turn, t, tool, "\n".join(text_lines), i))
        i = j
    return events


def main(bundle, dump=False):
    bundle = Path(bundle)
    lines = gzip.open(bundle / "solve_parsed.txt.gz", "rt", errors="replace").read().splitlines()
    ev = parse(lines)
    if not ev:
        print(f"{bundle}: no turns")
        return
    by_cat = Counter()
    calls = Counter()
    model_time = 0.0
    stages = {}
    rows = []
    for k, (role, turn, t, tool, text, ln) in enumerate(ev):
        nxt = ev[k + 1][2] if k + 1 < len(ev) else t
        dur = (nxt - t).total_seconds() / 3600
        if role == "Assistant" and tool:
            cat = classify(tool, text)
            by_cat[cat] += dur
            calls[cat] += 1
            rows.append((t, dur, cat, tool, text.strip().splitlines()[1][:110] if len(text.strip().splitlines()) > 1 else text.strip()[:110], ln))
            for stage, pat in (("first_card", r"awm exp_protocol new|memory/cards/exp-01"), ("first_lock", r"awm exp_protocol lock"),
                               ("first_train_launch", r"(python3?|torchrun|accelerate launch)[^\n]*\btrain_\w*\.py(?![^\n]*--dry-run)"),
                               ("first_eval", r"evaluate\.py"), ("first_rl", r"grpo|dpo|ppo"), ("final_model_written", r"final_model")):
                if stage not in stages and re.search(pat, text, re.I):
                    stages[stage] = t
        elif role == "Assistant":
            by_cat["assistant_text_turn"] += dur
        else:
            model_time += dur  # user(tool result) -> next assistant: model generation
    start, end = ev[0][2], ev[-1][2]
    total = (end - start).total_seconds() / 3600
    print(f"== {bundle.name}: {start:%H:%M}Z -> {end:%H:%M}Z = {total:.2f} h, {len(ev)} turns")
    print("   hours by category (tool execution time):")
    for cat, h in by_cat.most_common():
        print(f"     {cat:20s} {h:5.2f} h  ({calls[cat]} calls)")
    print(f"     {'model_generation':20s} {model_time:5.2f} h  (gaps from tool result to next assistant turn)")
    print("   stages (first occurrence):")
    for s, t in sorted(stages.items(), key=lambda x: x[1]):
        print(f"     {s:20s} {t:%H:%M}Z  (+{(t - start).total_seconds() / 3600:.2f} h)")
    if dump:
        print("   tool calls (start, hours, category, tool, text, line):")
        for t, dur, cat, tool, txt, ln in rows:
            if dur >= 0.05:
                print(f"     {t:%H:%M}Z {dur:5.2f}h {cat:18s} {tool:6s} L{ln} {txt}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump = "--turns" in sys.argv
    for b in args:
        main(b, dump)
        print()
