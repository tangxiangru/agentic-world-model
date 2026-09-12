"""Turn one scientist session trace (Claude Code stream-json with a timestamp prefix per line)
into a compact, inspectable timeline plus a content-addressed store of every file version the
scientist wrote through the Write/Edit tools.

    python -m tools.wm_benchmark.trace_timeline <cells_dir> <out_dir> [cell ...]

Per cell, writes <out_dir>/<cell>/:
  events.jsonl    one record per tool call / tool result / assistant text / session boundary,
                  in trace order (seq). Bash commands are kept in full; file contents are
                  replaced by sha256 + size and stored once under <out_dir>/_files/<sha256>.
  results/<seq>.txt   full text of every tool result (events.jsonl keeps the first 1,500 chars).
  fs.jsonl        the virtual-filesystem replay of Write/Edit: after each op, the path's
                  content hash. Edits to a path never Written are flagged base_unknown.
  summary.json    session metadata (cwd, model, harness version, resumes, compactions),
                  every `awm wm submit|outcome` call with its parsed result (stage, archived
                  path), and every Bash call that looks like a training/merging launch.

Nothing here interprets the science; it only reshapes the trace so an extractor (human or
agent) can read the launch history of one session without loading 2 MB of JSON.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

LINE_RE = re.compile(r"^\[(\S+)\] (\{.*\})\s*$")
RESULT_KEEP = 1500
LAUNCH_RE = re.compile(
    r"(python3?|accelerate|torchrun|deepspeed|trl)\b[^\n|]*?(train|sft|dpo|grpo|rft|ppo|orpo|kto|"
    r"soup|merge|average|distill|finetune|fine_tune|rl\b|rlvr|reward)", re.I)
SUBMIT_RE = re.compile(r"awm\s+wm\s+(submit|outcome)\b(.*)")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(f"<{item.get('type')}>")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else json.dumps(content)


def parse_submit_result(text: str):
    """The recorder prints a JSON object; the scientist often pipes it through tail, so we
    recover the fields we need from whatever suffix survived."""
    out = {}
    for key in ("card_id", "stage", "archived"):
        m = re.search(r'"%s":\s*(null|"[^"]*")' % key, text)
        if m:
            out[key] = None if m.group(1) == "null" else m.group(1).strip('"')
    m = re.search(r'"snapshotted":\s*\[(.*?)\]', text, re.S)
    if m:
        out["snapshotted"] = re.findall(r'"([^"]+)"', m.group(1))
    m = re.search(r'"missing":\s*\[(.*?)\]', text, re.S)
    if m:
        out["missing"] = re.findall(r'"([^"]+)"', m.group(1))
    return out


def process_cell(cell_dir: Path, out_dir: Path, files_dir: Path):
    trace = cell_dir / "solve_out_sanitized.txt"
    cell = cell_dir.name
    dest = out_dir / cell
    (dest / "results").mkdir(parents=True, exist_ok=True)
    events = []
    fs = {}          # path -> content (str) after replay
    fs_log = []
    pending = {}     # tool_use_id -> seq of the tool_use event
    summary = {"cell": cell, "sessions": [], "compactions": 0, "submits": [], "launches": [],
               "tool_counts": {}, "unparsed_lines": 0, "events": 0, "first_ts": None, "last_ts": None}
    seq = 0
    with trace.open(errors="replace") as fh:
        for raw in fh:
            m = LINE_RE.match(raw)
            if not m:
                summary["unparsed_lines"] += 1
                continue
            ts, body = m.group(1), m.group(2)
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                summary["unparsed_lines"] += 1
                continue
            summary["first_ts"] = summary["first_ts"] or ts
            summary["last_ts"] = ts
            etype, sub = ev.get("type"), ev.get("subtype")
            if etype == "system" and sub == "init":
                summary["sessions"].append({"ts": ts, "cwd": ev.get("cwd"), "model": ev.get("model"),
                                            "claude_code_version": ev.get("claude_code_version"),
                                            "session_id": ev.get("session_id")})
                seq += 1
                events.append({"seq": seq, "ts": ts, "kind": "init", "cwd": ev.get("cwd"),
                               "model": ev.get("model"), "version": ev.get("claude_code_version")})
                continue
            if etype == "system" and sub == "compact_boundary":
                summary["compactions"] += 1
                seq += 1
                events.append({"seq": seq, "ts": ts, "kind": "compact"})
                continue
            if etype == "assistant":
                for blk in ev.get("message", {}).get("content", []):
                    btype = blk.get("type")
                    if btype == "text" and blk.get("text", "").strip():
                        seq += 1
                        events.append({"seq": seq, "ts": ts, "kind": "text",
                                       "text": blk["text"][:2000], "len": len(blk["text"])})
                    elif btype == "tool_use":
                        seq += 1
                        name, inp, tid = blk.get("name"), blk.get("input") or {}, blk.get("id")
                        summary["tool_counts"][name] = summary["tool_counts"].get(name, 0) + 1
                        rec = {"seq": seq, "ts": ts, "kind": "tool_use", "tool": name, "id": tid}
                        pending[tid] = seq
                        if name == "Write":
                            content = inp.get("content", "")
                            h = sha(content.encode())
                            (files_dir / h).write_text(content) if not (files_dir / h).exists() else None
                            rec.update({"path": inp.get("file_path"), "sha256": h, "bytes": len(content.encode())})
                            fs[inp.get("file_path")] = content
                            fs_log.append({"seq": seq, "ts": ts, "op": "write", "path": inp.get("file_path"), "sha256": h})
                        elif name == "Edit":
                            path = inp.get("file_path")
                            old, new = inp.get("old_string", ""), inp.get("new_string", "")
                            rec.update({"path": path, "old_string": old[:3000], "new_string": new[:3000],
                                        "replace_all": bool(inp.get("replace_all"))})
                            entry = {"seq": seq, "ts": ts, "op": "edit", "path": path}
                            if path in fs:
                                base = fs[path]
                                if old in base:
                                    fs[path] = base.replace(old, new) if inp.get("replace_all") else base.replace(old, new, 1)
                                    h = sha(fs[path].encode())
                                    (files_dir / h).write_text(fs[path]) if not (files_dir / h).exists() else None
                                    entry["sha256"] = h
                                else:
                                    entry["error"] = "old_string_not_found"
                            else:
                                entry["error"] = "base_unknown"
                            fs_log.append(entry)
                        elif name == "Bash":
                            cmd = inp.get("command", "")
                            rec.update({"command": cmd, "description": inp.get("description"),
                                        "timeout": inp.get("timeout"), "background": bool(inp.get("run_in_background"))})
                            sm = SUBMIT_RE.search(cmd)
                            if sm:
                                summary["submits"].append({"seq": seq, "ts": ts, "verb": sm.group(1),
                                                           "args": sm.group(2).strip()[:200], "command": cmd[:400]})
                            if LAUNCH_RE.search(cmd):
                                summary["launches"].append({"seq": seq, "ts": ts, "command": cmd[:600]})
                        elif name == "Read":
                            rec.update({"path": inp.get("file_path"), "offset": inp.get("offset"), "limit": inp.get("limit")})
                        else:
                            rec["input"] = json.dumps(inp)[:1500]
                        events.append(rec)
            elif etype == "user":
                content = ev.get("message", {}).get("content")
                if not isinstance(content, list):
                    continue
                for blk in content:
                    if blk.get("type") != "tool_result":
                        continue
                    seq += 1
                    txt = result_text(blk.get("content"))
                    use_seq = pending.get(blk.get("tool_use_id"))
                    (dest / "results" / f"{seq}.txt").write_text(txt)
                    rec = {"seq": seq, "ts": ts, "kind": "tool_result", "for_seq": use_seq,
                           "is_error": bool(blk.get("is_error")), "len": len(txt), "text": txt[:RESULT_KEEP]}
                    events.append(rec)
                    if use_seq is not None:
                        for s in summary["submits"]:
                            if s["seq"] == use_seq:
                                s["result_seq"] = seq
                                s["result"] = parse_submit_result(txt)
                                s["is_error"] = bool(blk.get("is_error"))
                        for l in summary["launches"]:
                            if l["seq"] == use_seq:
                                l["result_seq"] = seq
                                l["is_error"] = bool(blk.get("is_error"))
                                l["result_head"] = txt[:300]
    summary["events"] = seq
    with (dest / "events.jsonl").open("w") as fh:
        for rec in events:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (dest / "fs.jsonl").open("w") as fh:
        for rec in fs_log:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary["files_written"] = sorted({r["path"] for r in fs_log if r["op"] == "write"})
    summary["edits_base_unknown"] = sorted({r["path"] for r in fs_log if r.get("error") == "base_unknown"})
    (dest / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main(argv):
    cells_dir, out_dir = Path(argv[1]), Path(argv[2])
    files_dir = out_dir / "_files"
    files_dir.mkdir(parents=True, exist_ok=True)
    names = argv[3:] or sorted(p.name for p in cells_dir.iterdir() if (p / "solve_out_sanitized.txt").exists())
    for name in names:
        s = process_cell(cells_dir / name, out_dir, files_dir)
        print(f"{name}: events={s['events']} sessions={len(s['sessions'])} compactions={s['compactions']} "
              f"submits={len(s['submits'])} launches={len(s['launches'])} writes={len(s['files_written'])} "
              f"base_unknown={len(s['edits_base_unknown'])} unparsed={s['unparsed_lines']}")


if __name__ == "__main__":
    main(sys.argv)
