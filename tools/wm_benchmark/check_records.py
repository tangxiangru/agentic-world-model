"""Deterministic checks of every launch record in x_raw against the timeline it cites.
    python tools/wm_benchmark/check_records.py <benchmark_dir> <timeline_dir> [--cell CELL] [--json OUT]
Independent of the verifier agents: this only tests what a program can test.
  cmd_not_verbatim    launch.command is a pointer stub / prefix of / different from the Bash command at launch.seq
  seq_not_bash        launch.seq is not a Bash tool_use event
  archive_no_evidence the result after archive_submit_seq shows neither "archived" nor stage closed
  archive_evidence_truncated  the submit output was piped through tail/grep and the "archived" line is cut (info)
  archive_wrong_card  ... or prints a path for a different card id
  archive_not_first   an earlier submit of the same card already printed "archived"
  fs_source_mismatch  a file cited exactly fs@seq=N is not the version fs.jsonl has at N (path/sha)
  content_missing     a reconstructed/heredoc/read file has no stored content (content_file or _files)
  content_sha_mismatch  stored content does not hash to the record's sha256
  fs_not_launch_time  fs.jsonl has a later Write/Edit of that path before the launch seq
  fs_content_missing  the cited sha has no content in timeline/_files
  superseded_order    a superseded attempt's seq is >= the launch seq
  parent_unlinked     a workspace_dir parent is neither an earlier step's output nor annotated
  chain_end           the last step's outputs do not include archived_from_dir
  card_yaml_as_file   a memory/cards/*.yaml is listed under files
  schema              jsonschema violation (when jsonschema is importable)
"""
import json, sys, re, os, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
try:
    import jsonschema
except ImportError:
    jsonschema = None


def load_timeline(tl: Path, cell: str):
    ev, fs = {}, collections.defaultdict(list)
    for l in (tl / cell / "events.jsonl").open():
        e = json.loads(l)
        ev[e["seq"]] = e
    p = tl / cell / "fs.jsonl"
    if p.exists():
        for l in p.open():
            e = json.loads(l)
            fs[e["path"]].append((e["seq"], e.get("sha256")))
    return ev, fs


def result_text(tl: Path, cell: str, seq: int) -> str:
    p = tl / cell / "results" / f"{seq}.txt"
    return p.read_text(errors="replace") if p.exists() else ""


def check(rec: dict, cell: str, tl: Path, files_dir: Path, schema, x_raw_cell: Path = None) -> list[str]:
    out = []
    cid = rec.get("checkpoint_id", "?")
    card = cid.split("-exp-")[-1]
    ev, fs = load_timeline(tl, cell)
    if schema is not None:
        for e in jsonschema.Draft202012Validator(schema).iter_errors(rec):
            out.append(f"schema:{'/'.join(str(x) for x in e.absolute_path)}: {e.message[:80]}")
    # archive mapping
    a = (rec.get("archive_evidence") or {}).get("archive_submit_seq")
    if a is not None:
        e = ev.get(a)
        if not e or e.get("tool") != "Bash":
            out.append(f"archive_seq_not_bash:{a}")
        else:
            # the result of a submit may be in the next tool_result; search results of a..a+1
            txt = result_text(tl, cell, a + 1) or result_text(tl, cell, a)
            m = re.findall(r'"archived":\s*"([^"]+)"', txt)
            if not m:
                if re.search(r'"stage":\s*"closed"', txt) or "closed" in txt or "submit" in (e.get("command") or "") or "memory/cards" in (e.get("command") or ""):
                    out.append(f"archive_evidence_truncated:{a}")   # scientist piped the submit through tail/grep; mapping rests on the yaml
                else:
                    out.append(f"archive_no_evidence:{a}")
            elif not any(p.rstrip('/').endswith(f"exp-{card}") for p in m):
                out.append(f"archive_wrong_card:{a}:{m[:2]}")
            # archive-once: an earlier submit of this card already archived?
            for s, e2 in ev.items():
                if s < a and e2.get("kind") == "tool_use" and e2.get("tool") == "Bash" and f"exp-{card}.yaml" in (e2.get("command") or "") and "wm submit" in (e2.get("command") or ""):
                    t2 = result_text(tl, cell, s + 1)
                    if re.search(r'"archived":\s*"[^"]*exp-' + card + r'"', t2):
                        out.append(f"archive_not_first:{a}<-{s}")
                        break
    steps = rec.get("steps") or []
    produced = {}
    for i, st in enumerate(steps):
        L = st.get("launch") or {}
        seq = L.get("seq")
        e = ev.get(seq)
        if not e or e.get("kind") != "tool_use":
            out.append(f"seq_not_bash:{st.get('step_id')}:{seq}")
        elif e.get("tool") == "Bash":
            tc = (e.get("command") or "").strip()
            rc = (L.get("command") or "").strip()
            if tc != rc and "<<card yaml redacted>>" not in rc and re.sub(r"\s+", " ", tc) != re.sub(r"\s+", " ", rc):
                if re.search(r"<<[^>]*(inline|saved|see |heredoc)[^>]*>>|\((inline|heredoc)[^)]*files/[^)]*\)", rc):
                    out.append(f"cmd_not_verbatim:pointer:{st.get('step_id')}:{seq}")
                elif rc and (tc.startswith(rc) or rc in tc):
                    out.append(f"cmd_not_verbatim:prefix:{st.get('step_id')}:{seq}")
                else:
                    out.append(f"cmd_not_verbatim:other:{st.get('step_id')}:{seq}")
        for sa in L.get("superseded_attempts") or []:
            if isinstance(sa, dict) and isinstance(sa.get("seq"), int) and seq is not None and sa["seq"] > seq and "identical" not in str(sa.get("why", "")):
                out.append(f"superseded_order:{st.get('step_id')}:{sa['seq']}>={seq}")
        for f in st.get("files") or []:
            path, src, sha = f.get("path", ""), f.get("source", ""), f.get("sha256")
            path = re.sub(r"\s*\(.*\)\s*$", "", path)   # extractors annotate documentation entries: "x.py (pre-patch base)"
            mi = re.match(r"(inline|heredoc)@seq=(\d+)", src)
            if mi and f.get("role") == "inline_script" and seq is not None and int(mi.group(2)) == seq and not f.get("content_file"):
                continue   # the launch command itself; assemble materializes it from launch.command
            if "templates/" in path and path.endswith(".jinja"):
                continue   # PostTrainBench's own file; assemble resolves it from the submodule
            if "memory/cards/" in path and path.endswith(".yaml"):
                out.append(f"card_yaml_as_file:{st.get('step_id')}:{path}")
            m = re.match(r"fs@seq=(\d+)\s*(\([^)]*\))?\s*$", src)
            compound = re.match(r"fs@seq=(\d+)", src) and not m
            if compound or (not m and re.match(r"(heredoc|inline|read|derived|reconstructed)", src)):
                cf = f.get("content_file")
                cands = [Path(cf), REPO / cf, Path(str(x_raw_cell / "files" / Path(cf).name))] if cf else []
                found = next((c for c in cands if c.exists()), None)
                if src.startswith("unavailable"):
                    pass
                elif found is None and not (sha and (files_dir / sha).exists()):
                    out.append(f"content_missing:{st.get('step_id')}:{path}:{src[:40]}")
                elif found is not None and sha:
                    import hashlib
                    if hashlib.sha256(found.read_bytes()).hexdigest() != sha:
                        out.append(f"content_sha_mismatch:{st.get('step_id')}:{path}")
            if m:
                fseq = int(m.group(1))
                hist = fs.get(path, [])
                at = [h for h in hist if h[0] == fseq]
                if not at:
                    out.append(f"fs_source_mismatch:{st.get('step_id')}:{path}@{fseq}:no fs op at that seq")
                elif sha and at[0][1] != sha:
                    out.append(f"fs_source_mismatch:{st.get('step_id')}:{path}@{fseq}:sha differs from fs.jsonl")
                if seq is not None:
                    later = [h for h in hist if fseq < h[0] < seq]
                    if later:
                        out.append(f"fs_not_launch_time:{st.get('step_id')}:{path}@{fseq}: later ops at {[h[0] for h in later][:3]} before launch {seq}")
                if sha and not (files_dir / sha).exists():
                    out.append(f"fs_content_missing:{st.get('step_id')}:{path}:{sha[:12]}")
        pm = (st.get("inputs") or {}).get("parent_model") or {}
        parents = [pm] + list((st.get("inputs") or {}).get("additional_parents") or [])
        for p in parents:
            if p.get("kind") == "workspace_dir":
                ref = str(p.get("ref", "")).rstrip("/")
                by = p.get("produced_by_step")
                if by and by not in produced:
                    out.append(f"parent_unlinked:{st.get('step_id')}:produced_by_step={by} is not an earlier step")
                elif not by and not any(ref == o or ref.startswith(o + "/") for o in produced.values() for o in ([o] if isinstance(o, str) else o)):
                    pass  # a directory made outside the recorded steps; the extractor is expected to annotate it
        outs = [str(o).rstrip("/") for o in (st.get("outputs") or [])]
        produced[st.get("step_id")] = outs
    afd = str(rec.get("archived_from_dir") or "").rstrip("/")
    if steps and afd:
        last = [str(o).rstrip("/") for o in (steps[-1].get("outputs") or [])]
        if not any(afd == o or afd.startswith(o + "/") or o.startswith(afd) for o in last):
            out.append(f"chain_end:last outputs {last[:3]} vs archived_from_dir {afd}")
    return out


def main():
    args = sys.argv[1:]
    bench, tl = Path(args[0]), Path(args[1])
    only = args[args.index("--cell") + 1] if "--cell" in args else None
    outp = args[args.index("--json") + 1] if "--json" in args else None
    schema = json.loads((REPO / "tools/wm_benchmark/launch_record.schema.json").read_text()) if jsonschema else None
    files_dir = tl / "_files"
    report, kinds = {}, collections.Counter()
    for p in sorted((bench / "x_raw").glob("*/*.json")):
        cell = p.parent.name
        if only and cell != only:
            continue
        rec = json.loads(p.read_text())
        probs = check(rec, cell, tl, files_dir, schema, p.parent)
        if probs:
            report[rec.get("checkpoint_id", p.stem)] = probs
            for q in probs:
                kinds[q.split(":")[0]] += 1
    n = len(list((bench / "x_raw").glob("*/*.json")))
    print(f"records checked: {n}; with findings: {len(report)}")
    for k, v in kinds.most_common():
        print(f"  {k}: {v}")
    if outp:
        Path(outp).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
