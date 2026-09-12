"""Step 3a: for every recorder cell, list the checkpoints that carry ten-run labels and the
trace events that bracket each one, so an extractor knows where to look.

    python -m tools.wm_benchmark.prepare_targets <mirror_root> <timeline_dir> <labels_dir> <out_dir>

Writes <out_dir>/<cell>.json:
  {"cell", "benchmark", "base_model", "scientist_model", "targets": [
     {"checkpoint_id", "card_id", "labels": ["matrix:G01", ..., "native"],
      "plan_submit_seq", "closed_submit_seq", "archived_path",
      "card_yaml_write_seqs": [...], "output_checkpoint_hint": "...",   # from the yaml the scientist wrote, join hint only
      "launch_candidates": [{"seq","ts","command"}]  # launch-like Bash calls between plan and closed submits
     }]}
The hints are deterministic and may be wrong; the extractor must confirm them against the trace.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def main(argv):
    root, tl_dir, labels_dir, out_dir = map(Path, argv[1:5])
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [json.loads(l) for l in (labels_dir / "labels.jsonl").open()]
    by_ckpt = defaultdict(list)
    for r in labels:
        by_ckpt[r["checkpoint_id"]].append(f"{r['track']}:{r['serving_id']}" if r["track"] == "matrix" else "native")
    manifests = {}
    for m in root.glob("manifest*.json"):
        for row in json.loads(m.read_text()):
            manifests[row["cell_id"]] = row
    written = 0
    for cell_dir in sorted(tl_dir.iterdir()):
        if not (cell_dir / "summary.json").exists():
            continue
        cell = cell_dir.name
        ckpts = sorted(c for c in by_ckpt if c.rsplit("-exp", 1)[0] == cell)
        if not ckpts:
            continue
        summary = json.loads((cell_dir / "summary.json").read_text())
        events = [json.loads(l) for l in (cell_dir / "events.jsonl").open()]
        submits = summary["submits"]
        # Join hint: the last replayed version of memory/cards/<card>.yaml (Write or Edit) before the
        # closing submit carries result.output_checkpoint. Hint only; the extractor confirms it.
        fs_rows = [json.loads(l) for l in (cell_dir / "fs.jsonl").open()]
        yaml_versions = defaultdict(list)   # card -> [(seq, sha)]
        yaml_writes = defaultdict(list)
        for fr in fs_rows:
            m = re.search(r"cards/(exp-\d+)\.yaml$", fr.get("path") or "")
            if m and fr.get("sha256"):
                yaml_versions[m.group(1)].append((fr["seq"], fr["sha256"]))
                if fr["op"] == "write":
                    yaml_writes[m.group(1)].append(fr["seq"])

        def hint_for(card, before_seq):
            best = None
            for seq, sha in yaml_versions.get(card, []):
                if before_seq is None or seq <= before_seq:
                    best = sha
            if not best:
                return None
            content = (tl_dir / "_files" / best).read_text(errors="replace")
            h = re.search(r"^\s*output_checkpoint:\s*(\S+)", content, re.M)
            return h.group(1).strip("'\"") if h else None
        man = manifests.get(cell, {})
        targets = []
        for cid in ckpts:
            card = cid.rsplit("-", 2)[-2] + "-" + cid.rsplit("-", 1)[-1]  # exp-NN
            subs = [s for s in submits if re.search(rf"cards/{card}\.yaml", s.get("args", "") + s.get("command", ""))]
            plan = subs[0]["seq"] if subs else None
            # The recorder copies the checkpoint ONCE: at the first submit whose result reports
            # `archived`; later submits of the same card return the existing archive without copying.
            closed = next((s["seq"] for s in subs if (s.get("result") or {}).get("archived")), None)
            archived = next((s["result"]["archived"] for s in subs if (s.get("result") or {}).get("archived")), None)
            later_closes = [s["seq"] for s in subs if (s.get("result") or {}).get("archived") and s["seq"] != closed]
            if closed is None and subs:
                closed = subs[-1]["seq"]
            lo, hi = plan or 0, closed or 10 ** 9
            cands = [l for l in summary["launches"] if lo <= l["seq"] <= hi]
            targets.append({"checkpoint_id": cid, "card_id": card, "labels": sorted(by_ckpt[cid]),
                            "plan_submit_seq": plan, "archive_submit_seq": closed, "closed_submit_seq": closed,
                            "later_submits_after_archive": later_closes, "archived_path": archived,
                            "all_submit_seqs": [s["seq"] for s in subs],
                            "card_yaml_write_seqs": yaml_writes.get(card, []),
                            "output_checkpoint_hint": hint_for(card, closed),
                            "yaml_versions": yaml_versions.get(card, []),
                            "launch_candidates": [{"seq": l["seq"], "ts": l["ts"], "command": l["command"][:300]} for l in cands]})
        out = {"cell": cell, "benchmark": man.get("benchmark", "gsm8k"), "base_model": man.get("base_model", "google/gemma-3-4b-pt"),
               "scientist_model": man.get("scientist_model"), "n_events": summary["events"],
               "sessions": len(summary["sessions"]), "compactions": summary["compactions"], "targets": targets}
        (out_dir / f"{cell}.json").write_text(json.dumps(out, indent=1))
        written += 1
    print(f"cells with targets: {written}")


if __name__ == "__main__":
    main(sys.argv)
