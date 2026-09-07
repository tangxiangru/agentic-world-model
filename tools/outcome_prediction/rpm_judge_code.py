"""Inference-only judge arm with plan-time setup and launch scripts (RPM-faithful input).

The v2 judge saw only whitelisted canonical recipes. The RPM paper's inference-only
judge sees each candidate's plan and code plus scored history nodes. This arm adds,
for each candidate, the structured ``setup`` of its first (plan-stage) record and
the scripts snapshotted at launch, then reuses the frozen v2 pairs, histories,
candidate order, system prompt and executor unchanged.

Sibling pairs are retrospective: the later candidate's prose often names the
earlier one with its score. Free prose stays excluded; in setup strings and
scripts every mention of the other candidate (card id, output directory) and
every accuracy-looking number is redacted before the judge sees it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .rpm_judge import digest, execute, write_json
except ImportError:
    from rpm_judge import digest, execute, write_json

ARM = "code_within_run"
MAX_SCRIPT_CHARS = 12000


def redactor(other_card_id, other_dirs):
    tokens = [re.escape(other_card_id), re.escape(other_card_id.replace("-", "")), re.escape(other_card_id.replace("-", "_"))]
    for d in other_dirs:
        base = d.rstrip("/").rsplit("/", 1)[-1]
        if len(base) >= 3:
            tokens.append(re.escape(base))
    sibling = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(tokens) + r")(?![A-Za-z0-9_])")
    # 0.483 / .483 / 48.3% style values: accuracies leak, learning rates do not.
    acc = re.compile(r"(?<![\w.\-])(?:0?\.\d{2,4}|\d{1,2}\.\d%|\d{2}%)(?![\w.])")

    def scrub(text):
        text = sibling.sub("[sibling]", text)
        return acc.sub("[n]", text)

    def walk(obj):
        if isinstance(obj, str):
            return scrub(obj)
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        return obj

    return walk


def candidate_extras(raw_root, ex, other, walk):
    cell, card = ex["example_id"].split("/")
    folder = raw_root / "cells" / cell / "wm" / "cards" / card
    first = json.loads((folder / "record-01.json").read_text())["card"]
    setup = dict(first.get("setup") or {})
    setup.pop("output_dir", None)  # bare path, carries the scientist's naming only
    scripts = {}
    for path in sorted((folder / "snapshot").glob("*")):
        if path.name == "MANIFEST.json" or not path.is_file():
            continue
        text = path.read_text(errors="replace")
        if len(text) > MAX_SCRIPT_CHARS:
            text = text[:MAX_SCRIPT_CHARS] + "\n# [truncated]\n"
        scripts[path.name] = text
    return {"plan_time_setup": walk(setup), "launch_scripts": walk(scripts)}


def other_dirs(ex, raw_root):
    cell, card = ex["example_id"].split("/")
    folder = raw_root / "cells" / cell / "wm" / "cards" / card
    dirs = set()
    for rec in folder.glob("record-*.json"):
        c = json.loads(rec.read_text())["card"]
        for v in ((c.get("setup") or {}).get("output_dir"), (c.get("result") or {}).get("output_checkpoint")):
            if v:
                dirs.add(str(v))
    return dirs


def prepare(args):
    src = args.source_dir
    labels = json.loads((src / "hidden_labels.json").read_text())
    protocol = json.loads((src / "protocol.json").read_text())
    rows = {}
    for line in args.examples.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["example_id"]] = r
    jobs = []
    for label in labels:
        base = json.loads((src / "inputs" / "within_run" / (label["id"] + ".json")).read_text())
        a, b = rows[label["a_id"]], rows[label["b_id"]]
        left, right = (b, a) if label["swapped"] else (a, b)
        payload = dict(base)
        for key, me, other in (("candidate_A", left, right), ("candidate_B", right, left)):
            walk = redactor(other["card_id"], other_dirs(other, args.raw_root))
            payload[key] = {**base[key], **candidate_extras(args.raw_root, me, other, walk)}
        payload["candidate_encoding"] = (
            "plan_time_setup is the structured setup registered before launch; launch_scripts "
            "are the scripts as snapshotted at launch. '[sibling]' and '[n]' are redactions of "
            "the other candidate's identifiers and of accuracy-like numbers."
        )
        path = args.output_dir / "inputs" / ARM / (label["id"] + ".json")
        write_json(path, payload)
        jobs.append(
            {
                "id": label["id"],
                "arm": ARM,
                "input": str(path.resolve()),
                "input_sha256": digest(payload),
                "swapped": label["swapped"],
                "characters": len(json.dumps(payload)),
            }
        )
    write_json(args.output_dir / "hidden_labels.json", labels)
    write_json(args.output_dir / "folds.json", json.loads((src / "folds.json").read_text()))
    write_json(args.output_dir / "jobs.json", jobs)
    protocol = {
        **protocol,
        "version": 3,
        "arm": ARM,
        "derived_from": str(src),
        "inputs_add": ["plan-stage structured setup", "launch-time script snapshots"],
        "redaction": "other candidate's card id / output dirs -> [sibling]; accuracy-like numbers -> [n]",
        "inputs_exclude": protocol["inputs_exclude"],
    }
    write_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps({"pairs": len(jobs), "characters": sum(j["characters"] for j in jobs)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/traj/raw/awm-gsm8k-trajectories"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/analysis/rpm/judge"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/rpm/judge_code"))
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--arms", nargs="+", default=[ARM])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--call-budget", type=float, default=0.6)
    parser.add_argument("--total-budget", type=float, default=30)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else execute)(args)


if __name__ == "__main__":
    main()
