"""Extract recipe -> official-outcome examples from the recorder cells.

One example per experiment card. The *recipe* is the ordered chain of setup-only
steps (base model -> ... -> this card) plus the scripts snapshotted for each step.
No result, conclusion, evidence or dev-set number enters the recipe; free text is
scrubbed of accuracy-like numbers. The label is the official test accuracy of the
checkpoint the card produced (``wm_metrics/<card>.json``).

Usage:  python -m tools.wm_study.extract --out data/analysis/wm_study/cards.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

RAW = Path("data/traj/raw/awm-gsm8k-trajectories")
MANIFESTS = [("gsm8k", "manifest_r0.json"), ("aime2025", "manifest_aime_r0.json"), ("aime2025", "manifest_aime2_r0.json")]
BENCH_INFO = {
    "gsm8k": {"base_model": "google/gemma-3-4b-pt", "n_eval": 1319},
    "aime2025": {"base_model": "Qwen/Qwen3-4B-Base", "n_eval": 30},
}

# Setup keys kept verbatim (structured); free-text keys are scrubbed.
SETUP_KEYS = ("base_model", "budget", "command", "data", "method", "output_dir", "parent_checkpoint", "resume_argv", "progress")
PCT = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d+)?\s?%")
DEC = re.compile(r"(?<![\w.\-])0\.\d{2,4}(?![\d.])")
ACC_WORD = re.compile(r"(?i)\b(acc(?:uracy)?|score[sd]?|pass@?\d*|solved|correct|k/30|/30|\d+\s*/\s*30)\b")
EXP_ID = re.compile(r"(?i)exp[-_ ]?0*(\d{1,2})\b")


def ts(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def scrub_text(text: str) -> str:
    """Remove accuracy-like numbers from free text; keep hyperparameters."""
    text = PCT.sub("[n]%", text)
    # Every decimal fraction and k/30 count goes: sibling outcomes are quoted in
    # free text without any "accuracy" keyword nearby.
    text = DEC.sub("[n]", text)
    text = re.sub(r"\b\d{1,2}\s*/\s*30\b", "[n]/30", text)
    text = re.sub(r"(?i)\b(\d{2,3})\s*(?:pts?|points?)\b", "[n] pts", text)
    return text


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def normalize_family(setup) -> str:
    fam = str(((setup or {}).get("method") or {}).get("family") or "").lower()
    fam = re.sub(r"[\s\-]+", "_", fam)
    if not fam or fam == "none":
        return "unknown"
    if any(w in fam for w in ("merge", "soup", "averag")):
        return "merge"
    if "decod" in fam:
        return "decoding"
    if "eval" in fam or "selection" in fam:
        return "evaluation"
    if fam.startswith("sft") or fam in {"distill", "distillation", "finetune", "continued_pretraining"}:
        return "sft" if fam.startswith("sft") else fam
    if fam in {"rft", "rejection_sampling", "star"}:
        return "rft"
    if fam in {"grpo", "ppo", "dpo", "rl", "rlvr"}:
        return fam if fam != "rlvr" else "rl"
    return fam


def parse_parents(setup):
    """Return (parent card ids, complete?) from setup.parent_checkpoint.origin."""
    pc = (setup or {}).get("parent_checkpoint") or {}
    origin = str(pc.get("origin") or "")
    path = str(pc.get("path") or "")
    ids = ["exp-%02d" % int(m) for m in EXP_ID.findall(origin)]
    if ids:
        return sorted(set(ids)), True
    if origin in ("base_model", "base", "") or "base" in origin.lower():
        if origin == "" and not path:
            return [], False
        return [], True
    ids = ["exp-%02d" % int(m) for m in EXP_ID.findall(path)]
    if ids:
        return sorted(set(ids)), True
    return [], False


def parse_data_deps(setup):
    deps = set()
    for d in (setup or {}).get("data") or []:
        for key in ("source", "built_by", "path", "selection"):
            s = str(d.get(key) or "")
            if "derived" in s or "exp-" in s.lower():
                deps.update("exp-%02d" % int(m) for m in EXP_ID.findall(s))
    return sorted(deps)


def recipe_step(card_id, setup, scripts, relation):
    kept = {k: setup.get(k) for k in SETUP_KEYS if k in setup}
    kept = scrub(kept)
    return {
        "card_id": card_id,
        "relation": relation,
        "family": normalize_family(setup),
        "setup": kept,
        "scripts": scripts,
    }


def read_scripts(card_dir: Path):
    snap = card_dir / "snapshot"
    out = []
    if not snap.exists():
        return out
    for p in sorted(snap.iterdir()):
        if p.name == "MANIFEST.json" or not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        out.append({"name": p.name, "content": scrub_text(text)[:60000]})
    return out


def load_cell(cell_dir: Path, benchmark: str, manifest_row: dict):
    cell_id = cell_dir.name
    ledger = [json.loads(l) for l in (cell_dir / "wm/records.jsonl").read_text().splitlines() if l.strip()]
    cards = {}
    for card_dir in sorted((cell_dir / "wm/cards").glob("exp-*")):
        card_id = card_dir.name
        final = json.loads((card_dir / "card.json").read_text())
        records = []
        for rp in sorted(card_dir.glob("record-*.json")):
            rec = json.loads(rp.read_text())
            n = int(re.search(r"record-(\d+)", rp.name)[1])
            stage = next((e.get("stage") for e in ledger if e.get("card_id") == card_id and e.get("record_n") == n), None)
            if stage is None:
                stage = next((e.get("stage") for e in ledger if e.get("card_id") == card_id and e.get("at") == rec.get("at")), None)
            records.append({"n": n, "at": rec["at"], "stage": stage, "card": rec["card"]})
        if not records:
            continue
        plan_recs = [r for r in records if r["stage"] == "plan"] or records[:1]
        closed_recs = [r for r in records if r["stage"] == "closed"]
        plan_rec = plan_recs[0]
        plan_card = plan_rec["card"]
        plan_setup = plan_card.get("setup") or {}
        final_setup = final.get("setup") or {}
        label_path = cell_dir / "wm_metrics" / f"{card_id}.json"
        label = json.loads(label_path.read_text()) if label_path.exists() else None
        parents, complete = parse_parents(plan_setup)
        parents_final, _ = parse_parents(final_setup)
        cards[card_id] = {
            "cell_id": cell_id,
            "benchmark": benchmark,
            "scientist_model": manifest_row["scientist_model"],
            "card_id": card_id,
            "example_id": f"{cell_id}/{card_id}",
            "plan_at": plan_rec["at"],
            "plan_stage_recorded": plan_rec["stage"] == "plan",
            "close_at": closed_recs[-1]["at"] if closed_recs else None,
            "n_records": len(records),
            "family": normalize_family(plan_setup),
            "family_final": normalize_family(final_setup),
            "parents": parents,
            "parents_final": parents_final,
            "lineage_complete_local": complete,
            "data_deps": parse_data_deps(plan_setup),
            "plan_setup": scrub({k: plan_setup.get(k) for k in SETUP_KEYS if k in plan_setup}),
            "final_setup": scrub({k: final_setup.get(k) for k in SETUP_KEYS if k in final_setup}),
            "setup_changed": json.dumps(plan_setup, sort_keys=True) != json.dumps(final_setup, sort_keys=True),
            "plan_hypothesis": scrub(plan_card.get("hypothesis")),
            "plan_evaluation": scrub(plan_card.get("evaluation")),
            "scripts": read_scripts(card_dir),
            "official_accuracy": label["accuracy"] if label else None,
            "official_stderr": label.get("stderr") if label else None,
            "execution": ((final.get("result") or {}).get("execution")),
            "local_measurements": (final.get("result") or {}).get("measurements"),
            "final_conclusion_decision": ((final.get("conclusion") or {}).get("decision")),
        }
    return cards


def build_lineage(cards: dict):
    """Attach the ordered recipe chain (ancestors by weight + data deps) to each card."""
    for card in cards.values():
        chain, seen, incomplete = [], set(), not card["lineage_complete_local"]

        def visit(cid, relation):
            nonlocal incomplete
            if cid in seen:
                return
            if cid not in cards:
                incomplete = True
                return
            seen.add(cid)
            c = cards[cid]
            if not c["lineage_complete_local"]:
                incomplete = True
            for p in c["parents"]:
                visit(p, "weights")
            for d in c["data_deps"]:
                visit(d, "data")
            chain.append(recipe_step(cid, c["plan_setup"], c["scripts"], relation))

        for p in card["parents"]:
            visit(p, "weights")
        for d in card["data_deps"]:
            visit(d, "data")
        chain.append(recipe_step(card["card_id"], card["plan_setup"], card["scripts"], "target"))
        card["recipe"] = chain
        card["lineage_complete"] = not incomplete
        card["depth"] = sum(1 for s in chain if s["relation"] in ("weights", "target"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    args = ap.parse_args()
    rows = []
    for benchmark, mname in MANIFESTS:
        manifest = {r["cell_id"]: r for r in json.loads((RAW / mname).read_text())}
        for cell_id, mrow in sorted(manifest.items()):
            cards = load_cell(RAW / "cells" / cell_id, benchmark, mrow)
            build_lineage(cards)
            rows.extend(cards[c] for c in sorted(cards))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = Counter()
    for r in rows:
        b = r["benchmark"]
        summary[(b, "cards")] += 1
        summary[(b, "labeled")] += r["official_accuracy"] is not None
        summary[(b, "labeled_complete_lineage")] += (r["official_accuracy"] is not None and r["lineage_complete"])
        summary[(b, "plan_stage_recorded")] += r["plan_stage_recorded"]
        summary[(b, "setup_changed")] += r["setup_changed"]
        summary[(b, "no_scripts")] += not r["scripts"]
    for k in sorted(summary):
        print(k, summary[k])
    print("wrote", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
