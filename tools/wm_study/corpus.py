"""Convert the reconstructed exp-cards corpus (results/exp-cards/<name>/{train,test}/<run>/exp-NN.yaml)
into extra *training-only* rows for the world model, labeled with the card's own
largest-n local accuracy measurement.

These rows never enter a test fold: they come from other sessions (PostTrainBench
prior runs), on the same or other base models, and carry dev-set labels, not official ones.

Usage: python -m tools.wm_study.corpus --out data/analysis/wm_study/corpus_cards.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tools.wm_study.extract import SETUP_KEYS, build_lineage, normalize_family, parse_data_deps, parse_parents, scrub


def convert_run(run_dir: Path, side: str, min_n: int):
    cards = {}
    for f in sorted(run_dir.glob("exp-*.yaml")):
        try:
            c = yaml.safe_load(f.read_text())
        except Exception:
            continue
        if not isinstance(c, dict) or not isinstance(c.get("setup"), dict):
            continue
        card_id = f.stem
        setup = c["setup"]
        parents, complete = parse_parents(setup)
        ms = [m for m in ((c.get("result") or {}).get("measurements") or []) if isinstance(m, dict) and isinstance(m.get("value"), (int, float)) and not isinstance(m.get("value"), bool)]
        ms = [m for m in ms if str(m.get("metric") or "accuracy").lower().startswith("acc") and 0 <= m["value"] <= 1]
        best = max(ms, key=lambda m: (m.get("n") or 0)) if ms else None
        label = best["value"] if best and (best.get("n") or 0) >= min_n else None
        base = None
        pc = setup.get("parent_checkpoint") or {}
        if pc.get("origin") == "base_model":
            base = str(pc.get("path") or "")
        cards[card_id] = {
            "cell_id": f"ptb-{side}-{run_dir.name}",
            "benchmark": "gsm8k",
            "scientist_model": "ptb-corpus",
            "card_id": card_id,
            "example_id": f"ptb-{side}-{run_dir.name}/{card_id}",
            "plan_at": f"2026-01-01T00:00:{int(card_id.split('-')[1]):02d}Z",
            "plan_stage_recorded": False,
            "close_at": None,
            "family": normalize_family(setup),
            "parents": parents,
            "lineage_complete_local": complete,
            "data_deps": parse_data_deps(setup),
            "plan_setup": scrub({k: setup.get(k) for k in SETUP_KEYS if k in setup}),
            "scripts": [],
            "official_accuracy": None,
            "local_accuracy": label,
            "local_n": (best.get("n") if best else None),
            "execution": (c.get("result") or {}).get("execution"),
            "base_model_declared": base,
            "side": side,
        }
    # propagate the declared base model down the lineage
    def root_base(cid, seen=()):
        c = cards.get(cid)
        if not c or cid in seen:
            return None
        if c["base_model_declared"]:
            return c["base_model_declared"]
        for p in c["parents"]:
            b = root_base(p, (*seen, cid))
            if b:
                return b
        return None

    for cid, c in cards.items():
        c["base_model"] = root_base(cid) or ""
        c["plan_setup"].setdefault("base_model", c["base_model"])
    build_lineage(cards)
    return [cards[k] for k in sorted(cards)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("results/exp-cards/gsm8k-gemma-holdout-v1"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/corpus_cards.jsonl"))
    ap.add_argument("--min-n", type=int, default=100)
    args = ap.parse_args()
    rows = []
    for side in ("train", "test"):
        for run_dir in sorted((args.corpus / side).iterdir()):
            if run_dir.is_dir():
                rows.extend(convert_run(run_dir, side, args.min_n))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lab = [r for r in rows if r["local_accuracy"] is not None and r["lineage_complete"]]
    gem = [r for r in lab if "gemma" in r["base_model"].lower()]
    print(f"cards {len(rows)}, labeled+complete {len(lab)}, of which gemma {len(gem)} over {len({r['cell_id'] for r in gem})} runs; bases: {sorted({r['base_model'][:30] for r in lab})[:8]}")


if __name__ == "__main__":
    main()
