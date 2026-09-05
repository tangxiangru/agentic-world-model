"""Per-cell table and score correlations over harvested PTB cells.

    .venv/bin/python tools/wma-rca/cells.py results/ptb/<batch> [more batches] --out <dir>

For every harvested cell: final official accuracy, cards and families, training hours,
epoch-rows, whether checkpoints were kept / intermediate checkpoints scored (C5) / weights
averaged (C6) / the full test set read for selection, hours used, the packaged card. Then the
Spearman correlation of each numeric feature with the final score over all cells, and the
arm means. Writes cells.json and cells.md under --out.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rcalib import (TRAIN_FAMILIES, batch_cells, cell_arm, final_accuracy, g, load_cards, load_verdicts,
                    measurement, num, spearman, time_taken_h, write_outputs)

C5_TEXT = re.compile(r"checkpoint[- ]?(selection|sweep)|save_steps|intermediate checkpoint|epoch-1 checkpoint"
                     r"|checkpoint-\d+", re.I)
SOUP_TEXT = re.compile(r"\bsoup\b|weight averag|uniform average", re.I)
FULL_SET = re.compile(r"--limit -1|1319", re.I)
NUMERIC = ["n_cards", "n_train_cards", "train_wall_h", "epoch_rows_k", "max_rows_k", "first_sft_value",
           "gain_after_first_sft", "ckpt_kept", "c5_selection", "soup", "full_set_selection", "max_eval_n",
           "decode_fix", "flat_or_neg_train_h", "failed_or_killed", "time_taken_h"]


def card_row(card_id: str, card: dict) -> dict:
    family = g(card, "setup.method.family")
    hyper = g(card, "setup.method.hyperparams") or {}
    rows = 0.0
    for entry in g(card, "setup.data") or []:
        if not isinstance(entry, dict):
            continue
        n = num(entry.get("n_examples"))
        source = str(entry.get("source", ""))
        if n and family in TRAIN_FAMILIES and not re.search(r"no training|no new data|derived:", source):
            rows += n
    epochs = num(hyper.get("epochs")) or 0.0
    m = measurement(card)
    return {
        "card": card_id, "family": family, "parent": g(card, "setup.parent_checkpoint.origin"),
        "elapsed_h": num(g(card, "situation.elapsed_h")), "planned_h": num(g(card, "setup.budget.planned_h")),
        "wall_h": num(g(card, "result.wall_h")), "execution": g(card, "result.execution"),
        "epochs": epochs, "rows": rows, "epoch_rows": epochs * rows,
        "ckpt_every": g(card, "setup.checkpoints.every_steps"), "ckpt_keep": g(card, "setup.checkpoints.keep"),
        "n": m["n"], "value": m["value"], "delta": m["delta"], "max_n": m["max_n"],
        "verdict": g(card, "conclusion.verdict"), "decision": g(card, "conclusion.decision"),
        "text": str(card),
    }


def cell_features(batch: str, cell: str, cell_dir: Path) -> dict:
    cards = load_cards(cell_dir)
    rows = [card_row(card_id, card) for card_id, card in cards.items()]
    train = [r for r in rows if r["family"] in TRAIN_FAMILIES]
    all_text = "\n".join(r["text"] for r in rows)
    first_sft = next((r for r in train if r["value"] is not None), None)
    final = final_accuracy(cell_dir)
    taken = time_taken_h(cell_dir)
    packaged = None
    for r in rows:
        if "final_model" in r["text"] and r["decision"] == "adopt":
            packaged = r["card"]
    if packaged is None:
        adopted = [r["card"] for r in rows if r["decision"] == "adopt"]
        packaged = adopted[-1] if adopted else None
    verdicts = load_verdicts(cell_dir)
    return {
        "batch": batch, "cell": cell, "arm": cell_arm(cell_dir), "final": final,
        "n_cards": len(rows), "n_train_cards": len(train),
        "families": [r["family"] for r in rows],
        "train_wall_h": round(sum(r["wall_h"] or 0 for r in train), 2),
        "epoch_rows_k": round(sum(r["epoch_rows"] for r in train) / 1000, 1),
        "max_rows_k": round(max([r["rows"] for r in train] + [0]) / 1000, 1),
        "first_sft_card": first_sft["card"] if first_sft else None,
        "first_sft_value": first_sft["value"] if first_sft else None,
        "gain_after_first_sft": round(final - first_sft["value"], 4) if (final is not None and first_sft) else None,
        "ckpt_kept": any(r["ckpt_every"] not in (None, "null") or str(r["ckpt_keep"]) == "all" for r in train),
        "c5_selection": bool(C5_TEXT.search(all_text)) and any(
            r["family"] in ("other", "checkpoint", "decode-config", "merge") and r["card"] != "exp-01"
            and re.search(r"checkpoint", r["text"], re.I) for r in rows),
        "soup": any(r["family"] == "merge" or SOUP_TEXT.search(r["text"]) for r in rows),
        "full_set_selection": any((r["max_n"] or 0) >= 1319 for r in rows) or bool(FULL_SET.search(all_text)),
        "max_eval_n": max([r["max_n"] or 0 for r in rows] + [0]),
        "decode_fix": any(r["family"] == "decode-config" for r in rows),
        "flat_or_neg_train_h": round(sum((r["wall_h"] or 0) for r in train
                                         if r["delta"] is not None and r["delta"] <= 0.0), 2),
        "failed_or_killed": sum(1 for r in rows if r["execution"] not in (None, "completed")),
        "time_taken_h": round(taken, 2) if taken else None,
        "packaged_card": packaged,
        "n_verdicts": sum(1 for v in verdicts.values() if not v.get("_rejected")),
        "n_rejected_verdicts": sum(1 for v in verdicts.values() if v.get("_rejected")),
        "cards": [{k: v for k, v in r.items() if k != "text"} for r in rows],
    }


def correlations(cells: list[dict]) -> dict:
    scored = [c for c in cells if c["final"] is not None]
    out: dict[str, dict] = {}
    for key in NUMERIC:
        pairs = [(float(c[key]), c["final"]) for c in scored if c.get(key) is not None]
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None
        by_arm = {}
        for arm in sorted({c["arm"] for c in cells}):
            values = [float(c[key]) for c in cells if c["arm"] == arm and c.get(key) is not None]
            by_arm[arm] = round(sum(values) / len(values), 3) if values else None
        out[key] = {"spearman": round(rho, 3) if rho is not None else None, "n": len(pairs), "arm_means": by_arm}
    return out


def markdown(cells: list[dict], corr: dict) -> str:
    lines = ["# Cells", "", f"{len(cells)} cells; arms: "
             + ", ".join(f"{arm} {sum(1 for c in cells if c['arm'] == arm)}" for arm in sorted({c['arm'] for c in cells})), "",
             "| cell | arm | final | cards | 1st-SFT | gain after | train h | epoch-rows k | ckpt kept | C5 | soup | full set | max n | hours used | packaged | verdicts |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---|---:|"]
    for c in cells:
        lines.append("| {cell} | {arm} | {final} | {n_cards} | {fs} | {gain} | {train_wall_h} | {epoch_rows_k} | {ck} | {c5} | {soup} | {full} | {max_eval_n} | {tt} | {packaged_card} | {nv} |".format(
            cell=c["cell"], arm=c["arm"], final=f"{c['final']:.4f}" if c["final"] is not None else "—",
            n_cards=c["n_cards"], fs=f"{c['first_sft_value']:.3f}" if c["first_sft_value"] is not None else "—",
            gain=f"{c['gain_after_first_sft']:+.3f}" if c["gain_after_first_sft"] is not None else "—",
            train_wall_h=c["train_wall_h"], epoch_rows_k=c["epoch_rows_k"], ck="y" if c["ckpt_kept"] else "n",
            c5="y" if c["c5_selection"] else "n", soup="y" if c["soup"] else "n", full="y" if c["full_set_selection"] else "n",
            max_eval_n=c["max_eval_n"], tt=c["time_taken_h"] if c["time_taken_h"] is not None else "—",
            packaged_card=c["packaged_card"], nv=c["n_verdicts"]))
    arms = sorted({c["arm"] for c in cells})
    lines += ["", "## Spearman with the final score, and arm means", "",
              "| feature | ρ | n | " + " | ".join(arms) + " |", "|---|---:|---:|" + "---:|" * len(arms)]
    for key, entry in corr.items():
        lines.append(f"| {key} | {entry['spearman'] if entry['spearman'] is not None else '—'} | {entry['n']} | "
                     + " | ".join(str(entry["arm_means"].get(arm)) for arm in arms) + " |")
    for arm in arms:
        finals = [c["final"] for c in cells if c["arm"] == arm and c["final"] is not None]
        if len(finals) >= 2:
            mean = sum(finals) / len(finals)
            sd = (sum((f - mean) ** 2 for f in finals) / (len(finals) - 1)) ** 0.5
            lines.append(f"\n{arm}: n={len(finals)} mean={mean:.4f} SD={sd:.4f} range={min(finals):.4f}–{max(finals):.4f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batches", nargs="+", help="results/ptb/<batch> directories")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cells = [cell_features(batch, cell, d) for batch, cell, d, _ in batch_cells(args.batches)]
    if not cells:
        print("no harvested cells under the given batches")
        return 2
    corr = correlations(cells)
    text = markdown(cells, corr)
    write_outputs(Path(args.out), "cells", {"cells": cells, "correlations": corr}, text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
