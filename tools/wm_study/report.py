"""Compare the arms on the same decision sets, with paired run-level bootstrap CIs.

Usage: python -m tools.wm_study.report --runs data/analysis/wm_study/rpm/runs
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(runs: Path, arm, benchmark):
    d = runs / arm / benchmark
    if not d.exists():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in d.glob("set-*.json") if not p.name.endswith(".events.jsonl")}


def wm_only(labels, wm_dir: Path, benchmark):
    """Selector that takes the world model's top-ranked candidate, no agent."""
    out = {}
    for set_id, lab in labels.items():
        if lab["benchmark"] != benchmark:
            continue
        f = wm_dir / benchmark / set_id / "predictions.json"
        if not f.exists():
            continue
        pred = json.loads(f.read_text())["candidates"]
        choice = min(pred, key=lambda L: pred[L]["rank"])
        ys = {L: c["official_accuracy"] for L, c in lab["candidates"].items()}
        out[set_id] = {"score": {"chosen_acc": ys[choice], "oracle_acc": max(ys.values()), "random_acc": float(np.mean(list(ys.values()))), "regret": max(ys.values()) - ys[choice], "top1": float(ys[choice] >= max(ys.values()) - 1e-12), "valid": True}, "cell_id": lab["cell_id"], "meta": {}}
    return out


def stats(recs, key):
    return float(np.mean([r["score"][key] for r in recs]))


def paired_bootstrap(a, b, key, n=5000, seed=0):
    """Run-level paired bootstrap of mean(b) - mean(a) over shared decision sets."""
    ids = sorted(set(a) & set(b))
    by_run = defaultdict(list)
    for i in ids:
        by_run[a[i]["cell_id"]].append(i)
    runs = sorted(by_run)
    diffs = np.array([b[i]["score"][key] - a[i]["score"][key] for i in ids])
    idx = {i: k for k, i in enumerate(ids)}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        sample = rng.choice(len(runs), len(runs), replace=True)
        members = [idx[i] for r in sample for i in by_run[runs[r]]]
        boots.append(diffs[members].mean())
    boots = np.array(boots)
    return {"n_sets": len(ids), "n_runs": len(runs), "mean_diff": float(diffs.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "p_gt0": float((boots > 0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("data/analysis/wm_study/rpm/runs"))
    ap.add_argument("--labels", type=Path, default=Path("data/analysis/wm_study/rpm/private_labels.json"))
    ap.add_argument("--wm-dir", type=Path, default=Path("data/analysis/wm_study/rpm/wm_tool"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/rpm/report.md"))
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text())
    lines = ["# RPM inference-only vs. RPM + world model", ""]
    result = {}
    for benchmark in args.benchmarks:
        base, wm = load(args.runs, "base", benchmark), load(args.runs, "wm", benchmark)
        wm2 = load(args.runs, "wm2", benchmark)
        wmo = wm_only(labels, args.wm_dir, benchmark)
        shared = sorted(set(base) & set(wm))
        if not shared:
            continue
        arms = {"RPM inference-only agent": base, "RPM + world model agent (v1 guidance)": wm, "RPM + world model agent (v2 guidance: default to model)": {k: v for k, v in wm2.items() if k in shared}, "world model alone (argmax)": {k: v for k, v in wmo.items() if k in shared}}
        lines += [f"## {benchmark}: {len(shared)} decision sets, {len({base[i]['cell_id'] for i in shared})} held-out runs", "", "| selector | chosen acc | top-1 | regret | invalid | cost $ |", "|---|---:|---:|---:|---:|---:|"]
        ref = [base[i] for i in shared]
        lines.append(f"| random | {stats(ref, 'random_acc'):.4f} | | {stats(ref, 'oracle_acc') - stats(ref, 'random_acc'):.4f} | | |")
        for name, recs in arms.items():
            rs = [recs[i] for i in shared if i in recs]
            if not rs:
                continue
            cost = sum((r["meta"].get("cost_usd") or 0) for r in rs)
            lines.append(f"| {name} | {stats(rs, 'chosen_acc'):.4f} | {stats(rs, 'top1'):.3f} | {stats(rs, 'regret'):.4f} | {sum(not r['score']['valid'] for r in rs)} | {cost:.0f} |")
        lines.append(f"| oracle | {stats(ref, 'oracle_acc'):.4f} | 1.000 | 0 | | |")
        pb = paired_bootstrap(base, wm, "chosen_acc")
        pb_wmo = paired_bootstrap(base, {k: v for k, v in wmo.items() if k in shared}, "chosen_acc") if wmo else None
        rel = pb["mean_diff"] / stats(ref, "chosen_acc") * 100
        lines += ["", f"**WM agent minus inference-only agent, chosen accuracy:** {pb['mean_diff']:+.4f} ({rel:+.1f}% relative), run-level paired bootstrap 95% CI [{pb['ci95'][0]:+.4f}, {pb['ci95'][1]:+.4f}], P(diff>0)={pb['p_gt0']:.3f}, over {pb['n_sets']} sets / {pb['n_runs']} runs."]
        if pb_wmo:
            lines.append(f"**WM alone minus inference-only agent:** {pb_wmo['mean_diff']:+.4f}, 95% CI [{pb_wmo['ci95'][0]:+.4f}, {pb_wmo['ci95'][1]:+.4f}].")
        if len(set(wm2) & set(shared)) == len(shared):
            pb2 = paired_bootstrap(base, wm2, "chosen_acc")
            rel2 = pb2["mean_diff"] / stats(ref, "chosen_acc") * 100
            w2 = sum(wm2[i]["score"]["chosen_acc"] > base[i]["score"]["chosen_acc"] for i in shared)
            l2 = sum(wm2[i]["score"]["chosen_acc"] < base[i]["score"]["chosen_acc"] for i in shared)
            lines.append(f"**WM agent v2 minus inference-only agent:** {pb2['mean_diff']:+.4f} ({rel2:+.1f}% relative), 95% CI [{pb2['ci95'][0]:+.4f}, {pb2['ci95'][1]:+.4f}], P(diff>0)={pb2['p_gt0']:.3f}; better on {w2}, worse on {l2}.")
            result.setdefault(benchmark, {})["wm2"] = {"chosen": stats([wm2[i] for i in shared], "chosen_acc"), "diff": pb2}
        wins = sum(wm[i]["score"]["chosen_acc"] > base[i]["score"]["chosen_acc"] for i in shared)
        losses = sum(wm[i]["score"]["chosen_acc"] < base[i]["score"]["chosen_acc"] for i in shared)
        lines.append(f"Per-set: WM agent better on {wins}, worse on {losses}, tied on {len(shared) - wins - losses}.")
        # stratified by how much the choice matters, and follow/override behaviour
        lines += ["", "| subset | n | random | inference-only | WM agent v1 | WM agent v2 | WM alone | oracle |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        def gap_of(i):
            ys = [c["official_accuracy"] for c in labels[i]["candidates"].values()]
            return max(ys) - min(ys)
        for label, ids in (("all", shared), ("gap >= 2 pts", [i for i in shared if gap_of(i) >= 0.02]), ("gap >= 5 pts", [i for i in shared if gap_of(i) >= 0.05]), ("gap >= 10 pts", [i for i in shared if gap_of(i) >= 0.10])):
            if not ids:
                continue
            def m(recs):
                rs = [recs[i] for i in ids if i in recs]
                return f"{stats(rs, 'chosen_acc'):.3f}" if len(rs) == len(ids) else "-"
            lines.append(f"| {label} | {len(ids)} | {stats([base[i] for i in ids], 'random_acc'):.3f} | {m(base)} | {m(wm)} | {m(wm2)} | {m(wmo)} | {stats([base[i] for i in ids], 'oracle_acc'):.3f} |")
        for name, arm in (("v1", wm), ("v2", wm2)):
            ids = [i for i in shared if i in arm and arm[i]["score"]["valid"]]
            if not ids:
                continue
            tops = {}
            for i in ids:
                pred = json.loads((args.wm_dir / benchmark / i / "predictions.json").read_text())["candidates"]
                tops[i] = min(pred, key=lambda L: pred[L]["rank"])
            fol = [i for i in ids if arm[i]["score"]["choice"] == tops[i]]
            ovr = [i for i in ids if arm[i]["score"]["choice"] != tops[i]]
            lines.append(f"WM agent {name} followed the model's top pick on {len(fol)}/{len(ids)} sets (chosen {stats([arm[i] for i in fol], 'chosen_acc') if fol else float('nan'):.3f} vs inference-only {stats([base[i] for i in fol], 'chosen_acc') if fol else float('nan'):.3f} there) and overrode it on {len(ovr)} (chosen {stats([arm[i] for i in ovr], 'chosen_acc') if ovr else float('nan'):.3f} vs inference-only {stats([base[i] for i in ovr], 'chosen_acc') if ovr else float('nan'):.3f} vs WM alone {stats([wmo[i] for i in ovr], 'chosen_acc') if ovr else float('nan'):.3f}).")
        lines.append("")
        result[benchmark] = {"base": stats(ref, "chosen_acc"), "wm": stats([wm[i] for i in shared], "chosen_acc"), "diff": pb}
    text = "\n".join(lines)
    args.out.write_text(text + "\n")
    print(text)
    (args.out.with_suffix(".json")).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
