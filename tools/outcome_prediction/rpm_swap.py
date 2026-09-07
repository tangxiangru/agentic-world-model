"""Outcome-independent candidate-order audit for the frozen RPM judges."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .rpm_judge import digest, write_json
except ImportError:
    from rpm_judge import digest, write_json


def prepare(directory, count=10):
    jobs = json.loads((directory / "jobs.json").read_text())
    original = [j for j in jobs if j["arm"] in {"within_run", "cross_run"}]
    pair_ids = sorted({j["id"] for j in original}, key=lambda key: digest(["swap-audit", key]))[
        :count
    ]
    new = []
    for job in original:
        if job["id"] not in pair_ids:
            continue
        payload = json.loads(Path(job["input"]).read_text())
        payload["candidate_A"], payload["candidate_B"] = (
            payload["candidate_B"],
            payload["candidate_A"],
        )
        arm = job["arm"] + "_swap"
        path = directory / "inputs" / arm / (job["id"] + ".json")
        write_json(path, payload)
        new.append(
            {
                **job,
                "arm": arm,
                "swapped": not job["swapped"],
                "input": str(path.resolve()),
                "input_sha256": digest(payload),
                "characters": len(json.dumps(payload)),
            }
        )
    write_json(directory / "jobs.json", original + new)
    write_json(
        directory / "swap_protocol.json",
        {
            "pair_ids": pair_ids,
            "selection": "First ten pair IDs by SHA256(swap-audit, id), independent of outcomes",
            "change": "Only swap candidate_A/candidate_B; identical histories and trained model",
            "purpose": "Audit single-run order/repeat variability; cannot distinguish position bias from sampling noise",
        },
    )
    print(json.dumps({"pairs": len(pair_ids), "calls": len(new)}))


def score(directory):
    protocol = json.loads((directory / "swap_protocol.json").read_text())
    results = {}
    for arm in ("within_run", "cross_run"):
        rows = []
        for pair_id in protocol["pair_ids"]:
            original = directory / "outputs" / arm / (pair_id + ".json")
            swapped = directory / "outputs" / (arm + "_swap") / (pair_id + ".json")
            if not original.exists() or not swapped.exists():
                continue
            a, b = json.loads(original.read_text()), json.loads(swapped.read_text())
            if a["valid"] and b["valid"]:
                rows.append(
                    {
                        "id": pair_id,
                        "consistent": a["choice_a"] == b["choice_a"],
                        "absolute_probability_difference": abs(a["p_a"] - b["p_a"]),
                        "cost_usd": b.get("cost_usd", 0),
                    }
                )
        results[arm] = {
            "valid_pairs": len(rows),
            "choice_consistency": float(np.mean([r["consistent"] for r in rows])) if rows else None,
            "mean_absolute_probability_difference": float(
                np.mean([r["absolute_probability_difference"] for r in rows])
            )
            if rows
            else None,
            "reported_cost_usd": sum(r["cost_usd"] for r in rows),
            "pairs": rows,
        }
    write_json(directory / "swap_metrics.json", results)
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--directory", type=Path, default=Path("data/analysis/rpm/judge"))
    args = parser.parse_args()
    prepare(args.directory) if args.command == "prepare" else score(args.directory)


if __name__ == "__main__":
    main()
