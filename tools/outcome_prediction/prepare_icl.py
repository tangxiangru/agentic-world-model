"""Make a reproducible blinded, cell-disjoint in-context learning exercise."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/analysis/outcome_prediction/icl")
    )
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.examples.read_text().splitlines() if s.strip()]
    rows = [r for r in rows if r["eligible"] and r["y"] is not None]
    models = sorted({r["scientist_model"] for r in rows})
    test_cells = []
    for model in models:
        cells = {r["cell_id"] for r in rows if r["scientist_model"] == model}
        cells = sorted(
            cells, key=lambda x: hashlib.sha256(("icl-20260904:" + x).encode()).hexdigest()
        )
        test_cells.extend(cells[4 * args.fold : 4 * (args.fold + 1)])
    train_cells = sorted({r["cell_id"] for r in rows} - set(test_cells))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = {"train_cells": train_cells, "test_cells": sorted(test_cells)}
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    hidden, train, test = [], [], []
    for row in rows:
        is_test = row["cell_id"] in test_cells
        bucket = test if is_test else train
        opaque = ("test-" if is_test else "train-") + f"{len(bucket) + 1:03d}"
        # No researcher identity, file names, parent scores, or post-run fields.
        item = {"id": opaque, "recipe_sequence": [s["recipe"] for s in row["lineage"]]}
        if is_test:
            hidden.append(
                {
                    "id": opaque,
                    "example_id": row["example_id"],
                    "cell_id": row["cell_id"],
                    "y": row["y"],
                }
            )
        else:
            item["accuracy"] = row["y"]
        bucket.append(item)
    for name, data in [
        ("train.jsonl", train),
        ("test.jsonl", test),
        ("hidden_labels.jsonl", hidden),
    ]:
        (args.output_dir / name).write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in data)
        )
    print(
        json.dumps(
            {
                "train_examples": len(train),
                "test_examples": len(test),
                "train_cells": len(train_cells),
                "test_cells": len(test_cells),
            }
        )
    )


if __name__ == "__main__":
    main()
