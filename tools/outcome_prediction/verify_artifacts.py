"""Verify cohort reconstruction, held-out coverage, saved scores, and reproducibility."""

import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark import load_examples, predict_methods, regression_metrics, split_groups
from build_examples import build
from sensitivity import selected_ridge


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def verify_scores(records, metrics):
    y = np.array([r["y"] for r in records])
    for method, expected in metrics["methods"].items():
        p = np.array([r["predictions"][method] for r in records])
        calculated = regression_metrics(y, p)
        for key in ("mae", "rmse", "r2"):
            assert np.isclose(calculated[key], expected[key], atol=1e-12), (method, key)


def main():
    root = Path("data/analysis/outcome_prediction")
    rebuilt, audit = build(Path("data/traj/raw/awm-gsm8k-trajectories"))
    assert rebuilt == jsonl(root / "examples.jsonl"), "Saved examples differ from current extractor"
    rows, _ = load_examples(root / "examples.jsonl", False)
    ids = {r["example_id"] for r in rows}
    assert len(rows) == 146 and len({r["cell_id"] for r in rows}) == 32
    records = jsonl(root / "grouped/predictions.jsonl")
    assert {r["example_id"] for r in records} == ids and len(records) == len(rows)
    folds = json.loads((root / "grouped/fold_assignments.json").read_text())
    for fold in folds:
        assert set(fold["train_cells"]).isdisjoint(fold["test_cells"])
        part = [r for r in records if r["fold"] == fold["fold"]]
        assert {r["cell_id"] for r in part} == set(fold["test_cells"])
    verify_scores(records, json.loads((root / "grouped/metrics.json").read_text()))
    nested = jsonl(root / "sensitivity/nested_predictions.jsonl")
    verify_scores(nested, json.loads((root / "sensitivity/metrics.json").read_text())["nested"])
    # Refit one full fold to catch stale features or stale saved model outputs.
    tr, te = split_groups(rows, 8, 42)[0]
    train, test = [rows[i] for i in tr], [rows[i] for i in te]
    actual = predict_methods(train, test, 42)
    lookup = {r["example_id"]: r for r in records}
    for method, pred in actual.items():
        assert np.allclose(
            pred, [lookup[r["example_id"]]["predictions"][method] for r in test], atol=1e-10
        ), method
    actual_nested, _ = selected_ridge(train, test, "recipe_text", 42)
    nested_lookup = {r["example_id"]: r for r in nested}
    assert np.allclose(
        actual_nested,
        [nested_lookup[r["example_id"]]["predictions"]["recipe_text_nested"] for r in test],
        atol=1e-10,
    )
    by_id = {r["example_id"]: r for r in rows}
    test_cells_seen = set()
    for name in ("icl", "icl_fold1", "icl_fold2"):
        directory = root / name
        split = json.loads((directory / "split.json").read_text())
        assert set(split["train_cells"]).isdisjoint(split["test_cells"])
        assert test_cells_seen.isdisjoint(split["test_cells"])
        test_cells_seen.update(split["test_cells"])
        training = jsonl(directory / "train.jsonl")
        expected_training = [r for r in rows if r["cell_id"] in split["train_cells"]]
        assert len(training) == len(expected_training)
        for prompt, example in zip(training, expected_training, strict=True):
            assert prompt["accuracy"] == example["y"]
            assert prompt["recipe_sequence"] == [s["recipe"] for s in example["lineage"]]
        hidden = {r["id"]: r for r in jsonl(directory / "hidden_labels.jsonl")}
        tests = jsonl(directory / "test.jsonl")
        for prompt in tests:
            assert set(prompt) == {"id", "recipe_sequence"}
            target = by_id[hidden[prompt["id"]]["example_id"]]
            assert target["cell_id"] in split["test_cells"]
            assert prompt["recipe_sequence"] == [s["recipe"] for s in target["lineage"]]
        verify_scores(
            jsonl(directory / "scored_predictions.jsonl"),
            json.loads((directory / "metrics.json").read_text()),
        )
    combined = jsonl(root / "icl_combined/predictions.jsonl")
    assert len(combined) == 107 and len({r["example_id"] for r in combined}) == 107
    verify_scores(combined, json.loads((root / "icl_combined/metrics.json").read_text()))
    files = [
        root / "examples.jsonl",
        root / "audit.json",
        root / "report.md",
        root / "overview.png",
    ]
    files += list(root.glob("*/metrics.json"))
    files += list(Path("tools/outcome_prediction").glob("*.py"))
    summary = {
        "status": "passed",
        "cohort": audit["counts"],
        "grouped_oof_checkpoints": len(records),
        "blinded_checkpoints": len(combined),
        "blinded_runs": len(test_cells_seen),
        "checks": [
            "deterministic cohort reconstruction",
            "whole-run test separation and coverage",
            "saved metric recomputation",
            "first-fold statistical and nested-model refit",
            "blinded prompt/label boundary and cross-session disjointness",
        ],
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (root / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
