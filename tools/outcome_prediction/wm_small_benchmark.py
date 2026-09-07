"""Small-data, nested session-CV experiment-outcome benchmark.

Only prepare/run write exclusive new artifact directories. No agent API calls.
Saved agent forecasts are evaluation-only: their training-row banks include
outer-test labels, so no learned calibration or blend can be fitted from them.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from tools.outcome_prediction.wm_small_data import (
    BASE_REFERENCES,
    SEED,
    build_examples,
    cohort,
    load_jsonl,
    sha,
)

SPECS = {
    "reference_ridge": ("ridge_absolute", "reference", [10.0, 100.0]),
    "ridge_current_delta": ("ridge_delta", "current", [10.0, 100.0]),
    "ridge_parent_delta": ("ridge_delta", "parent", [10.0, 100.0]),
    "ridge_history_delta": ("ridge_delta", "history", [10.0, 100.0]),
    "ridge_history_absolute": ("ridge_absolute", "history", [10.0, 100.0]),
    "trees_current_delta": ("hgb_delta", "current", [10.0]),
    "trees_parent_delta": ("hgb_delta", "parent", [10.0]),
    "trees_history_delta": ("hgb_delta", "history", [10.0]),
    "trees_history_absolute": ("hgb_absolute", "history", [10.0]),
    "embedding_ridge_delta": ("embedding_ridge_delta", "history", [10.0, 100.0]),
    "embedding_kernel_delta": ("embedding_kernel_delta", "history", [1.0, 10.0]),
}
# Select model families/contexts only with inner TRAIN folds, never pooled OOF scores.
SELECTABLE = tuple(k for k in SPECS if k != "reference_ridge")
SOURCE_FILES = (
    "tools/outcome_prediction/wm_small_data.py",
    "tools/outcome_prediction/wm_small_models.py",
    "tools/outcome_prediction/wm_small_embeddings.py",
    "tools/outcome_prediction/wm_small_benchmark.py",
    "tools/outcome_prediction/wm_grade_inventory.py",
    "tools/wm_study/agent_predict.py",
    "tools/wm_study/compare_predictors.py",
    "tools/wm_study/wm.py",
    "tools/wm_study/wm_delta.py",
    "tools/wm_study/decision_sets.py",
    "tools/wm_study/extract.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    Path(path).chmod(0o600)


def weighted_mean(values, groups):
    cells = defaultdict(list)
    for value, group in zip(values, groups):
        cells[group].append(value)
    return float(np.mean([np.mean(v) for v in cells.values()])) if cells else None


def evaluate(rows, labels, predictions):
    if len(rows) != len(labels) or len(rows) != len(predictions):
        raise ValueError("Mismatched evaluation lengths")
    y, p = np.asarray(labels), np.asarray(predictions)
    if not len(rows):
        return {"n": 0, "sessions": 0, "mae": None, "rmse": None, "r2": None}
    groups = [r["cell_id"] for r in rows]
    mean = weighted_mean(y, groups)
    mse, variance = weighted_mean((y - p) ** 2, groups), weighted_mean((y - mean) ** 2, groups)
    return {
        "n": len(rows),
        "sessions": len(set(groups)),
        "mae": weighted_mean(np.abs(y - p), groups),
        "rmse": float(np.sqrt(mse)),
        "r2": 1 - mse / variance if variance > 0 else None,
        "pooled_card_mae": float(np.abs(y - p).mean()),
        "pooled_card_r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        if np.var(y) > 0
        else None,
    }


def bootstrap_gain(rows, labels, prediction, reference, repeats=2000):
    groups = defaultdict(list)
    for row, y, p, b in zip(rows, labels, prediction, reference):
        groups[row["cell_id"]].append((abs(y - p), abs(y - b)))
    values = np.asarray([np.mean(groups[k], axis=0) for k in sorted(groups)])
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    relative = 1 - samples[:, 0] / np.maximum(samples[:, 1], 1e-12)
    return {
        "relative_mae_reduction": float(1 - values[:, 0].mean() / values[:, 1].mean()),
        "ci95": np.percentile(relative, [2.5, 97.5]).tolist(),
        "sessions": len(groups),
        "replicates": repeats,
        "interpretation": "exploratory session bootstrap of OOF errors; shared CV training sets and prior exploration limit confirmation",
    }


def prepare(cards_path, inventory_path, agent_dir, output):
    cards = load_jsonl(cards_path)
    inventory = {r["example_id"]: r for r in load_jsonl(inventory_path)}
    selected, excluded = cohort(cards)
    examples, labels = build_examples(cards, inventory, selected)
    agent, agent_hashes = {}, {}
    for r in examples:
        path = Path(agent_dir) / r["benchmark"] / (r["example_id"].replace("/", "__") + ".json")
        record = read(path)
        value = (record.get("pred") or {}).get("predicted_accuracy")
        if (
            record["example_id"] != r["example_id"]
            or record["fold"] != r["fold"]
            or abs(record["parent_acc"] - r["parent_reference"]) > 1e-12
            or abs(record["true"] - labels[r["example_id"]]) > 1e-12
            or record["meta"].get("is_error")
            or type(value) not in (int, float)
            or not np.isfinite(value)
        ):
            raise ValueError("Incompatible saved agent prediction: " + r["example_id"])
        agent[r["example_id"]] = float(np.clip(value, 0, 1))
        agent_hashes[str(path.resolve())] = sha(path)
    paths = [Path(cards_path), Path(inventory_path), *(Path(p) for p in SOURCE_FILES)]
    policy = {
        "schema": "small-outcome-predictors-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "fixed_before_new_fits",
        "seed": SEED,
        "outer_folds": 8,
        "inner_folds": 3,
        "specs": SPECS,
        "selectable": SELECTABLE,
        "primary_result": "inner-selected predictor evaluated on outer held-out sessions",
        "primary_metric": "equal-session MAE, per benchmark",
        "bootstrap_repeats": 2000,
        "reference_constants": BASE_REFERENCES,
        "reference_warning": "Legacy base constants are NOT verified official base scores; multi-parent means are references, not a real single-parent checkpoint score.",
        "features": "positive first-plan settings + static whitelisted settings/calls from reconstructed pre-proposal code; current first-record-closed or first-result-nonempty candidates have proposal features masked. n_examples/generated yields/progress/hp.other excluded; generic history has NO outcome labels. Frozen general-text embeddings of canonical safe summaries, not full code semantics.",
        "context": "current proposal plus recipe history and only declared weight-parent reference score; older ancestor scores omitted to match saved agent information. Retrospective-GIVEN, not historical grade availability.",
        "cohort": "Same broader legacy 178 GSM +156 AIME cohort; no sibling requirement. No new score-gap or prediction-error filtering.",
        "comparison_limit": "Saved agent saw broader scrubbed legacy snapshots/text and a train bank; WM sees positive first-plan/reconstructed-code summaries. Same targets/folds/parent references, not bit-identical contexts or certified launch artifacts.",
        "hybrid": "Fixed 50/50 held-out forecast blend only. Never fit on saved TRAIN agent forecasts: those banks contain outer-TEST labels.",
        "learning_curves": {
            "models": ["ridge_history_delta", "trees_history_delta"],
            "train_sessions": [8, 16, "all"],
            "fixed_alpha": 10.0,
            "selection": "deterministic nested subsets of outer TRAIN sessions; no full-TRAIN tuning for subset models",
        },
        "selection_limit": "Existing release previously explored. No confirmatory claim, no outer-label hyperparameter tuning, no post-results variant addition.",
        "sources": {str(p.resolve()): sha(p) for p in paths},
        "agent_files": agent_hashes,
    }
    output = Path(output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    write(output / "policy.json", policy)
    write(output / "inputs.json", examples)
    write(output / "labels.json", labels)
    write(output / "agent_predictions.json", agent)
    coverage = {
        b: {
            "examples": sum(r["benchmark"] == b for r in examples),
            "sessions": len({r["cell_id"] for r in examples if r["benchmark"] == b}),
            "base": sum(r["benchmark"] == b and r["from_base"] for r in examples),
            "continuation": sum(r["benchmark"] == b and not r["from_base"] for r in examples),
            "initial_fidelity_eligible": sum(
                r["benchmark"] == b and r["audit"]["initial_fidelity_eligible"] for r in examples
            ),
            "fidelity_reviewed_eligible": sum(
                r["benchmark"] == b and r["audit"]["fidelity_reviewed_eligible"] for r in examples
            ),
            "proposal_features_available": sum(
                r["benchmark"] == b and r["audit"]["proposal_features_available"] for r in examples
            ),
        }
        for b in BASE_REFERENCES
    }
    write(output / "coverage.json", {"included": coverage, "excluded": excluded})
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return coverage


def fit_one(rows, labels, train, test, spec, alpha, embeddings):
    from tools.outcome_prediction.wm_small_models import SmallPredictor

    name, view, _ = spec
    features = [r["views"][view] for r in rows]
    p = np.asarray([r["parent_reference"] for r in rows])
    use_embeddings = name.startswith("embedding_")
    tr_embedding = embeddings[train] if use_embeddings else None
    te_embedding = embeddings[test] if use_embeddings else None
    start = time.perf_counter()
    model = SmallPredictor(name, alpha=alpha, seed=SEED).fit(
        [features[i] for i in train],
        labels[train],
        p[train],
        [rows[i]["cell_id"] for i in train],
        embeddings=tr_embedding,
    )
    fitted = time.perf_counter()
    prediction = model.predict([features[i] for i in test], p[test], embeddings=te_embedding)
    return (
        model,
        prediction,
        {"fit_seconds": fitted - start, "prediction_seconds": time.perf_counter() - fitted},
    )


def run(bundle, embedding_file, embedding_metadata, output):
    bundle, output = Path(bundle), Path(output)
    manifest, policy = read(bundle / "manifest.json"), read(bundle / "policy.json")
    for name, expected in manifest.items():
        if sha(bundle / name) != expected:
            raise ValueError("Modified bundle: " + name)
    for path, expected in {**policy["sources"], **policy["agent_files"]}.items():
        if sha(path) != expected:
            raise ValueError("Modified frozen source: " + path)
    examples, all_labels, agent = (
        read(bundle / "inputs.json"),
        read(bundle / "labels.json"),
        read(bundle / "agent_predictions.json"),
    )
    embedding = np.load(embedding_file, allow_pickle=False)
    emeta = read(embedding_metadata)
    if (
        embedding.ndim != 2
        or len(embedding) != len(examples)
        or not np.isfinite(embedding).all()
        or emeta.get("input_sha256") != sha(bundle / "inputs.json")
        or emeta.get("embeddings_sha256") != sha(embedding_file)
    ):
        raise ValueError("Embedding input/artifact mismatch")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    write(
        output / "run_policy.json",
        {
            "bundle_manifest_sha256": sha(bundle / "manifest.json"),
            "policy_sha256": sha(bundle / "policy.json"),
            "embedding_file_sha256": sha(embedding_file),
            "embedding_metadata": emeta,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    folds_meta, oof, curve_oof, model_hashes = {}, {}, {}, {}
    for benchmark in BASE_REFERENCES:
        indices = [i for i, r in enumerate(examples) if r["benchmark"] == benchmark]
        rows, emb = [examples[i] for i in indices], embedding[indices]
        y = np.asarray([all_labels[r["example_id"]] for r in rows])
        predictions = {name: np.full(len(rows), np.nan) for name in SPECS}
        predictions["inner_selected"] = np.full(len(rows), np.nan)
        curve_predictions = {
            name + "__" + str(n): np.full(len(rows), np.nan)
            for name in ("ridge_history_delta", "trees_history_delta")
            for n in (8, 16, "all")
        }
        folds_meta[benchmark] = []
        for outer in range(8):
            tr = np.array([i for i, r in enumerate(rows) if r["fold"] != outer], dtype=int)
            te = np.array([i for i, r in enumerate(rows) if r["fold"] == outer], dtype=int)
            if not len(te):
                continue
            training_groups = sorted({rows[i]["cell_id"] for i in tr})
            assert not set(training_groups) & {rows[i]["cell_id"] for i in te}
            rng = np.random.default_rng(SEED + outer)
            shuffled = list(rng.permutation(training_groups))
            inner_group = {g: i % 3 for i, g in enumerate(shuffled)}
            fold_record = {
                "outer_fold": outer,
                "train_sessions": training_groups,
                "test_sessions": sorted({rows[i]["cell_id"] for i in te}),
                "models": {},
            }
            for name, spec in SPECS.items():
                candidates = []
                for alpha in spec[2]:
                    inner_pred = np.full(len(rows), np.nan)
                    for inner in range(3):
                        itr = np.array(
                            [i for i in tr if inner_group[rows[i]["cell_id"]] != inner], dtype=int
                        )
                        ite = np.array(
                            [i for i in tr if inner_group[rows[i]["cell_id"]] == inner], dtype=int
                        )
                        _, pred, _ = fit_one(rows, y, itr, ite, spec, alpha, emb)
                        inner_pred[ite] = pred
                    loss = evaluate([rows[i] for i in tr], y[tr], inner_pred[tr])["mae"]
                    candidates.append({"alpha": alpha, "inner_mae": loss})
                chosen = min(candidates, key=lambda c: (c["inner_mae"], c["alpha"]))
                model, pred, timings = fit_one(rows, y, tr, te, spec, chosen["alpha"], emb)
                predictions[name][te] = pred
                model_path = output / f"{benchmark}__fold{outer}__{name}.joblib"
                joblib.dump(model, model_path, compress=3)
                model_path.chmod(0o600)
                model_hashes[model_path.name] = sha(model_path)
                fold_record["models"][name] = {
                    "candidates": candidates,
                    "selected": chosen,
                    "timing": timings,
                    "serialized_bytes": model_path.stat().st_size,
                    "metadata": model.metadata(),
                }
            winner = min(
                SELECTABLE, key=lambda n: (fold_record["models"][n]["selected"]["inner_mae"], n)
            )
            predictions["inner_selected"][te] = predictions[winner][te]
            fold_record["inner_selected"] = winner
            for name in ("ridge_history_delta", "trees_history_delta"):
                for size in (8, 16, "all"):
                    selected_groups = set(shuffled if size == "all" else shuffled[:size])
                    subset = np.array(
                        [i for i in tr if rows[i]["cell_id"] in selected_groups], dtype=int
                    )
                    _, pred, _ = fit_one(rows, y, subset, te, SPECS[name], 10.0, emb)
                    curve_predictions[name + "__" + str(size)][te] = pred
            folds_meta[benchmark].append(fold_record)
            print(f"{benchmark} fold {outer + 1}/8 complete; inner-selected {winner}", flush=True)
        if any(
            not np.isfinite(p).all() for p in [*predictions.values(), *curve_predictions.values()]
        ):
            raise ValueError("Unfilled OOF predictions")
        oof[benchmark] = {
            r["example_id"]: {name: float(p[i]) for name, p in predictions.items()}
            for i, r in enumerate(rows)
        }
        curve_oof[benchmark] = {
            r["example_id"]: {name: float(p[i]) for name, p in curve_predictions.items()}
            for i, r in enumerate(rows)
        }
    write(output / "predictions.json", oof)
    write(output / "learning_curve_predictions.json", curve_oof)
    write(output / "training.json", folds_meta)
    write(output / "model_hashes.json", model_hashes)
    write(
        output / "prediction_freeze.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "predictions_sha256": sha(output / "predictions.json"),
            "training_sha256": sha(output / "training.json"),
            "note": "All outer predictions frozen before aggregate scoring. Inner scoring uses only that outer fold's TRAIN labels; full label file is decoded for CV, not a sealed-label claim.",
        },
    )
    report, learning = {}, {}
    for benchmark in BASE_REFERENCES:
        rows = [r for r in examples if r["benchmark"] == benchmark]
        y = np.asarray([all_labels[r["example_id"]] for r in rows])
        ag = np.asarray([agent[r["example_id"]] for r in rows])
        names = list(next(iter(oof[benchmark].values())))
        predictions = {
            name: np.asarray([oof[benchmark][r["example_id"]][name] for r in rows])
            for name in names
        }
        predictions["inference_only_saved"] = ag
        predictions["parent_reference_unchanged"] = np.asarray(
            [r["parent_reference"] for r in rows]
        )
        predictions["fixed_half_blend"] = (ag + predictions["inner_selected"]) / 2
        report[benchmark] = {}
        for name, p in predictions.items():
            entry = {
                "all": evaluate(rows, y, p),
                "vs_inference_only": bootstrap_gain(rows, y, p, ag),
            }
            for subset in (
                "base",
                "continuation",
                "initial_fidelity_eligible",
                "fidelity_reviewed_eligible",
                "proposal_features_available",
            ):
                ix = [
                    i
                    for i, r in enumerate(rows)
                    if (
                        r["audit"][subset]
                        if subset in r["audit"]
                        else (r["from_base"] if subset == "base" else not r["from_base"])
                    )
                ]
                entry[subset] = evaluate([rows[i] for i in ix], y[ix], p[ix])
            report[benchmark][name] = entry
        learning[benchmark] = {}
        for name in next(iter(curve_oof[benchmark].values())):
            pred = [curve_oof[benchmark][r["example_id"]][name] for r in rows]
            learning[benchmark][name] = evaluate(rows, y, pred)
    write(output / "report.json", report)
    write(output / "learning_curves.json", learning)
    return {
        b: {
            name: {
                "mae": m["all"]["mae"],
                "relative_gain": m["vs_inference_only"]["relative_mae_reduction"],
            }
            for name, m in models.items()
        }
        for b, models in report.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    p.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/analysis/wm_rpm/data_v3/private/inventory.jsonl"),
    )
    p.add_argument("--agent", type=Path, default=Path("data/analysis/wm_study/agent_predict"))
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("run")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--embedding-metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        prepare(args.cards, args.inventory, args.agent, args.output)
        if args.command == "prepare"
        else run(args.bundle, args.embeddings, args.embedding_metadata, args.output)
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
