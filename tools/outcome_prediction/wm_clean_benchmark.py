"""Fixed small-predictor refits on the clean labeled recorder cohort.

No API calls or paid inference. Prepare freezes inputs, sources, splits and model
specifications; run persists every held-out forecast before reporting errors.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from tools.outcome_prediction.wm_clean_refresh import load_jsonl, load_partition, read, sha, write
from tools.outcome_prediction.wm_code_benchmark import (
    checked_features,
    history_features,
    paired_comparison,
    positive_input,
)
from tools.outcome_prediction.wm_small_benchmark import evaluate, weighted_mean
from tools.outcome_prediction.wm_small_data import folds

SPECS = {
    "old_delta": ("old_delta", "parent"),
    "old_absolute": ("old_absolute", "parent"),
    "current_delta": ("old_delta", "current"),
    "history_delta": ("old_delta", "history"),
    "code_delta": ("old_delta", "code_parent"),
    "regularized_delta": ("regularized_delta", "code_parent"),
    "ridge_delta": ("ridge_delta", "code_parent"),
}
BASELINES = ("train_mean", "train_median", "parent_or_train_mean")
SELECTABLE = (*SPECS, *BASELINES, "fixed_blend")
PRIMARY = "inner_selected"
DEFAULT_DATA = Path("data/analysis/wm_clean/01406da734fb_v2")
DEFAULT_BUNDLE = Path("data/analysis/wm_clean_predictors/v1_bundle")
DEFAULT_OUTPUT = Path("data/analysis/wm_clean_predictors/v1_results")


def weighted_median(labels, groups):
    labels = np.asarray(labels, dtype=float)
    if (
        labels.ndim != 1
        or len(labels) != len(groups)
        or not len(labels)
        or not np.isfinite(labels).all()
    ):
        raise ValueError("Invalid labels/groups for weighted median")
    counts = Counter(groups)
    order = np.argsort(labels, kind="stable")
    weights = np.asarray([1 / counts[g] for g in groups])
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
    return float(labels[order[min(index, len(order) - 1)]])


def make_regimes(examples):
    ids = [r["example_id"] for r in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate examples")
    cells = {}
    for row in examples:
        if row["split"] not in {"train", "test"} or row["source_arm"] not in {"old", "new"}:
            raise ValueError("Unknown split/source arm")
        identity = (row["benchmark"], row["split"], row["source_arm"], row["scientist_model"])
        if row["cell_id"] in cells and cells[row["cell_id"]] != identity:
            raise ValueError("Conflicting session identity or train/test overlap")
        cells[row["cell_id"]] = identity
    result = {}

    def add(name, benchmark, train, test):
        if not train or not test:
            return
        if {r["cell_id"] for r in train} & {r["cell_id"] for r in test}:
            raise ValueError("Train/test session overlap")
        result[name + "__" + benchmark] = {
            "benchmark": benchmark,
            "train_ids": sorted(r["example_id"] for r in train),
            "test_ids": sorted(r["example_id"] for r in test),
        }

    for benchmark in sorted({r["benchmark"] for r in examples}):
        rows = [r for r in examples if r["benchmark"] == benchmark]
        train, test = ([r for r in rows if r["split"] == part] for part in ("train", "test"))
        add("session_holdout", benchmark, train, test)
        if benchmark == "aime2025":
            add("old_data_only", benchmark, [r for r in train if r["source_arm"] == "old"], test)
            add(
                "new_scientist_holdout",
                benchmark,
                [r for r in rows if r["source_arm"] == "old"],
                [r for r in rows if r["source_arm"] == "new"],
            )
    return result


def add_code_views(examples, inventory):
    from tools.outcome_prediction.wm_code_data_features import extract_data_features
    from tools.outcome_prediction.wm_code_features import extract_config_features

    memo, audits = {}, {}
    for row in examples:
        key = row["example_id"]
        payload = positive_input(inventory[key])
        config, config_audit = extract_config_features(payload)
        data, data_audit = extract_data_features(payload)
        memo[key] = {
            **{"config." + k: v for k, v in checked_features(config).items()},
            **{"data." + k: v for k, v in checked_features(data).items()},
        }
        audits[key] = {"config": config_audit, "data": data_audit}
    result = copy.deepcopy(examples)
    for row in result:
        current = memo[row["example_id"]]
        parents = [memo[k] for k in row["parent_ids"]]
        row["views"]["code_parent"] = {
            **row["views"]["parent"],
            **{"extra.current." + k: v for k, v in current.items()},
            **{"extra.parent." + k: v for k, v in history_features(current, parents).items()},
        }
    return result, audits


def prepare(data_bundle, output):
    from tools.outcome_prediction.wm_clean_context import build_context

    data_bundle, output = Path(data_bundle), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new immutable experiment bundle")
    started = time.perf_counter()
    examples, labels, context_audit = build_context(data_bundle)
    inventory = {r["example_id"]: r for r in load_jsonl(data_bundle / "private/inventory.jsonl")}
    # Metadata only; never feed archive source identity or split as a feature.
    split = read(data_bundle / "split.json")["cell_partition"]
    for row in examples:
        row["split"] = split[row["cell_id"]]
        row["source_arm"] = (
            "new"
            if inventory[row["example_id"]]["provenance"]["manifest_name"]
            == "manifest_aime2_r0.json"
            else "old"
        )
    examples, code_audit = add_code_views(examples, inventory)
    regimes = make_regimes(examples)
    source_paths = [
        Path(__file__),
        *[
            Path("tools/outcome_prediction") / n
            for n in (
                "wm_clean_context.py",
                "wm_clean_models.py",
                "wm_clean_refresh.py",
                "wm_small_data.py",
                "wm_small_models.py",
                "wm_small_benchmark.py",
                "wm_code_benchmark.py",
                "wm_code_features.py",
                "wm_code_data_features.py",
            )
        ],
    ]
    policy = {
        "schema": "clean-lightweight-predictors-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "specifications_frozen_before_real_fits",
        "data_bundle": str(data_bundle.resolve()),
        "data_manifest_sha256": sha(data_bundle / "manifest.json"),
        "sources": {str(p.resolve()): sha(p) for p in source_paths},
        "specs": SPECS,
        "primary": PRIMARY,
        "selection": "Lowest equal-session MAE in up to five TRAIN-only whole-session folds, with fixed-name tie break. No held-out-error choice or retuning. Includes trivial baselines and fixed blend.",
        "selectable": SELECTABLE,
        "old_method": "Clean refit of prior table's 3-seed 300-round HGB delta architecture, including original unnormalized equal-session weights. Positive safe features and stricter exact-parent matching differ from the old broader pipeline; not a numeric replication of the published old table.",
        "targets": "All supervised targets are clean officially labeled rows. Immediate exact-checkpoint parent score only when verified. Zero when unknown is solely mathematical centering: then residual target equals absolute accuracy, not a claimed known base score.",
        "primary_metric": "equal-session MAE of final accuracy, per benchmark; pooled MAE/R2 also reported",
        "regimes": "Primary: frozen clean session holdout. Secondary: old-only training on identical primary test targets, and old-AIME to new-scientist holdout. Regimes reuse labels in different roles; not an untouched prospective-test claim.",
        "code_features": "Existing positive pre-proposal config/data extractors, no additional recovery or arbitrary numeric/text embeddings.",
        "hybrid": "Fixed 50/50 regularized tree plus ridge; no LLM forecasts used or called.",
        "llm_comparison": "Unavailable on expanded clean cohort: cached forecasts have different banks/folds and no new AIME2 coverage. No fresh claim of beating inference-only LLM/RPM.",
        "scope_limit": "149 conservatively screened targets, not all 436 labels; all new Opus 4.6 records quarantined. Exact checkpoint bytes/execution unverified. Missing lineage/code coverage can limit delta usefulness.",
        "multiple_comparisons": "Every arm shown. Primary train-selected result frozen before test scoring; post-hoc best arm is exploratory, not deployable selection proof.",
        "extraction_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, mode=0o700)
    for name, value in (
        ("inputs.json", examples),
        ("labels.json", labels),
        ("regimes.json", regimes),
        ("policy.json", policy),
        ("context_audit.json", context_audit),
        ("code_audit.json", code_audit),
    ):
        write(output / name, value)
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return {
        "examples": len(examples),
        "regimes": {
            k: {"train": len(v["train_ids"]), "test": len(v["test_ids"])}
            for k, v in regimes.items()
        },
        "known_parent_scores": sum(r["reference_known"] for r in examples),
    }


def baseline_predictions(train, y, test):
    groups = [r["cell_id"] for r in train]
    mean, median = weighted_mean(y, groups), weighted_median(y, groups)
    return {
        "train_mean": np.full(len(test), mean),
        "train_median": np.full(len(test), median),
        "parent_or_train_mean": np.asarray(
            [r["parent_reference"] if r["reference_known"] else mean for r in test]
        ),
    }


def fit_suite(train, labels, test, model_dir=None):
    from tools.outcome_prediction.wm_clean_models import CleanPredictor

    y = np.asarray([labels[r["example_id"]] for r in train])
    ptrain, ptest = ([r["parent_reference"] for r in rs] for rs in (train, test))
    groups = [r["cell_id"] for r in train]
    predictions = baseline_predictions(train, y, test)
    records = {}
    for name, (kind, view) in SPECS.items():
        started = time.perf_counter()
        model = CleanPredictor(kind).fit([r["views"][view] for r in train], y, ptrain, groups)
        predictions[name] = model.predict([r["views"][view] for r in test], ptest)
        records[name] = {
            "view": view,
            "elapsed_seconds": time.perf_counter() - started,
            "metadata": model.metadata(),
        }
        if model_dir is not None:
            path = Path(model_dir) / (name + ".joblib")
            joblib.dump(model, path, compress=3)
            path.chmod(0o600)
            records[name].update(
                {
                    "model_path": str(path),
                    "model_sha256": sha(path),
                    "serialized_bytes": path.stat().st_size,
                }
            )
    predictions["fixed_blend"] = (predictions["regularized_delta"] + predictions["ridge_delta"]) / 2
    return predictions, records


def select_and_fit(rows_train, train_labels, rows_test, model_dir=None):
    """Test targets are not accepted by this function, including for selection."""
    ids = [r["example_id"] for r in rows_train]
    test_ids = [r["example_id"] for r in rows_test]
    if (
        len(ids) != len(set(ids))
        or len(test_ids) != len(set(test_ids))
        or set(ids) & set(test_ids)
        or set(ids) != set(train_labels)
    ):
        raise ValueError("Training label/row identities must match exactly")
    if {r["cell_id"] for r in rows_train} & {r["cell_id"] for r in rows_test}:
        raise ValueError("Train/test session overlap")
    for v in train_labels.values():
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Invalid training label")
    n_groups = len({r["cell_id"] for r in rows_train})
    if n_groups < 2 or not rows_test:
        raise ValueError("Need at least two training sessions and nonempty test")
    fold_of = folds(rows_train, k=min(5, n_groups))
    inner = {name: np.full(len(rows_train), np.nan) for name in SELECTABLE}
    inner_records = []
    for fold in sorted(set(fold_of.values())):
        train = [r for r in rows_train if fold_of[r["cell_id"]] != fold]
        val_idx = [i for i, r in enumerate(rows_train) if fold_of[r["cell_id"]] == fold]
        valid = [rows_train[i] for i in val_idx]
        preds, record = fit_suite(
            train, {r["example_id"]: train_labels[r["example_id"]] for r in train}, valid
        )
        for name in SELECTABLE:
            inner[name][val_idx] = preds[name]
        inner_records.append(
            {
                "fold": fold,
                "train_ids": [r["example_id"] for r in train],
                "validation_ids": [r["example_id"] for r in valid],
                "fits": record,
            }
        )
    if not all(np.isfinite(v).all() for v in inner.values()):
        raise ValueError("Unfilled inner predictions")
    y = [train_labels[r["example_id"]] for r in rows_train]
    inner_mae = {name: evaluate(rows_train, y, pred)["mae"] for name, pred in inner.items()}
    selected = min(SELECTABLE, key=lambda name: (inner_mae[name], name))
    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, mode=0o700, exist_ok=False)
    predictions, records = fit_suite(rows_train, train_labels, rows_test, model_dir)
    predictions[PRIMARY] = predictions[selected].copy()
    details = {
        "selected": selected,
        "inner_mae": inner_mae,
        "inner_fold_of": fold_of,
        "inner_fits": inner_records,
        "final_fits": records,
        "train_ids": ids,
        "test_ids": test_ids,
    }
    return predictions, details


def validate_bundle(bundle):
    bundle = Path(bundle)
    for name, expected in read(bundle / "manifest.json").items():
        if sha(bundle / name) != expected:
            raise ValueError("Changed experiment bundle: " + name)
    policy = read(bundle / "policy.json")
    data = Path(policy["data_bundle"])
    if sha(data / "manifest.json") != policy["data_manifest_sha256"]:
        raise ValueError("Changed clean dataset manifest")
    for partition in ("train", "test"):
        load_partition(data, partition)
    for path, expected in policy["sources"].items():
        if sha(path) != expected:
            raise ValueError("Changed frozen source: " + path)
    if policy["specs"] != {k: list(v) for k, v in SPECS.items()}:
        raise ValueError("Changed model specifications")


def run(bundle, output):
    bundle, output = Path(bundle), Path(output)
    validate_bundle(bundle)
    examples, labels, regimes = (
        read(bundle / name) for name in ("inputs.json", "labels.json", "regimes.json")
    )
    by_id = {r["example_id"]: r for r in examples}
    if (
        len(by_id) != len(examples)
        or set(by_id) != set(labels)
        or regimes != make_regimes(examples)
    ):
        raise ValueError("Dataset/regime identity mismatch")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    started = time.perf_counter()
    write(
        output / "run_policy.json",
        {
            "bundle": str(bundle.resolve()),
            "manifest_sha256": sha(bundle / "manifest.json"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")
            },
            "no_llm_calls": True,
        },
    )
    frozen, training = {}, {}
    for name, regime in regimes.items():
        train, test = ([by_id[k] for k in regime[part + "_ids"]] for part in ("train", "test"))
        pred, detail = select_and_fit(
            train, {r["example_id"]: labels[r["example_id"]] for r in train}, test, output / name
        )
        frozen[name] = {
            r["example_id"]: {m: float(p[i]) for m, p in pred.items()} for i, r in enumerate(test)
        }
        training[name] = detail
        print(
            name + ": fits/predictions complete; TRAIN-only selector=" + detail["selected"],
            flush=True,
        )
    write(output / "predictions.json", frozen)
    write(output / "training.json", training)
    write(
        output / "prediction_freeze.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "predictions_sha256": sha(output / "predictions.json"),
            "training_sha256": sha(output / "training.json"),
            "elapsed_seconds": time.perf_counter() - started,
            "note": "All regime predictions persisted before aggregate test scoring. This is a retrospective comparison, not sealed-label execution.",
        },
    )
    report = {}
    for name, regime in regimes.items():
        test = [by_id[k] for k in regime["test_ids"]]
        y = np.asarray([labels[r["example_id"]] for r in test])
        predictions = {
            m: np.asarray([frozen[name][r["example_id"]][m] for r in test])
            for m in (*SELECTABLE, PRIMARY)
        }
        known = np.asarray([r["reference_known"] for r in test])
        new = np.asarray([r["source_arm"] == "new" for r in test])
        subsets = {
            "all": np.ones(len(test), dtype=bool),
            "known_parent": known,
            "unknown_parent": ~known,
            "new_aime": new,
        }
        report[name] = {
            "train_examples": len(regime["train_ids"]),
            "train_sessions": len({by_id[k]["cell_id"] for k in regime["train_ids"]}),
            "selected_using_train_only": training[name]["selected"],
            "metrics": {
                m: {
                    sub: evaluate([r for r, keep in zip(test, mask) if keep], y[mask], p[mask])
                    for sub, mask in subsets.items()
                }
                for m, p in predictions.items()
            },
            "paired_vs_old_delta": {
                m: paired_comparison(test, y, p, predictions["old_delta"])
                for m, p in predictions.items()
                if m != "old_delta"
            },
        }
        if known.any():
            rows = [r for r, keep in zip(test, known) if keep]
            reference = np.asarray([r["parent_reference"] for r in rows])
            report[name]["known_parent_unchanged"] = evaluate(rows, y[known], reference)
            report[name]["delta_metrics_on_known_parent"] = {
                m: evaluate(rows, y[known] - reference, p[known] - reference)
                for m, p in predictions.items()
            }
    # Same test targets, adding only the new training examples.
    main, old = "session_holdout__aime2025", "old_data_only__aime2025"
    if main in regimes and old in regimes:
        if regimes[main]["test_ids"] != regimes[old]["test_ids"]:
            raise ValueError("More-data comparison changed test targets")
        rows = [by_id[k] for k in regimes[main]["test_ids"]]
        y = [labels[r["example_id"]] for r in rows]
        report["new_training_data_effect"] = {
            m: paired_comparison(
                rows,
                y,
                [frozen[main][r["example_id"]][m] for r in rows],
                [frozen[old][r["example_id"]][m] for r in rows],
            )
            for m in (*SELECTABLE, PRIMARY)
        }
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--data-bundle", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.data_bundle, args.bundle), indent=2))
    else:
        report = run(args.bundle, args.output)
        print(
            json.dumps(
                {
                    k: {
                        "selected": v["selected_using_train_only"],
                        "old_delta": v["metrics"]["old_delta"]["all"],
                        "inner_selected": v["metrics"][PRIMARY]["all"],
                    }
                    for k, v in report.items()
                    if "metrics" in v
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
