"""CPU-only delta experiments: complete labels, grouped selection, frozen forecasts.

The published-reference cohort and the strict measured-parent sensitivity are
separate experiments. No raw inventory, incomplete delta rows, paid inference,
or test labels enter fitting, preprocessing, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
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

from tools.outcome_prediction.wm_clean_benchmark import weighted_median
from tools.outcome_prediction.wm_clean_models import CleanPredictor
from tools.outcome_prediction.wm_clean_refresh import read, sha, write
from tools.outcome_prediction.wm_code_benchmark import paired_comparison
from tools.outcome_prediction.wm_small_benchmark import evaluate, weighted_mean
from tools.outcome_prediction.wm_small_data import folds
from tools.outcome_prediction.wm_small_models import SmallPredictor

DEFAULT_DATA = Path("data/analysis/wm_delta_reference/01406da734fb_v1")
DEFAULT_BUNDLE = Path("data/analysis/wm_one_step_predictors/v1_bundle")
DEFAULT_OUTPUT = Path("data/analysis/wm_one_step_predictors/v1_results")
SEED = 20260906
# Fixed before fits. Absolute models are same-row controls, not selectable delta arms.
SPECS = {
    **{
        f"ridge_{view}_{alpha}": ["small", "ridge_delta", view, alpha]
        for view in ("parent", "current", "history") for alpha in (10, 100)
    },
    "hgb_current": ["small", "hgb_delta", "current", 10],
    "hgb_history": ["small", "hgb_delta", "history", 10],
    "legacy_hgb_current": ["legacy", "old_delta", "current", None],
    "regularized_hgb_current": ["legacy", "regularized_delta", "current", None],
    "absolute_ridge_control": ["small", "ridge_absolute", "current", 100],
    "absolute_hgb_control": ["small", "hgb_absolute", "current", 10],
}
BASELINES = ("parent_unchanged", "mean_delta", "median_delta", "median_delta_by_reference")
SELECTABLE = (*BASELINES, *(k for k in SPECS if not k.startswith("absolute_")))
ARMS = (*BASELINES, *SPECS)
PRIMARY = "train_selected"


def accuracy(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Both target and reference accuracies must be known, finite and in [0,1]")
    return float(value)


def validate_rows(rows, labels=None):
    ids = [r["example_id"] for r in rows]
    if len(ids) != len(set(ids)) or (labels is not None and set(ids) != set(labels)):
        raise ValueError("Duplicate or misaligned supervised identities")
    for row in rows:
        reference = accuracy(row.get("parent_reference"))
        if row.get("reference_kind") not in {"measured_parent", "published_base"}:
            raise ValueError("Unverified reference kind")
        if labels is not None:
            label = labels[row["example_id"]]
            target = accuracy(label.get("accuracy"))
            if label.get("delta_accuracy") != target - reference:
                raise ValueError("Missing or inconsistent delta label")
        if not row.get("cell_id"):
            raise ValueError("Missing session identity")


def regimes_for(rows):
    result = {}
    for cohort in ("combined", "measured_parent"):
        for benchmark in sorted({r["benchmark"] for r in rows}):
            selected = [r for r in rows if r["benchmark"] == benchmark and (
                cohort == "combined" or r["reference_kind"] == "measured_parent"
            )]
            parts = {p: [r for r in selected if r["split"] == p] for p in ("train", "test")}
            if not all(parts.values()):
                continue
            if {r["cell_id"] for r in parts["train"]} & {r["cell_id"] for r in parts["test"]}:
                raise ValueError("Train/test session overlap")
            result[cohort + "__" + benchmark] = {
                "benchmark": benchmark, "cohort": cohort,
                **{p + "_ids": sorted(r["example_id"] for r in rs) for p, rs in parts.items()},
            }
    return result


def prepare(data_bundle, output):
    from tools.outcome_prediction.wm_delta_reference_data import load_partition
    from tools.outcome_prediction.wm_one_step_features import features

    data_bundle, output = Path(data_bundle), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new experiment bundle; never overwrite frozen results")
    rows, labels = [], {}
    for part in ("train", "test"):
        examples, targets = load_partition(data_bundle, part)
        for example in examples:
            key = example["example_id"]
            if key in labels:
                raise ValueError("Train/test target overlap")
            rows.append({
                **{k: example[k] for k in ("example_id", "cell_id", "benchmark", "scientist_model")},
                "split": part, "parent_reference": example["parent"]["accuracy"],
                "reference_kind": example["reference_kind"],
                "views": {mode: features(example, mode=mode) for mode in ("parent", "current", "history")},
            })
            labels[key] = targets[key]
    validate_rows(rows, labels)
    regimes = regimes_for(rows)
    # Freeze all outcome-prediction modules: no hidden extractor/model dependency
    # may drift between preparation and fitting. This reads, never edits, sources.
    sources = {str(p.resolve()): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    policy = {
        "schema": "one-step-clean-delta-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_bundle": str(data_bundle.resolve()),
        "data_manifest_sha256": sha(data_bundle / "manifest.json"),
        "sources": sources, "specs": SPECS, "baselines": BASELINES,
        "selectable": SELECTABLE, "primary": PRIMARY, "seed": SEED,
        "selection": "Minimum equal-session MAE in up to five TRAIN-only grouped folds, fixed-name tie break; separate model per benchmark and cohort.",
        "clean_only": "Every fit/validation/test target has valid target and parent/reference accuracy. No incomplete examples in preprocessing or auxiliary training; no accuracy imputation.",
        "references": "Published fixed-base references and archive-measured parent scores distinguished; measured-parent-only sensitivity also fitted.",
        "primary_metric": "Equal-session MAE; ordinary per-experiment MAE, RMSE, final-accuracy R2 and delta R2 also reported.",
        "target": "Delta accuracy = target minus supplied verified reference. Final forecasts clipped to [0,1]. Absolute controls use identical complete-delta rows.",
        "features": "Fixed positive numeric recipe/code whitelist; no IDs, scientist identity, target outcomes, arbitrary text embeddings, or ancestor scores. Immediate parent accuracy allowed.",
        "missing_settings": "Missing recipe settings may have indicators; this never substitutes a missing accuracy.",
        "baselines_note": "Mean/median delta are predictions learned from complete training labels, not replacements for missing reference/target accuracy.",
        "llm": "No paid calls or cached forecasts used for fitting/selection. A fair RPM comparison requires matching prospective prompt inputs and demonstration banks.",
        "test_status": "Previously explored session split: developmental evidence, not a new untouched test. No test-error-driven tuning in this run.",
    }
    output.mkdir(parents=True, mode=0o700)
    for name, value in (("inputs.json", rows), ("labels.json", labels), ("regimes.json", regimes), ("policy.json", policy)):
        write(output / name, value)
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return {"rows": len(rows), "regimes": {k: {p: len(v[p + "_ids"]) for p in ("train", "test")} for k, v in regimes.items()}}


def baseline_predictions(train, labels, test):
    validate_rows(train, labels)
    validate_rows(test)
    delta = [labels[r["example_id"]]["delta_accuracy"] for r in train]
    groups = [r["cell_id"] for r in train]
    mean, median = weighted_mean(delta, groups), weighted_median(delta, groups)
    by_kind = {}
    for kind in sorted({r["reference_kind"] for r in train}):
        indices = [i for i, r in enumerate(train) if r["reference_kind"] == kind]
        by_kind[kind] = weighted_median([delta[i] for i in indices], [groups[i] for i in indices])
    reference = np.asarray([r["parent_reference"] for r in test])
    return {
        "parent_unchanged": reference.copy(),
        "mean_delta": np.clip(reference + mean, 0, 1),
        "median_delta": np.clip(reference + median, 0, 1),
        "median_delta_by_reference": np.clip(reference + [by_kind.get(r["reference_kind"], median) for r in test], 0, 1),
    }


def fit_suite(train, labels, test, model_dir=None):
    predictions = baseline_predictions(train, labels, test)
    y = [labels[r["example_id"]]["accuracy"] for r in train]
    reference, test_reference = ([r["parent_reference"] for r in rs] for rs in (train, test))
    groups = [r["cell_id"] for r in train]
    records = {}
    for name, (implementation, family, view, alpha) in SPECS.items():
        model = SmallPredictor(family, alpha=alpha, seed=SEED) if implementation == "small" else CleanPredictor(family)
        model.fit([r["views"][view] for r in train], y, reference, groups)
        predictions[name] = model.predict([r["views"][view] for r in test], test_reference)
        records[name] = {"view": view, "metadata": model.metadata()}
        if model_dir is not None:
            path = Path(model_dir) / (name + ".joblib")
            if path.exists():
                raise FileExistsError(path)
            joblib.dump(model, path, compress=3)
            path.chmod(0o600)
            records[name].update(model_path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    return predictions, records


def select_and_fit(train, labels, test, model_dir=None):
    """Deliberately accepts no held-out labels. Selection uses grouped TRAIN OOF."""
    validate_rows(train, labels)
    validate_rows(test)
    if {r["cell_id"] for r in train} & {r["cell_id"] for r in test}:
        raise ValueError("Train/test session overlap")
    if {r["example_id"] for r in train} & {r["example_id"] for r in test}:
        raise ValueError("Train/test target overlap")
    group_count = len({r["cell_id"] for r in train})
    if group_count < 2 or not test:
        raise ValueError("Need at least two training sessions and a nonempty test")
    fold_of = folds(train, k=min(5, group_count))
    inner = {name: np.full(len(train), np.nan) for name in ARMS}
    fold_records = []
    for fold in sorted(set(fold_of.values())):
        fit = [r for r in train if fold_of[r["cell_id"]] != fold]
        indices = [i for i, r in enumerate(train) if fold_of[r["cell_id"]] == fold]
        valid = [train[i] for i in indices]
        predictions, record = fit_suite(fit, {r["example_id"]: labels[r["example_id"]] for r in fit}, valid)
        for name in ARMS:
            inner[name][indices] = predictions[name]
        fold_records.append({
            "fold": fold, "train_ids": [r["example_id"] for r in fit],
            "validation_ids": [r["example_id"] for r in valid], "fits": record,
        })
    if not all(np.isfinite(p).all() for p in inner.values()):
        raise ValueError("Incomplete out-of-fold forecasts")
    y = [labels[r["example_id"]]["accuracy"] for r in train]
    cv = {name: evaluate(train, y, p) for name, p in inner.items()}
    selected = min(SELECTABLE, key=lambda name: (cv[name]["mae"], name))
    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, mode=0o700, exist_ok=False)
    predictions, fitted = fit_suite(train, labels, test, model_dir)
    predictions[PRIMARY] = predictions[selected].copy()
    return predictions, {
        "selected": selected, "cv_metrics": cv, "fold_of": fold_of,
        "fold_records": fold_records, "final_fits": fitted,
        "train_ids": [r["example_id"] for r in train], "test_ids": [r["example_id"] for r in test],
        "oof_predictions": {r["example_id"]: {name: float(p[i]) for name, p in inner.items()} for i, r in enumerate(train)},
    }


def validate_bundle(bundle):
    from tools.outcome_prediction.wm_delta_reference_data import load_partition

    bundle = Path(bundle)
    manifest = read(bundle / "manifest.json")
    required = {"inputs.json", "labels.json", "regimes.json", "policy.json"}
    if not required.issubset(manifest):
        raise ValueError("Incomplete experiment manifest")
    for name, expected in manifest.items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()) or sha(path) != expected:
            raise ValueError("Changed or invalid experiment file: " + name)
    policy = read(bundle / "policy.json")
    for path, expected in policy["sources"].items():
        if sha(path) != expected:
            raise ValueError("Changed frozen training source: " + path)
    data = Path(policy["data_bundle"])
    if sha(data / "manifest.json") != policy["data_manifest_sha256"]:
        raise ValueError("Changed source data manifest")
    for part in ("train", "test"):
        load_partition(data, part)
    if policy["specs"] != SPECS or policy["selectable"] != list(SELECTABLE):
        raise ValueError("Changed model selection specification")


def run(bundle, output):
    bundle, output = Path(bundle), Path(output)
    validate_bundle(bundle)
    rows, labels, regimes = (read(bundle / name) for name in ("inputs.json", "labels.json", "regimes.json"))
    validate_rows(rows, labels)
    if regimes != regimes_for(rows):
        raise ValueError("Changed regime definitions")
    by_id = {r["example_id"]: r for r in rows}
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    started = time.perf_counter()
    write(output / "run_policy.json", {
        "bundle": str(bundle.resolve()), "manifest_sha256": sha(bundle / "manifest.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "packages": {n: version(n) for n in ("numpy", "scipy", "scikit-learn", "joblib")},
        "no_llm_calls": True,
    })
    frozen, training = {}, {}
    for name, regime in regimes.items():
        train, test = ([by_id[k] for k in regime[p + "_ids"]] for p in ("train", "test"))
        predictions, detail = select_and_fit(train, {r["example_id"]: labels[r["example_id"]] for r in train}, test, output / name)
        frozen[name] = {r["example_id"]: {m: float(p[i]) for m, p in predictions.items()} for i, r in enumerate(test)}
        training[name] = detail
        print(f"{name}: forecasts complete; TRAIN selector={detail['selected']}", flush=True)
    write(output / "predictions.json", frozen)
    write(output / "training.json", training)
    write(output / "prediction_freeze.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "predictions_sha256": sha(output / "predictions.json"), "training_sha256": sha(output / "training.json"),
        "note": "All test forecasts persisted before aggregate test scoring; retrospective development holdout, not sealed-label execution.",
    })
    report = {}
    for name, regime in regimes.items():
        test = [by_id[k] for k in regime["test_ids"]]
        y = np.asarray([labels[r["example_id"]]["accuracy"] for r in test])
        reference = np.asarray([r["parent_reference"] for r in test])
        predictions = {m: np.asarray([frozen[name][r["example_id"]][m] for r in test]) for m in (*ARMS, PRIMARY)}
        subsets = {"all": np.ones(len(test), dtype=bool), **{kind: np.asarray([r["reference_kind"] == kind for r in test]) for kind in ("published_base", "measured_parent")}}
        metrics = {}
        for m, p in predictions.items():
            metrics[m] = {}
            for sub, mask in subsets.items():
                selected_rows = [r for r, keep in zip(test, mask) if keep]
                metrics[m][sub] = {
                    "accuracy": evaluate(selected_rows, y[mask], p[mask]),
                    "delta": evaluate(selected_rows, (y - reference)[mask], (p - reference)[mask]),
                }
        report[name] = {
            "train_n": len(regime["train_ids"]), "test_n": len(test),
            "train_sessions": len({by_id[k]["cell_id"] for k in regime["train_ids"]}),
            "train_reference_counts": dict(Counter(by_id[k]["reference_kind"] for k in regime["train_ids"])),
            "selected_using_train_only": training[name]["selected"], "metrics": metrics,
            "selected_comparisons": {m: paired_comparison(test, y, predictions[PRIMARY], predictions[m]) for m in ("parent_unchanged", "median_delta_by_reference", "legacy_hgb_current")},
        }
    write(output / "report.json", report)
    write(output / "manifest.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
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
        print(json.dumps({k: {"selected": v["selected_using_train_only"], "metrics": v["metrics"][PRIMARY]["all"]} for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
