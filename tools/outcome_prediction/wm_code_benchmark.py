"""Fixed-tree ablations of positively extracted, decision-time code features.

New artifacts only; no agent calls, code execution, encoder, or model selection.
The existing release has already been explored, so all comparisons are exploratory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from tools.outcome_prediction.wm_small_benchmark import evaluate, read, write
from tools.outcome_prediction.wm_small_data import SEED, load_jsonl, proposal_available, sha, stamp
from tools.outcome_prediction.wm_small_models import HGB_PARAMETERS, SmallPredictor

SPECS = {
    "v1_current": ("current", (), "none"),
    "v1_parent": ("parent", (), "none"),
    "v1_history": ("history", (), "none"),
    "config_current": ("current", ("config",), "none"),
    "data_current": ("current", ("data",), "none"),
    "code_current": ("current", ("config", "data"), "none"),
    "code_only": ("reference", ("config", "data"), "none"),
    "code_parent": ("parent", ("config", "data"), "parents"),
    "code_history": ("history", ("config", "data"), "history"),
}
PRIMARY = "code_parent"
BASELINE_NAMES = {
    "v1_current": "trees_current_delta",
    "v1_parent": "trees_parent_delta",
    "v1_history": "trees_history_delta",
}
SUBSETS = ("base", "continuation", "proposal_features_available", "fidelity_reviewed_eligible")
NEW_SOURCES = (
    "tools/outcome_prediction/wm_code_benchmark.py",
    "tools/outcome_prediction/wm_code_features.py",
    "tools/outcome_prediction/wm_code_data_features.py",
)


def verify_bundle(bundle):
    bundle = Path(bundle)
    for name, expected in read(bundle / "manifest.json").items():
        if sha(bundle / name) != expected:
            raise ValueError("Modified source bundle: " + name)
    policy = read(bundle / "policy.json")
    for path, expected in {**policy["sources"], **policy["agent_files"]}.items():
        if sha(path) != expected:
            raise ValueError("Modified frozen source: " + path)


def positive_input(row):
    """Exclude outcomes, observations, generic counts, and unrelated plan prose."""
    source = row["model_input"]
    setup = (source.get("plan") or {}).get("setup") or {}
    command = setup.get("command") or {}
    method = setup.get("method") or {}
    method_view = {
        k: copy.deepcopy(method[k])
        for k in ("family", "framework", "peft", "target_format")
        if k in method
    }
    method_view["hyperparams"] = {
        k: copy.deepcopy(v)
        for k, v in (method.get("hyperparams") or {}).items()
        if k not in {"other", "notes", "result", "accuracy", "score", "progress"}
    }
    data = [
        {
            k: copy.deepcopy(d[k])
            for k in ("source", "build_command", "built_by", "mixture_weight", "selection")
            if k in d
        }
        for d in setup.get("data") or []
        if isinstance(d, dict)
    ]
    return {
        "plan": {
            "setup": {
                "method": method_view,
                "command": {
                    k: copy.deepcopy(command[k]) for k in ("argv", "script", "cwd") if k in command
                },
                "data": data,
            }
        },
        "code": [
            {k: copy.deepcopy(c[k]) for k in ("role", "status", "script_path", "content") if k in c}
            for c in source.get("code") or []
            if c.get("status") == "reconstructed"
        ],
    }


def checked_features(features):
    """Enforce a numeric fixed-schema boundary; categorical prose is not allowed."""
    if not isinstance(features, dict):
        raise TypeError("Feature extractor must return a dictionary")
    out = {}
    for key, value in features.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Invalid feature key")
        if value is None:
            out[key] = None
        elif isinstance(value, (float, int, np.number, bool)) and math.isfinite(float(value)):
            out[key] = float(value)
        else:
            raise ValueError("Nonfinite or nonnumeric code feature: " + key)
    return out


def history_features(current, ancestors):
    """Fixed-width mean/latest history and current-versus-latest interactions.

    Never sum declared maxima: they can be ceilings rather than achieved exposure.
    Unknown latest values stay unknown instead of silently taking an older value.
    """
    result = {"steps": float(len(ancestors))}
    for key in sorted(set(current).union(*(set(a) for a in ancestors))):
        values = [a.get(key) for a in ancestors if a.get(key) is not None]
        last = ancestors[-1].get(key) if ancestors else None
        result["mean." + key] = float(np.mean(values)) if values else None
        result["latest." + key] = last
        result["action_change." + key] = (
            current[key] - last if current.get(key) is not None and last is not None else None
        )
    return result


def build_views(examples, inventory, config_extractor=None, data_extractor=None):
    if config_extractor is None:
        from tools.outcome_prediction.wm_code_features import extract_config_features

        config_extractor = extract_config_features
    if data_extractor is None:
        from tools.outcome_prediction.wm_code_data_features import extract_data_features

        data_extractor = extract_data_features
    extractors = {"config": config_extractor, "data": data_extractor}
    memo, audit = {}, {}

    def step(identifier, history=False):
        memo_key = (identifier, history)
        if memo_key in memo:
            return memo[memo_key]
        row = inventory[identifier]
        if not history and not proposal_available(row):
            blocks = {name: {"available": 0.0} for name in extractors}
            evidence = {"masked": True, "reason": "v1_proposal_gate"}
        else:
            payload = positive_input(row)
            blocks, evidence = {}, {"masked": False, "blocks": {}}
            for name, extractor in extractors.items():
                features, detail = extractor(payload)
                blocks[name] = {"available": 1.0, **checked_features(features)}
                evidence["blocks"][name] = detail
        memo[memo_key] = blocks
        audit[identifier + ("::history" if history else "::current")] = evidence
        return blocks

    output = []
    for original in examples:
        identifier = original["example_id"]
        row = inventory[identifier]
        cutoff = stamp(row["first_submitted_at"])
        history_ids = list(original["history_ids"])
        if len(set(history_ids)) != len(history_ids):
            raise ValueError("Duplicate recipe-history step")
        for ancestor in history_ids:
            if (
                ancestor == identifier
                or ancestor not in inventory
                or inventory[ancestor]["cell_id"] != original["cell_id"]
                or stamp(inventory[ancestor]["first_submitted_at"]) >= cutoff
            ):
                raise ValueError("Unsafe recipe history: " + ancestor)
        history_ids.sort(key=lambda h: (stamp(inventory[h]["first_submitted_at"]), h))
        current = step(identifier)
        views = {}
        for name, (base_view, blocks, context) in SPECS.items():
            features = copy.deepcopy(original["views"][base_view])
            chosen = (
                []
                if context == "none"
                else [h for h in history_ids if context == "history" or h in original["parent_ids"]]
            )
            for block in blocks:
                features.update(
                    {"extra.current." + block + "." + k: v for k, v in current[block].items()}
                )
                if context != "none":
                    aggregate = history_features(
                        current[block], [step(h, True)[block] for h in chosen]
                    )
                    features.update(
                        {"extra.history." + block + "." + k: v for k, v in aggregate.items()}
                    )
            views[name] = features
        output.append({**copy.deepcopy(original), "views": views})
    return output, audit


def coverage_report(examples):
    result = {}
    for task in sorted({r["benchmark"] for r in examples}):
        rows = [r for r in examples if r["benchmark"] == task]
        available = [r for r in rows if r["audit"]["proposal_features_available"]]
        fields = sorted(
            {k for r in rows for k in r["views"]["code_current"] if k.startswith("extra.")}
        )
        counts = {}
        for key in fields:
            values = [r["views"]["code_current"].get(key) for r in available]
            observed = [v for v in values if v is not None]
            counts[key] = {
                "observed": len(observed),
                "nonzero": sum(v != 0 for v in observed),
                "distinct_values": len(set(observed)),
            }
        result[task] = {
            "examples": len(rows),
            "usable_proposals": len(available),
            "sessions": len({r["cell_id"] for r in rows}),
            "fields": counts,
            "raw_feature_widths": {
                name: len({k for r in rows for k in r["views"][name]}) for name in SPECS
            },
        }
    return result


def paired_comparison(rows, labels, prediction, reference, repeats=2000):
    """Paired whole-session uncertainty for MAE and squared-error comparisons."""
    if not len(rows) or not len(rows) == len(labels) == len(prediction) == len(reference):
        raise ValueError("Mismatched paired-comparison rows")
    cells = defaultdict(list)
    for r, y, p, b in zip(rows, labels, prediction, reference):
        cells[r["cell_id"]].append((abs(y - p), abs(y - b), (y - p) ** 2, (y - b) ** 2, y, y * y))
    values = np.asarray([np.mean(cells[g], axis=0) for g in sorted(cells)])
    rng = np.random.default_rng(SEED)
    boot = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    mean = values.mean(axis=0)

    def stats(v):
        variance = np.maximum(v[..., 5] - v[..., 4] ** 2, 1e-12)
        return {
            "relative_mae_reduction": 1 - v[..., 0] / np.maximum(v[..., 1], 1e-12),
            "relative_rmse_reduction": 1 - np.sqrt(v[..., 2] / np.maximum(v[..., 3], 1e-12)),
            "r2_increase": (v[..., 3] - v[..., 2]) / variance,
        }

    point, sampled = stats(mean), stats(boot)
    return {
        **{
            k: {"estimate": float(v), "ci95": np.percentile(sampled[k], [2.5, 97.5]).tolist()}
            for k, v in point.items()
        },
        "sessions": len(cells),
        "replicates": repeats,
        "interpretation": "Exploratory paired session bootstrap; overlapping CV training sets, multiple arms and previous release exploration limit confirmation.",
    }


def prepare(source_bundle, inventory_path, old_results, output):
    source_bundle, old_results, output = map(Path, (source_bundle, old_results, output))
    verify_bundle(source_bundle)
    inventory = {r["example_id"]: r for r in load_jsonl(inventory_path)}
    old = read(source_bundle / "inputs.json")
    extraction_start = time.perf_counter()
    examples, audit = build_views(old, inventory)
    extraction_seconds = time.perf_counter() - extraction_start
    sources = {str(Path(p).resolve()): sha(p) for p in NEW_SOURCES}
    sources[str(Path(inventory_path).resolve())] = sha(inventory_path)
    sources[str((old_results / "predictions.json").resolve())] = sha(
        old_results / "predictions.json"
    )
    policy = {
        "schema": "code-feature-ablation-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(source_bundle.resolve()),
        "source_manifest_sha256": sha(source_bundle / "manifest.json"),
        "old_results": str(old_results.resolve()),
        "sources": sources,
        "specs": SPECS,
        "primary": PRIMARY,
        "primary_metric": "equal-session MAE per task",
        "secondary_metrics": ["equal-session RMSE", "equal-session R2", "pooled metrics"],
        "seed": SEED,
        "feature_extraction_seconds": extraction_seconds,
        "outer_folds": 8,
        "hgb_parameters": HGB_PARAMETERS,
        "target": "child accuracy minus GIVEN parent reference; inherited v1 base-reference assumptions",
        "selection": "No tuning or model selection. Nine fixed arms declared before new fits. Prior release exploration makes all results exploratory.",
        "code_boundary": "Existing reconstructed code only, before each first submission. Same masked-current and ancestor time gates as v1; no final snapshots, new recovery, child outcomes, generic n_examples, or ancestor scores.",
        "context_limit": "Code-list availability is assumed at use time, but historical source gaps remain; main-code transitive dependencies and executed-byte identity are not certified.",
        "llm": "Same cached held-out predictions; no new calls. Richer structured extraction does not imply the previous LLM had identical inputs. Never train from saved agent forecasts.",
        "hybrid": "Fixed 50/50 primary-WM + target-held-out-LLM diagnostic only; no learned calibration.",
    }
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    write(output / "policy.json", policy)
    write(output / "inputs.json", examples)
    write(output / "extraction_audit.json", audit)
    write(output / "coverage.json", coverage_report(examples))
    write(output / "labels.json", read(source_bundle / "labels.json"))
    write(output / "agent_predictions.json", read(source_bundle / "agent_predictions.json"))
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return {
        "examples": len(examples),
        "arms": list(SPECS),
        "primary": PRIMARY,
        "feature_extraction_seconds": extraction_seconds,
    }


def run(bundle, output):
    bundle, output = Path(bundle), Path(output)
    for name, expected in read(bundle / "manifest.json").items():
        if sha(bundle / name) != expected:
            raise ValueError("Modified feature bundle: " + name)
    policy = read(bundle / "policy.json")
    verify_bundle(policy["source_bundle"])
    if sha(Path(policy["source_bundle"]) / "manifest.json") != policy["source_manifest_sha256"]:
        raise ValueError("Changed source manifest")
    for path, expected in policy["sources"].items():
        if sha(path) != expected:
            raise ValueError("Changed extraction source: " + path)
    if policy["specs"] != {k: [v[0], list(v[1]), v[2]] for k, v in SPECS.items()}:
        raise ValueError("Changed ablation specifications")
    examples, labels = read(bundle / "inputs.json"), read(bundle / "labels.json")
    agent = read(bundle / "agent_predictions.json")
    old_predictions = read(Path(policy["old_results"]) / "predictions.json")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    write(
        output / "run_policy.json",
        {
            "bundle_manifest_sha256": sha(bundle / "manifest.json"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "primary": PRIMARY,
        },
    )
    write(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "joblib")},
        },
    )
    oof, training, model_hashes = {}, {}, {}
    for task in sorted({r["benchmark"] for r in examples}):
        rows = [r for r in examples if r["benchmark"] == task]
        y = np.asarray([labels[r["example_id"]] for r in rows])
        parents = np.asarray([r["parent_reference"] for r in rows])
        predictions = {name: np.full(len(rows), np.nan) for name in SPECS}
        training[task] = []
        for fold in range(8):
            tr = np.array([i for i, r in enumerate(rows) if r["fold"] != fold])
            te = np.array([i for i, r in enumerate(rows) if r["fold"] == fold])
            train_groups = sorted({rows[i]["cell_id"] for i in tr})
            test_groups = sorted({rows[i]["cell_id"] for i in te})
            if not len(tr) or not len(te) or set(train_groups) & set(test_groups):
                raise ValueError("Invalid session split")
            record = {
                "fold": fold,
                "train_sessions": train_groups,
                "test_sessions": test_groups,
                "models": {},
            }
            for name in SPECS:
                start = time.perf_counter()
                model = SmallPredictor("hgb_delta", seed=SEED).fit(
                    [rows[i]["views"][name] for i in tr],
                    y[tr],
                    parents[tr],
                    [rows[i]["cell_id"] for i in tr],
                )
                fitted = time.perf_counter()
                pred = model.predict([rows[i]["views"][name] for i in te], parents[te])
                predicted = time.perf_counter()
                predictions[name][te] = pred
                if name in BASELINE_NAMES:
                    expected = np.array(
                        [
                            old_predictions[task][rows[i]["example_id"]][BASELINE_NAMES[name]]
                            for i in te
                        ]
                    )
                    if not np.allclose(pred, expected, atol=1e-12, rtol=0):
                        raise ValueError(
                            "Existing baseline did not reproduce: " + task + " " + name
                        )
                path = output / f"{task}__fold{fold}__{name}.joblib"
                joblib.dump(model, path, compress=3)
                path.chmod(0o600)
                model_hashes[path.name] = sha(path)
                record["models"][name] = {
                    "fit_seconds": fitted - start,
                    "prediction_seconds": predicted - fitted,
                    "test_examples": len(te),
                    "serialized_bytes": path.stat().st_size,
                    "metadata": model.metadata(),
                }
            training[task].append(record)
            print(f"{task} fold {fold + 1}/8: nine fixed feature arms complete", flush=True)
        if not all(np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Incomplete OOF predictions")
        oof[task] = {
            r["example_id"]: {name: float(p[i]) for name, p in predictions.items()}
            for i, r in enumerate(rows)
        }
    write(output / "predictions.json", oof)
    write(output / "training.json", training)
    write(output / "model_hashes.json", model_hashes)
    write(
        output / "prediction_freeze.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "predictions_sha256": sha(output / "predictions.json"),
            "note": "Every fixed-arm prediction persisted before aggregate scoring; full labels decoded for grouped CV, not globally sealed.",
        },
    )
    report = {}
    for task, forecast in oof.items():
        rows = [r for r in examples if r["benchmark"] == task]
        y = np.asarray([labels[r["example_id"]] for r in rows])
        predictions = {
            name: np.array([forecast[r["example_id"]][name] for r in rows]) for name in SPECS
        }
        predictions["inference_only_saved"] = np.array([agent[r["example_id"]] for r in rows])
        predictions["fixed_half_blend"] = (
            predictions[PRIMARY] + predictions["inference_only_saved"]
        ) / 2
        report[task] = {}
        for name, p in predictions.items():
            baseline = (
                name
                if name in BASELINE_NAMES
                else (
                    "v1_parent"
                    if name in {"code_parent", "fixed_half_blend"}
                    else ("v1_history" if name == "code_history" else "v1_current")
                )
            )
            entry = {
                "all": evaluate(rows, y, p),
                "matched_baseline": baseline,
                "vs_inference_only": paired_comparison(
                    rows, y, p, predictions["inference_only_saved"]
                ),
                "vs_matched_v1": paired_comparison(rows, y, p, predictions[baseline]),
            }
            for subset in SUBSETS:
                idx = [
                    i
                    for i, r in enumerate(rows)
                    if r["audit"].get(
                        subset,
                        r["from_base"]
                        if subset == "base"
                        else (not r["from_base"] if subset == "continuation" else False),
                    )
                ]
                entry[subset] = evaluate([rows[i] for i in idx], y[idx], p[idx])
            report[task][name] = entry
    write(output / "report.json", report)
    return {
        task: {name: e["all"] for name, e in entries.items()} for task, entries in report.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--source-bundle", type=Path, default=Path("data/analysis/wm_small_predictors/v1_bundle")
    )
    prep.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/analysis/wm_rpm/data_v3/private/inventory.jsonl"),
    )
    prep.add_argument(
        "--old-results", type=Path, default=Path("data/analysis/wm_small_predictors/v1_results")
    )
    prep.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("run")
    fit.add_argument("--bundle", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        prepare(args.source_bundle, args.inventory, args.old_results, args.output)
        if args.command == "prepare"
        else run(args.bundle, args.output)
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
