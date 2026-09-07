"""Score the separately frozen format-recovery sensitivity; no model tuning."""

from pathlib import Path

import numpy as np

from tools.outcome_prediction import wm_agent_local_benchmark as experiment
from tools.outcome_prediction.wm_agent_local_fit import fit_predict
from tools.outcome_prediction.wm_clean_refresh import read, sha, write
from tools.outcome_prediction.wm_local_agent_io import _decode, _events


def score(bundle=experiment.BUNDLE, original=experiment.OUTPUT,
          recovery=Path("data/analysis/wm_agent_local/v1_recovery")):
    bundle, original, recovery = map(Path, (bundle, original, recovery))
    if (recovery / "report.json").exists():
        raise FileExistsError("Preserve the existing sensitivity report")
    # Both forecast freezes are required before any held-out label inspection.
    for directory in (original, recovery):
        freeze = read(directory / "prediction_freeze.json")
        experiment.previous.verify_files(directory, freeze["files"])
    recovery_policy_freeze = read(recovery / "recovery_policy_freeze.json")
    experiment.previous.verify_files(recovery, recovery_policy_freeze["files"])
    recovery_policy = read(recovery / "recovery_policy.json")
    if str(Path(__file__).resolve()) not in recovery_policy["sources"]:
        raise ValueError("Recovery scoring source was not frozen before inference")
    for path, expected_hash in recovery_policy["sources"].items():
        if sha(path) != expected_hash:
            raise ValueError("Changed frozen recovery source: " + path)
    run_policy = read(recovery / "run_policy.json")
    if run_policy["original_prediction_freeze_sha256"] != sha(original / "prediction_freeze.json") or run_policy["recovery_policy_freeze_sha256"] != sha(recovery / "recovery_policy_freeze.json"):
        raise ValueError("Changed linked recovery/original forecast freezes")
    policy, train, train_labels, queries, banks, _ = experiment.validate_prepared(bundle)
    old = {r["example_id"]: r for r in read(original / "predictions.json")}
    values = read(recovery / "predictions.json")
    merged = {r["example_id"]: r for r in values}
    if len(merged) != len(values) or set(merged) != set(old) or set(old) != {q["example_id"] for q in queries}:
        raise ValueError("Missing or duplicate recovery identities")
    required = {"selected_aliases", "weights", "model", "feature_keys", "rationale"}
    train_by_id = {r["example_id"]: r for r in train}
    recovered = []
    for query in queries:
        key = query["example_id"]
        before, after = old[key], merged[key]
        if before["selection"] is not None:
            if before != after:
                raise ValueError("Original valid selection/result changed during recovery")
            continue
        expected = None
        for attempt in range(1, before.get("selector_attempts", 0) + 1):
            call = read(original / "calls" / (experiment.key_for(query) + f"__selector__attempt{attempt}.json"))
            if call.get("example_id") != key or call.get("stage") != "selector" or call.get("attempt") != attempt or call.get("prompt_sha256") != before["selector_prompt_sha256"]:
                raise ValueError("Original rejected selector call/query/prompt mismatch")
            if not call.get("stdout"):
                continue
            events = _events(call["stdout"])
            response, error, _, _ = _decode(events, policy["model_requested"], call["returncode"], call["stderr"])
            if error or events != call["raw_events"] or response != call["response"] or not required.issubset(response):
                continue
            try:
                expected = experiment.resolve_selection(
                    {k: response[k] for k in required}, banks[query["benchmark"]],
                    [r for r in train if r["benchmark"] == query["benchmark"]], query)
                break
            except (ValueError, TypeError, KeyError):
                continue
        if expected != after["selection"]:
            raise ValueError("Recovery was not the first semantically valid original selection")
        if after["local_fit"] is None:
            continue
        selected = [train_by_id[k] for k in expected["selected_ids"]]
        labels = {r["example_id"]: train_labels[r["example_id"]] for r in selected}
        local = fit_predict(selected, labels, query, expected)
        median = fit_predict(selected, labels, query, {**expected, "model": "weighted_median", "feature_keys": []})
        if local != after["local_fit"] or median != after["median_fit"]:
            raise ValueError("Recovered fits fail deterministic training-only replay")
        prompt = experiment.selected_prompt(query, selected, labels, local["fit_metadata"]["normalized_sample_weights"], policy["feature_order"])
        if after["llm_prediction"] is not None:
            calls = [read(p) for p in sorted((recovery / "calls").glob(experiment.key_for(query) + "__predictor__attempt*.json"))]
            valid = [c for c in calls if not c.get("error")]
            if len(valid) != 1 or valid[0] is not calls[-1]:
                raise ValueError("Recovery did not keep the first valid direct forecast")
            for i, call in enumerate(calls, 1):
                if call["example_id"] != key or call["stage"] != "predictor" or call["attempt"] != i or call["prompt_sha256"] != experiment.digest(prompt):
                    raise ValueError("Recovered direct-LLM call/query/prompt mismatch")
            experiment.validate_success_call(valid[0], policy)
            if after["llm_prediction"] != experiment.previous.parse_prediction(experiment.canonical(valid[0]["response"])):
                raise ValueError("Recovered direct forecast changed")
        recovered.append(key)

    # Target labels are used for metrics only after every forecast was frozen
    # and additive recovery choices/fits were independently reconstructed.
    labels = read(Path(policy["training"]) / "labels.json")
    fixed = {(r["example_id"], r["arm"]): r for r in read(Path(policy["previous_results"]) / "predictions.json")}
    regressors = read(Path(policy["regressors"]) / "predictions.json")
    report = {"scored_at": experiment.now(), "recovered_ids": recovered, "benchmarks": {},
              "interpretation": "Blinded additive format-recovery sensitivity. Original valid choices unchanged; failed selections recovered from first semantically valid existing raw reply, never target-error feedback. Original strict run reported separately. Previously explored developmental holdout."}
    for benchmark in experiment.previous.TASKS:
        intended = [q for q in queries if q["benchmark"] == benchmark]
        common = [q for q in intended if merged[q["example_id"]]["local_fit"] is not None and merged[q["example_id"]]["llm_prediction"] is not None]
        ids = [q["example_id"] for q in common]
        y = np.asarray([labels[k]["accuracy"] for k in ids])
        reference = np.asarray([q["parent_reference"] for q in common])
        forecasts = {
            "agent_local_model": np.asarray([merged[k]["local_fit"]["predicted_accuracy"] for k in ids]),
            "selected_weighted_median": np.asarray([merged[k]["median_fit"]["predicted_accuracy"] for k in ids]),
            "selected_few_shot": np.asarray([merged[k]["llm_prediction"]["predicted_accuracy"] for k in ids]),
            "fixed_few_shot": np.asarray([fixed[k, "few_shot"]["predicted_accuracy"] for k in ids]),
            "fixed_zero_shot": np.asarray([fixed[k, "zero_shot"]["predicted_accuracy"] for k in ids]),
            **{arm: np.asarray([regressors["combined__" + benchmark][k][arm] for k in ids])
               for arm in ("train_selected", "hgb_current", "hgb_history")},
        }
        metrics = {}
        for arm, pred in forecasts.items():
            metrics[arm] = {}
            for subset in ("all", "published_base", "measured_parent"):
                mask = np.asarray([subset == "all" or q["reference_kind"] == subset for q in common], dtype=bool)
                subrows = [q for q, keep in zip(common, mask) if keep]
                metrics[arm][subset] = {
                    "accuracy": experiment.evaluate(subrows, y[mask], pred[mask]),
                    "delta": experiment.evaluate(subrows, (y - reference)[mask], (pred - reference)[mask]),
                }
        comparisons = {arm: experiment.compare_fixed_holdout(common, y, forecasts["agent_local_model"], forecasts[arm])
                       for arm in ("fixed_few_shot", "selected_few_shot", "selected_weighted_median")} if common else {}
        report["benchmarks"][benchmark] = {
            "intended": len(intended), "common_success": len(common), "common_ids": ids,
            "metrics": metrics, "primary_comparisons": comparisons,
            "coverage": {"original_success": sum(old[q["example_id"]]["llm_prediction"] is not None for q in intended),
                         "recovered_success": sum(q["example_id"] in recovered and merged[q["example_id"]]["llm_prediction"] is not None for q in intended)},
        }
    write(recovery / "report.json", report)
    write(recovery / "scoring_manifest.json", {"score_source_sha256": sha(__file__), "report_sha256": sha(recovery / "report.json"),
          "recovery_prediction_freeze_sha256": sha(recovery / "prediction_freeze.json"), "original_prediction_freeze_sha256": sha(original / "prediction_freeze.json")})
    return report


if __name__ == "__main__":
    import json

    result = score()
    print(json.dumps({b: {"n": value["common_success"], "mae_pp": {arm: round(100 * m["all"]["accuracy"]["mae"], 3) for arm, m in value["metrics"].items()}}
                      for b, value in result["benchmarks"].items()}, indent=2))
