"""Validated ten-run labels for checkpoint-prefix prediction examples.

The target is mean pass@1 across ten evaluation runs, NOT pass@10. Rates use
the [0, 1] scale. We never impute missing runs or treat repeats as new examples.
The pinned rescore writer stores per_problem values sorted by epoch but omits
epoch IDs; therefore a ragged per_problem matrix cannot safely be realigned.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXPECTED_RUNS = 10
EXPECTED_PROBLEMS = {"gsm8k": 1319, "aime2025": 30}
# Current project policy, doc/experiments/wm_exp_designs.md. These are retained
# even when a refreshed result has ten valid runs; this module does not release
# checkpoint/recipe-binding quarantines based on numeric validity alone.
QUARANTINED_EXP_IDS = frozenset({"r0-25-exp-02", "aime-r0-11-exp-02", "aime2-r0-12-exp-05"})


def _rate(value: Any) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def summarize_rates(
    rates: Sequence[float], per_run_n: Sequence[int] | None = None
) -> dict[str, Any]:
    """Describe repeats, distinguishing the unweighted mean from pooled rate.

    Pooled rate weights runs by their sample counts. The prediction target is
    always the unweighted mean; these coincide only if sample counts are equal
    (or by coincidence). ``nominal_se_across_runs`` assumes independent repeats
    and is not uncertainty from sampling benchmark questions.
    """
    if not rates or not all(_rate(value) for value in rates):
        raise ValueError("rates must be nonempty finite numbers in [0, 1]")
    if per_run_n is not None and (
        len(per_run_n) != len(rates) or not all(_positive_int(n) for n in per_run_n)
    ):
        raise ValueError("per_run_n must give one positive integer for each rate")
    values = [float(value) for value in rates]
    sample_sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "mean": statistics.mean(values),
        "population_sd_across_runs": statistics.pstdev(values),
        "sample_sd_across_runs": sample_sd,
        "nominal_se_across_runs": sample_sd / math.sqrt(len(values))
        if sample_sd is not None
        else None,
        "pooled_pass_rate": sum(value * n for value, n in zip(values, per_run_n)) / sum(per_run_n)
        if per_run_n is not None
        else None,
        "equal_run_sizes": len(set(per_run_n)) == 1 if per_run_n is not None else None,
    }


def load_label(path: str | Path, benchmark: str | None = None) -> dict[str, Any]:
    """Load the corpus rescore JSON, returning explicit missing/invalid status.

    Complete labels require ten runs, binary per-problem results for each run,
    matching per-run/aggregate statistics and the full known benchmark size.
    ``avg_pass_rate`` is null for invalid/incomplete labels. Available statistics
    remain available in ``avg_available_runs_pass_rate`` for audit only.
    Quarantines preserve valid numeric labels but disable training eligibility.
    No raw per-problem labels are copied into prediction inputs by this module.
    """
    path = Path(path)
    errors: list[str] = []
    checks: dict[str, bool | None] = {}
    result: dict[str, Any] = {
        "status": "missing",
        "source_path": str(path),
        "source_sha256": None,
        "exp_id": path.stem,
        "benchmark": benchmark,
        "metric": "mean_pass_at_1_over_10_runs",
        "rate_scale": "fraction_0_to_1",
        "expected_n_runs": EXPECTED_RUNS,
        "n_runs": 0,
        "per_run_pass_rate": [],
        "avg_pass_rate": None,
        "avg_available_runs_pass_rate": None,
        "per_run_n": None,
        "n_problems": None,
        "pooled_pass_rate": None,
        "population_sd_across_runs": None,
        "sample_sd_across_runs": None,
        "nominal_se_across_runs": None,
        "equal_run_sizes": None,
        "stored_accuracy": None,
        "stored_std_across_epochs": None,
        "consistency_checks": checks,
        "errors": errors,
        "quarantined": path.stem in QUARANTINED_EXP_IDS,
        "eligible_for_training": False,
    }
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        errors.append("result_file_missing")
        return result
    except OSError as exc:
        result["status"] = "invalid"
        errors.append(f"result_file_unreadable:{type(exc).__name__}")
        return result
    result["status"] = "invalid"
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        errors.append("result_file_invalid_json")
        return result
    if not isinstance(data, dict):
        errors.append("result_not_object")
        return result

    metadata = data.get("rescore10", {})
    if not isinstance(metadata, dict):
        errors.append("rescore10_metadata_not_object")
        metadata = {}
    recorded_benchmark = metadata.get("benchmark")
    if benchmark is not None and recorded_benchmark is not None and benchmark != recorded_benchmark:
        errors.append("benchmark_mismatch")
    resolved_benchmark = benchmark if benchmark is not None else recorded_benchmark
    result["benchmark"] = resolved_benchmark
    if resolved_benchmark not in EXPECTED_PROBLEMS:
        errors.append("benchmark_missing_or_unsupported")
    if metadata.get("id", path.stem) != path.stem:
        errors.append("result_exp_id_mismatch")
    # Runtime provenance belongs with the label/evaluation audit, not training
    # recipe code. These values are included unchanged for compatibility checks.
    result["evaluation_metadata"] = metadata
    result["scorer_patch"] = data.get("scorer_patch")
    result["n_sample_errors"] = data.get("n_sample_errors")

    rates = data.get("per_epoch_accuracy")
    if not isinstance(rates, list) or not rates or not all(_rate(value) for value in rates):
        errors.append("per_epoch_accuracy_missing_or_invalid")
        return result
    result["per_run_pass_rate"] = [float(value) for value in rates]
    result["n_runs"] = len(rates)
    stats = summarize_rates(rates)
    result["avg_available_runs_pass_rate"] = stats.pop("mean")
    result.update(stats)
    checks["exactly_ten_runs"] = len(rates) == EXPECTED_RUNS
    if not checks["exactly_ten_runs"]:
        errors.append("requires_exactly_ten_runs")
    checks["stored_epochs_matches"] = _positive_int(data.get("epochs")) and data["epochs"] == len(
        rates
    )
    if not checks["stored_epochs_matches"]:
        errors.append("stored_epochs_mismatch")

    stored_accuracy = data.get("accuracy")
    result["stored_accuracy"] = stored_accuracy if _rate(stored_accuracy) else None
    checks["stored_accuracy_matches"] = _rate(stored_accuracy) and math.isclose(
        stored_accuracy, result["avg_available_runs_pass_rate"], rel_tol=0, abs_tol=1e-12
    )
    if not checks["stored_accuracy_matches"]:
        errors.append("stored_accuracy_missing_or_mismatch")
    stored_sd = data.get("std_across_epochs")
    result["stored_std_across_epochs"] = stored_sd if _rate(stored_sd) else None
    checks["stored_population_sd_matches"] = _rate(stored_sd) and math.isclose(
        stored_sd, result["population_sd_across_runs"], rel_tol=0, abs_tol=1e-12
    )
    if not checks["stored_population_sd_matches"]:
        errors.append("stored_population_sd_missing_or_mismatch")

    n_problems = data.get("n_problems")
    result["n_problems"] = n_problems if _positive_int(n_problems) else None
    checks["expected_benchmark_size"] = (
        _positive_int(n_problems)
        and resolved_benchmark in EXPECTED_PROBLEMS
        and n_problems == EXPECTED_PROBLEMS[resolved_benchmark]
    )
    if not checks["expected_benchmark_size"]:
        errors.append("benchmark_problem_count_mismatch")

    problems = data.get("per_problem")
    if not isinstance(problems, dict) or not problems:
        errors.append("per_problem_missing_or_invalid")
        return result
    checks["stored_n_problems_matches"] = len(problems) == n_problems
    if not checks["stored_n_problems_matches"]:
        errors.append("stored_n_problems_mismatch")
    rectangular = all(isinstance(v, list) and len(v) == len(rates) for v in problems.values())
    checks["per_problem_matrix_complete"] = rectangular
    if not rectangular:
        errors.append("ragged_per_problem_matrix_epoch_alignment_unknown")
        return result
    binary = all(_rate(v) and v in (0, 1) for values in problems.values() for v in values)
    checks["per_problem_scores_binary"] = binary
    if not binary:
        errors.append("per_problem_scores_not_binary")
        return result
    reconstructed_rates = [
        statistics.mean(values[i] for values in problems.values()) for i in range(len(rates))
    ]
    checks["per_run_rates_match_problem_scores"] = all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(rates, reconstructed_rates)
    )
    if not checks["per_run_rates_match_problem_scores"]:
        errors.append("per_epoch_accuracy_mismatch_problem_scores")
    result["per_run_n"] = [len(problems)] * len(rates)
    pooled_stats = summarize_rates(rates, result["per_run_n"])
    result["pooled_pass_rate"] = pooled_stats["pooled_pass_rate"]
    result["equal_run_sizes"] = pooled_stats["equal_run_sizes"]
    if not errors:
        result["status"] = "complete"
        result["avg_pass_rate"] = result["avg_available_runs_pass_rate"]
        result["eligible_for_training"] = not result["quarantined"]
    return result
