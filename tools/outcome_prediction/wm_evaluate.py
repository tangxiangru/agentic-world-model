"""Score matched inference-only RPM decisions with versus without WM access.

This is an offline scorer, not an agent runner or a recipe-fidelity certificate.
Freeze its protocol and case inputs BEFORE running either arm. Keep labels out of
the agent workspace. A valid transcript must be normalized by the trusted runner;
this module checks that normalization against the frozen experiment contract.

The primary outcome is the official task-model accuracy of the selected recipe,
not preference-prediction accuracy. Tasks are reported separately. Multiple choices
from one scientist run are averaged before calculating task means and bootstrap
intervals. Failed or missing decisions remain in the denominator of coverage;
complete-pair estimates are explicitly conditional when any case is incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ARMS = ("rpm", "rpm_wm")
RETRIEVAL_TOOLS = ("list_evidence", "read_evidence", "search_evidence")
# Some CLI versions retain this non-data-access lifecycle tool with MCP enabled.
LIFECYCLE_TOOLS = {"EndConversation"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _number(value, *, low=0.0, high=1.0):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _hash(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def allowed_tools(arm):
    if arm not in ARMS:
        raise ValueError("Unknown evaluation arm")
    names = RETRIEVAL_TOOLS + (("predict_candidate",) if arm == "rpm_wm" else ())
    return {f"mcp__evidence__{name}" for name in names}


def validate_protocol(protocol):
    """Validate a label-free matched-arm protocol; no target values are needed."""
    if not isinstance(protocol, dict) or set(protocol) != {
        "schema_version",
        "target",
        "label_source",
        "failure_policy",
        "aggregation",
        "wm_model_sha256",
        "agent",
        "cases",
    }:
        raise ValueError("Unexpected protocol fields; keep target labels separate")
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation protocol")
    if protocol.get("target") != "immediate_official_accuracy":
        raise ValueError("This scorer does not score eventual-subtree outcomes")
    if protocol.get("label_source") not in ("archived_checkpoint", "fresh_execution"):
        raise ValueError("Protocol must explicitly distinguish archived versus fresh labels")
    if protocol.get("failure_policy") != "report_missing_no_retry_no_imputation":
        raise ValueError("Unsupported failure policy")
    if protocol.get("aggregation") != "equal_scientist_run_within_benchmark":
        raise ValueError("Unsupported aggregation policy")
    if not _hash(protocol.get("wm_model_sha256")):
        raise ValueError("Pin the learned WM artifact")
    common = protocol.get("agent", {})
    required = {
        "model",
        "effort",
        "max_turns",
        "max_budget_usd",
        "timeout_seconds",
        "common_system_sha256",
        "wm_instruction_sha256",
        "cli_version",
    }
    if not isinstance(common, dict) or set(common) != required:
        raise ValueError("Incomplete common agent configuration")
    for key in ("model", "effort", "cli_version"):
        if not isinstance(common[key], str) or not common[key]:
            raise ValueError("Invalid common agent identifier")
    for key in ("common_system_sha256", "wm_instruction_sha256"):
        if not _hash(common[key]):
            raise ValueError("Pin both shared and treatment-specific instructions")
    if type(common["max_turns"]) is not int or common["max_turns"] < 1:
        raise ValueError("Invalid turn budget")
    for key in ("max_budget_usd", "timeout_seconds"):
        if not _number(common[key], low=0.000001, high=1e9):
            raise ValueError("Invalid agent budget")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Protocol needs frozen decision cases")
    seen = set()
    cell_benchmarks = {}
    for case in cases:
        required_case = {
            "case_id",
            "cell_id",
            "benchmark",
            "candidate_ids",
            "evidence_sha256",
            "candidate_payloads_sha256",
            "packet_audit_sha256",
        }
        if not isinstance(case, dict) or set(case) != required_case:
            raise ValueError("Unexpected case fields; keep target labels separate")
        for key in ("case_id", "cell_id", "benchmark"):
            if not isinstance(case[key], str) or not case[key]:
                raise ValueError("Invalid case identifier")
        if case["case_id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["case_id"])
        previous = cell_benchmarks.setdefault(case["cell_id"], case["benchmark"])
        if previous != case["benchmark"]:
            raise ValueError("One scientist run cannot span benchmark strata")
        candidates = case["candidate_ids"]
        if (
            not isinstance(candidates, list)
            or len(candidates) < 2
            or any(not isinstance(c, str) or not c for c in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("Case needs distinct ordered candidate IDs")
        for key in ("evidence_sha256", "candidate_payloads_sha256", "packet_audit_sha256"):
            if not _hash(case[key]):
                raise ValueError("Pin evidence, candidate inputs and their exposure audit")
    return {case["case_id"]: case for case in cases}


def validate_run(run, protocol, case, arm):
    """Return a failure reason, never repair a malformed or partial decision."""
    if run is None:
        return "missing"
    if run.get("case_id") != case["case_id"] or run.get("arm") != arm:
        return "identity_mismatch"
    expected = {
        "protocol_sha256": digest(protocol),
        "case_sha256": digest(case),
        "agent_sha256": digest(protocol["agent"]),
        "wm_model_sha256": protocol["wm_model_sha256"] if arm == "rpm_wm" else None,
    }
    if run.get("provenance") != expected:
        return "provenance_mismatch"
    if not _hash(run.get("transcript_sha256")):
        return "missing_transcript_fingerprint"
    if run.get("status") != "success":
        return "runner_failure"
    if (
        type(run.get("returncode")) is not int
        or run["returncode"] != 0
        or run.get("result_subtype") != "success"
    ):
        return "non_success_terminal"
    if run.get("is_error") is not False:
        return "terminal_error"
    if run.get("model") != protocol["agent"]["model"]:
        return "model_mismatch"
    turns = run.get("num_turns")
    if type(turns) is not int or not 1 <= turns <= protocol["agent"]["max_turns"]:
        return "turn_budget_violation"
    available = run.get("tools_available")
    if (
        not isinstance(available, list)
        or any(not isinstance(t, str) for t in available)
        or len(available) != len(set(available))
        or set(available) - LIFECYCLE_TOOLS != allowed_tools(arm)
    ):
        return "tool_inventory_mismatch"
    calls = run.get("tool_calls")
    if not isinstance(calls, list):
        return "missing_tool_trace"
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in set(available):
            return "unexpected_tool_call"
        args = call.get("arguments")
        if not isinstance(args, dict):
            return "invalid_tool_arguments"
        if call["name"].endswith("__predict_candidate") and (
            set(args) != {"candidate_id"} or args["candidate_id"] not in case["candidate_ids"]
        ):
            return "out_of_scope_wm_query"
    decision = run.get("decision")
    if not isinstance(decision, dict) or set(decision) != {
        "choice",
        "ranking",
        "confidence",
        "rationale",
    }:
        return "invalid_decision_schema"
    ranking = decision["ranking"]
    if (
        not isinstance(ranking, list)
        or any(not isinstance(c, str) for c in ranking)
        or len(ranking) != len(case["candidate_ids"])
        or set(ranking) != set(case["candidate_ids"])
        or decision["choice"] != ranking[0]
    ):
        return "invalid_ranking"
    if not _number(decision["confidence"]) or not isinstance(decision["rationale"], str):
        return "invalid_decision_fields"
    cost = run.get("cost_usd")
    if cost is not None and not _number(cost, high=1e9):
        return "invalid_cost"
    return None


def _macro(rows, key):
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["cell_id"]].append(row[key])
    return mean(mean(v) for v in by_run.values()) if by_run else None


def _interval(run_values, *, seed, replicates):
    if len(run_values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(run_values, k=len(run_values))) for _ in range(replicates))
    return [samples[int(0.025 * (replicates - 1))], samples[int(0.975 * (replicates - 1))]]


def score(protocol, labels, runs, *, bootstrap_seed=20260905, bootstrap_replicates=5000):
    """Score every planned case; never select the cohort using score gaps."""
    cases = validate_protocol(protocol)
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if not isinstance(labels, dict) or set(labels) != set(cases):
        raise ValueError("Labels must cover exactly the frozen case IDs")
    for case_id, case in cases.items():
        y = labels[case_id]
        if not isinstance(y, dict) or set(y) != set(case["candidate_ids"]):
            raise ValueError("Labels must cover exactly the frozen candidate IDs")
        if any(not _number(value) for value in y.values()):
            raise ValueError("Official accuracy must be finite and between zero and one")
    indexed = {}
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("Malformed run record")
        key = (run.get("case_id"), run.get("arm"))
        if key[0] not in cases or key[1] not in ARMS:
            raise ValueError("Run is outside the frozen comparison")
        if key in indexed:
            raise ValueError("Duplicate attempt: this protocol forbids selective retries")
        indexed[key] = run
    outcomes = []
    for case_id, case in cases.items():
        row = {key: case[key] for key in ("case_id", "cell_id", "benchmark")}
        y = labels[case_id]
        row["oracle_accuracy"] = max(y.values())
        row["minimum_candidate_accuracy"] = min(y.values())
        row["arms"] = {}
        for arm in ARMS:
            run = indexed.get((case_id, arm))
            reason = validate_run(run, protocol, case, arm)
            record = {"valid": reason is None, "failure_reason": reason}
            if reason is None:
                choice = run["decision"]["choice"]
                record.update(
                    choice=choice,
                    selected_model_accuracy=y[choice],
                    regret=max(y.values()) - y[choice],
                    selected_an_oracle_tie=math.isclose(
                        y[choice], max(y.values()), rel_tol=0, abs_tol=1e-12
                    ),
                    wm_query_count=sum(
                        c["name"].endswith("__predict_candidate") for c in run["tool_calls"]
                    ),
                )
            row["arms"][arm] = record
        row["paired_valid"] = all(row["arms"][arm]["valid"] for arm in ARMS)
        if row["paired_valid"]:
            a, b = (row["arms"][arm]["selected_model_accuracy"] for arm in ARMS)
            row["wm_minus_rpm_accuracy"] = b - a
            row["winner"] = "tie" if abs(b - a) <= 1e-12 else ("rpm_wm" if b > a else "rpm")
        outcomes.append(row)
    results = {}
    for benchmark in sorted({case["benchmark"] for case in cases.values()}):
        part = [r for r in outcomes if r["benchmark"] == benchmark]
        paired = [r for r in part if r["paired_valid"]]
        summary = {
            "planned_cases": len(part),
            "planned_scientist_runs": len({r["cell_id"] for r in part}),
            "paired_valid_cases": len(paired),
            "paired_valid_scientist_runs": len({r["cell_id"] for r in paired}),
            "complete": len(paired) == len(part),
            "estimate_scope": "all planned cases"
            if len(paired) == len(part)
            else ("complete pairs only; failures may cause selection bias"),
            "arms": {},
            "paired_case_wins": dict(Counter(r["winner"] for r in paired)),
        }
        for arm in ARMS:
            paired_metrics = [dict(cell_id=r["cell_id"], **r["arms"][arm]) for r in paired]
            summary["arms"][arm] = {
                "valid_cases": sum(r["arms"][arm]["valid"] for r in part),
                "failure_reasons": dict(
                    Counter(
                        r["arms"][arm]["failure_reason"]
                        for r in part
                        if not r["arms"][arm]["valid"]
                    )
                ),
                "paired_run_macro_selected_model_accuracy": _macro(
                    paired_metrics, "selected_model_accuracy"
                ),
                "paired_run_macro_regret": _macro(paired_metrics, "regret"),
                "paired_run_macro_oracle_selection_rate": _macro(
                    paired_metrics, "selected_an_oracle_tie"
                ),
            }
        by_run = defaultdict(list)
        for row in paired:
            by_run[row["cell_id"]].append(row["wm_minus_rpm_accuracy"])
        values = [mean(by_run[cell]) for cell in sorted(by_run)]
        summary["paired_run_macro_wm_minus_rpm_accuracy"] = mean(values) if values else None
        summary["paired_run_bootstrap_95_interval"] = _interval(
            values, seed=bootstrap_seed, replicates=bootstrap_replicates
        )
        summary["valid_wm_decisions_using_wm"] = sum(
            r["arms"]["rpm_wm"].get("wm_query_count", 0) > 0 for r in part
        )
        # These are bounds on possible *selections*, not imputed performances of
        # failed agents. An invalid arm is allowed any of the frozen candidates.
        bound_rows = []
        for row in part:
            bounds = {}
            for arm in ARMS:
                result = row["arms"][arm]
                if result["valid"]:
                    bounds[arm] = (result["selected_model_accuracy"],) * 2
                else:
                    bounds[arm] = (row["minimum_candidate_accuracy"], row["oracle_accuracy"])
            bound_rows.append(
                {
                    "cell_id": row["cell_id"],
                    "low": bounds["rpm_wm"][0] - bounds["rpm"][1],
                    "high": bounds["rpm_wm"][1] - bounds["rpm"][0],
                }
            )
        summary["all_case_possible_selection_delta_bounds"] = [
            _macro(bound_rows, "low"),
            _macro(bound_rows, "high"),
        ]
        results[benchmark] = summary
    known_costs = [r["cost_usd"] for r in runs if _number(r.get("cost_usd"), high=1e9)]
    return {
        "schema_version": 1,
        "protocol_sha256": digest(protocol),
        "label_source": protocol["label_source"],
        "complete": all(r["paired_valid"] for r in outcomes),
        "status": "scored_all_planned_pairs"
        if all(r["paired_valid"] for r in outcomes)
        else "incomplete_comparison",
        "primary_metric": "official task-model accuracy of first-ranked recipe, macro-averaged by scientist run within task",
        "interval_scope": "paired scientist-run sampling only; not task-evaluation or LLM sampling uncertainty",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "cost": {
            "known_total_usd": math.fsum(known_costs),
            "recorded_attempts": len(runs),
            "attempts_with_unknown_cost": len(runs) - len(known_costs),
            "missing_attempts": 2 * len(cases) - len(runs),
        },
        "benchmarks": results,
        "cases": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="JSONL normalized run records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(
        json.loads(args.protocol.read_text()),
        json.loads(args.labels.read_text()),
        [json.loads(line) for line in args.runs.read_text().splitlines() if line.strip()],
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
