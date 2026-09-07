import copy
import json

import pytest

from tools.outcome_prediction import wm_evaluate as ev


def protocol():
    return {
        "schema_version": 1,
        "target": "immediate_official_accuracy",
        "label_source": "archived_checkpoint",
        "failure_policy": "report_missing_no_retry_no_imputation",
        "aggregation": "equal_scientist_run_within_benchmark",
        "wm_model_sha256": "a" * 64,
        "agent": {
            "model": "fixed-model",
            "effort": "high",
            "max_turns": 12,
            "max_budget_usd": 1.0,
            "timeout_seconds": 300,
            "common_system_sha256": "b" * 64,
            "wm_instruction_sha256": "c" * 64,
            "cli_version": "pinned-cli",
        },
        "cases": [
            {
                "case_id": case_id,
                "cell_id": cell,
                "benchmark": benchmark,
                "candidate_ids": ["a", "b"],
                "evidence_sha256": "d" * 64,
                "candidate_payloads_sha256": "e" * 64,
                "packet_audit_sha256": "f" * 64,
            }
            for case_id, cell, benchmark in (
                ("one", "run1", "gsm8k"),
                ("two", "run1", "gsm8k"),
                ("three", "run2", "gsm8k"),
                ("four", "run3", "aime2025"),
            )
        ],
    }


def run(p, case, arm, choice="a", wm_query=True):
    return {
        "case_id": case["case_id"],
        "arm": arm,
        "status": "success",
        "provenance": {
            "protocol_sha256": ev.digest(p),
            "case_sha256": ev.digest(case),
            "agent_sha256": ev.digest(p["agent"]),
            "wm_model_sha256": p["wm_model_sha256"] if arm == "rpm_wm" else None,
        },
        "transcript_sha256": "0" * 64,
        "returncode": 0,
        "result_subtype": "success",
        "is_error": False,
        "num_turns": 3,
        "model": p["agent"]["model"],
        "tools_available": sorted(ev.allowed_tools(arm)),
        "tool_calls": [
            {"name": "mcp__evidence__predict_candidate", "arguments": {"candidate_id": "a"}}
        ]
        if arm == "rpm_wm" and wm_query
        else [],
        "decision": {
            "choice": choice,
            "ranking": [choice, "b" if choice == "a" else "a"],
            "confidence": 0.5,
            "rationale": "fixture",
        },
        "cost_usd": 0.2,
    }


def fixture():
    p = protocol()
    labels = {
        "one": {"a": 0.2, "b": 0.4},
        "two": {"a": 0.2, "b": 0.6},
        "three": {"a": 0.0, "b": 0.8},
        "four": {"a": 0.1, "b": 0.1},
    }
    runs = [run(p, c, arm, "a" if arm == "rpm" else "b") for c in p["cases"] for arm in ev.ARMS]
    return p, labels, runs


def test_separate_task_accuracy_run_macro_and_ties():
    p, labels, runs = fixture()
    result = ev.score(p, labels, runs, bootstrap_replicates=100)
    gsm = result["benchmarks"]["gsm8k"]
    assert result["complete"]
    assert gsm["arms"]["rpm"]["paired_run_macro_selected_model_accuracy"] == pytest.approx(0.1)
    assert gsm["arms"]["rpm_wm"]["paired_run_macro_selected_model_accuracy"] == pytest.approx(0.65)
    assert gsm["paired_run_macro_wm_minus_rpm_accuracy"] == pytest.approx(0.55)
    assert gsm["paired_case_wins"] == {"rpm_wm": 3}
    assert gsm["paired_run_bootstrap_95_interval"] == pytest.approx([0.3, 0.8])
    assert gsm["all_case_possible_selection_delta_bounds"] == pytest.approx([0.55, 0.55])
    aime = result["benchmarks"]["aime2025"]
    assert aime["paired_case_wins"] == {"tie": 1}
    assert aime["paired_run_bootstrap_95_interval"] is None
    assert aime["arms"]["rpm"]["paired_run_macro_oracle_selection_rate"] == 1
    assert "accuracy" not in result  # no misleading pooled GSM/AIME accuracy


def test_missing_runs_preserved_with_bounds_not_imputation():
    p, labels, runs = fixture()
    runs = [r for r in runs if not (r["case_id"] == "three" and r["arm"] == "rpm_wm")]
    result = ev.score(p, labels, runs, bootstrap_replicates=100)
    assert not result["complete"]
    assert result["status"] == "incomplete_comparison"
    assert result["cost"]["missing_attempts"] == 1
    gsm = result["benchmarks"]["gsm8k"]
    assert gsm["planned_cases"] == 3 and gsm["paired_valid_cases"] == 2
    assert gsm["paired_run_macro_wm_minus_rpm_accuracy"] == pytest.approx(0.3)
    assert gsm["paired_run_bootstrap_95_interval"] is None
    assert gsm["all_case_possible_selection_delta_bounds"] == pytest.approx([0.15, 0.55])
    assert gsm["arms"]["rpm_wm"]["failure_reasons"] == {"missing": 1}
    assert "selection bias" in gsm["estimate_scope"]


def test_no_runs_is_incomplete_not_zero_performance():
    p, labels, _ = fixture()
    result = ev.score(p, labels, [], bootstrap_replicates=100)
    assert not result["complete"]
    assert result["cost"]["missing_attempts"] == 8
    assert result["benchmarks"]["gsm8k"]["paired_run_macro_wm_minus_rpm_accuracy"] is None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("returncode", 1, "non_success_terminal"),
        ("result_subtype", "error_max_turns", "non_success_terminal"),
        ("is_error", True, "terminal_error"),
        ("model", "other-model", "model_mismatch"),
        ("num_turns", 13, "turn_budget_violation"),
        ("num_turns", True, "turn_budget_violation"),
        ("tools_available", ["Bash"], "tool_inventory_mismatch"),
        ("tool_calls", [{"name": "Bash", "arguments": {}}], "unexpected_tool_call"),
        ("transcript_sha256", None, "missing_transcript_fingerprint"),
        ("cost_usd", float("nan"), "invalid_cost"),
        ("cost_usd", -1, "invalid_cost"),
    ],
)
def test_invalid_run_never_enters_performance(field, value, reason):
    p = protocol()
    c = p["cases"][0]
    r = run(p, c, "rpm")
    r[field] = value
    assert ev.validate_run(r, p, c, "rpm") == reason


def test_wm_only_predicts_whitelisted_candidates():
    p = protocol()
    c = p["cases"][0]
    r = run(p, c, "rpm_wm")
    r["tool_calls"][0]["arguments"]["candidate_id"] = "unapproved"
    assert ev.validate_run(r, p, c, "rpm_wm") == "out_of_scope_wm_query"
    r["tool_calls"][0]["arguments"] = {"candidate_id": "a", "payload": {}}
    assert ev.validate_run(r, p, c, "rpm_wm") == "out_of_scope_wm_query"


def test_only_optional_lifecycle_tool_is_permitted():
    p = protocol()
    c = p["cases"][0]
    r = run(p, c, "rpm")
    r["tools_available"].append("EndConversation")
    r["tool_calls"].append({"name": "EndConversation", "arguments": {}})
    assert ev.validate_run(r, p, c, "rpm") is None
    r["tools_available"].append("StructuredOutput")
    assert ev.validate_run(r, p, c, "rpm") == "tool_inventory_mismatch"


def test_treatment_without_wm_usage_remains_in_intent_to_treat_comparison():
    p, labels, runs = fixture()
    for r in runs:
        r["tool_calls"] = []
    result = ev.score(p, labels, runs, bootstrap_replicates=100)
    assert result["complete"]
    assert result["benchmarks"]["gsm8k"]["valid_wm_decisions_using_wm"] == 0


@pytest.mark.parametrize("change", ["evidence", "order", "agent", "wm"])
def test_protocol_and_case_fingerprints_must_match(change):
    p = protocol()
    c = p["cases"][0]
    r = run(p, c, "rpm_wm")
    if change == "evidence":
        c["evidence_sha256"] = "1" * 64
    elif change == "order":
        c["candidate_ids"].reverse()
    elif change == "agent":
        p["agent"]["max_turns"] += 1
    else:
        p["wm_model_sha256"] = "1" * 64
    assert ev.validate_run(r, p, c, "rpm_wm") == "provenance_mismatch"


@pytest.mark.parametrize(
    "decision",
    [
        {"choice": "b", "ranking": ["a", "b"], "confidence": 0.5, "rationale": "x"},
        {"choice": "a", "ranking": ["a", "a"], "confidence": 0.5, "rationale": "x"},
        {"choice": "a", "ranking": ["a"], "confidence": 0.5, "rationale": "x"},
        {"choice": "a", "ranking": ["a", "b"], "confidence": True, "rationale": "x"},
        {"choice": "a", "ranking": ["a", "b"], "confidence": float("nan"), "rationale": "x"},
        {"choice": "a"},
    ],
)
def test_strict_final_decision(decision):
    p = protocol()
    c = p["cases"][0]
    r = run(p, c, "rpm")
    r["decision"] = decision
    assert ev.validate_run(r, p, c, "rpm") in {
        "invalid_ranking",
        "invalid_decision_fields",
        "invalid_decision_schema",
    }


def test_unknown_cost_remains_unknown():
    p, labels, runs = fixture()
    runs[0]["cost_usd"] = None
    result = ev.score(p, labels, runs, bootstrap_replicates=100)
    assert result["cost"]["known_total_usd"] == pytest.approx(1.4)
    assert result["cost"]["attempts_with_unknown_cost"] == 1


def test_duplicate_retry_is_rejected_even_if_first_attempt_failed():
    p, labels, runs = fixture()
    duplicate = copy.deepcopy(runs[0])
    runs[0]["status"] = "timeout"
    with pytest.raises(ValueError, match="Duplicate attempt"):
        ev.score(p, labels, runs + [duplicate])


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 1.1, True])
def test_invalid_label_not_silently_zero(value):
    p, labels, runs = fixture()
    labels["one"]["a"] = value
    with pytest.raises(ValueError, match="Official accuracy"):
        ev.score(p, labels, runs)


def test_no_hidden_label_fields_in_case_contract():
    p = protocol()
    p["cases"][0]["accuracy"] = 0.9
    with pytest.raises(ValueError, match="keep target labels separate"):
        ev.validate_protocol(p)


def test_seeded_results_are_reproducible_and_json_finite():
    p, labels, runs = fixture()
    a = ev.score(p, labels, runs, bootstrap_replicates=100)
    b = ev.score(p, labels, list(reversed(runs)), bootstrap_replicates=100)
    assert a == b
    json.dumps(a, allow_nan=False)
