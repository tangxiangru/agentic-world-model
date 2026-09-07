import copy
import hashlib
import json

import pytest

from tools.outcome_prediction import wm_agent as agent
from tools.outcome_prediction import wm_evaluate as evaluate


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def encode(rows):
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()


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
            "common_system_sha256": sha(agent.COMMON_SYSTEM),
            "wm_instruction_sha256": sha(agent.WM_INSTRUCTIONS),
            "cli_version": "2.1.261",
        },
        "cases": [
            {
                "case_id": "case-one",
                "cell_id": "run-one",
                "benchmark": "gsm8k",
                "candidate_ids": ["candidate-a", "candidate-b"],
                "evidence_sha256": "d" * 64,
                "candidate_payloads_sha256": "e" * 64,
                "packet_audit_sha256": "f" * 64,
            }
        ],
    }


def fixture(arm="rpm", *, lifecycle=False, call=True):
    p = protocol()
    tools = sorted(evaluate.allowed_tools(arm)) + ([agent.LIFECYCLE_TOOL] if lifecycle else [])
    initial = {
        "type": "system",
        "subtype": "init",
        "session_id": "session-one",
        "claude_code_version": p["agent"]["cli_version"],
        "model": "fixed-model",
        "permissionMode": "dontAsk",
        "skills": [],
        "plugins": [],
        "agents": [],
        "slash_commands": [],
        "tools": tools,
        "mcp_servers": [{"name": "evidence", "status": "connected"}],
    }
    rows = [initial]
    audit = []
    if call:
        name = "predict_candidate" if arm == "rpm_wm" else "read_evidence"
        args = {"candidate_id": "candidate-a"} if arm == "rpm_wm" else {"path": "recipes/a.json"}
        rows.extend(
            [
                {
                    "type": "assistant",
                    "session_id": "session-one",
                    "parent_tool_use_id": None,
                    "message": {
                        "role": "assistant",
                        "model": "fixed-model",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-one",
                                "name": agent.PREFIX + name,
                                "input": args,
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "session_id": "session-one",
                    "parent_tool_use_id": None,
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call-one", "content": "{}"}
                        ],
                    },
                },
            ]
        )
        audit = [
            {
                "time_unix_ns": 1000,
                "tool": name,
                "arguments_sha256": sha(agent._json(args)),
                "status": "ok",
                "elapsed_ms": 1.2,
            }
        ]
    decision = {
        "choice": "candidate-a",
        "ranking": ["candidate-a", "candidate-b"],
        "confidence": 0.7,
        "rationale": "Complete approved recipe comparison.",
    }
    rows.extend(
        [
            {
                "type": "assistant",
                "session_id": "session-one",
                "parent_tool_use_id": None,
                "message": {
                    "role": "assistant",
                    "model": "fixed-model",
                    "content": [{"type": "text", "text": json.dumps(decision)}],
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "session_id": "session-one",
                "is_error": False,
                "terminal_reason": "completed",
                "stop_reason": "end_turn",
                "num_turns": 3,
                "result": json.dumps(decision),
                "total_cost_usd": 0.2,
                "permission_denials": [],
                "modelUsage": {"fixed-model": {"costUSD": 0.2}},
            },
        ]
    )
    return p, rows, audit


def normalize(p, rows, audit, arm="rpm", **overrides):
    options = {
        "protocol": p,
        "case_id": "case-one",
        "arm": arm,
        "returncode": 0,
        "elapsed_seconds": 2,
    }
    options.update(overrides)
    return agent.normalize_stream(encode(rows), encode(audit), **options)


def build(p, arm="rpm", **overrides):
    options = {
        "python_executable": "/repo/.venv/bin/python",
        "repository_root": "/repo",
        "server_config_path": "/private/run/config.json",
    }
    options.update(overrides)
    return agent.build_command(p, "case-one", arm, **options)


def test_matched_command_only_treatment_differs():
    p = protocol()
    rpm, wm = build(p), build(p, "rpm_wm")
    assert rpm["stdin"] == wm["stdin"]
    assert rpm["env_overrides"] == wm["env_overrides"]
    assert rpm["timeout_seconds"] == wm["timeout_seconds"] == 300
    assert rpm["requires_runtime_verification"]
    for result in (rpm, wm):
        cmd = result["argv"]
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
        assert "--restricted" in cmd and "--strict-mcp-config" in cmd
        assert "--safe-mode" not in cmd and "--bare" not in cmd
        assert "--dangerously-skip-permissions" not in cmd and "--json-schema" not in cmd
        assert cmd[cmd.index("--max-budget-usd") + 1] == "1.0"
        mcp = json.loads(cmd[cmd.index("--mcp-config") + 1])
        assert set(mcp["mcpServers"]) == {"evidence"}
        assert mcp["mcpServers"]["evidence"]["command"] == "/repo/.venv/bin/python"
        assert not any("TOKEN" in k or "API_KEY" in k for k in result["env_overrides"])
    base_system = rpm["argv"][rpm["argv"].index("--system-prompt") + 1]
    wm_system = wm["argv"][wm["argv"].index("--system-prompt") + 1]
    assert base_system == agent.COMMON_SYSTEM
    assert wm_system == base_system + "\n" + agent.WM_INSTRUCTIONS
    assert "ancestor" in base_system and "middle checkpoint" in base_system
    assert "predict_candidate" not in rpm["argv"][rpm["argv"].index("--allowedTools") + 1]
    assert "predict_candidate" in wm["argv"][wm["argv"].index("--allowedTools") + 1]


def test_builder_is_pure_and_does_not_copy_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-copy-this-fixture")
    built = build(protocol())
    assert "do-not-copy-this-fixture" not in json.dumps(built)
    built["env_overrides"]["ENABLE_TOOL_SEARCH"] = "true"
    assert build(protocol())["env_overrides"]["ENABLE_TOOL_SEARCH"] == "false"


@pytest.mark.parametrize(
    "override",
    [
        {"common_system": "changed"},
        {"wm_instruction": "changed"},
        {"python_executable": "python"},
        {"server_config_path": "config.json"},
        {"repository_root": "../repo"},
        {"claude_executable": "bad\x00exec"},
    ],
)
def test_builder_rejects_unpinned_instructions_and_implicit_paths(override):
    with pytest.raises(ValueError):
        build(protocol(), **override)


@pytest.mark.parametrize("arm", ["rpm", "rpm_wm"])
def test_valid_stream_matches_scorer_and_pins_both_raw_captures(arm):
    p, rows, audit = fixture(arm)
    result = normalize(p, rows, audit, arm)
    assert result["status"] == "success"
    assert result["normalization"]["failure_reason"] is None
    assert result["transcript_sha256"] == hashlib.sha256(encode(rows)).hexdigest()
    assert result["server_audit_sha256"] == hashlib.sha256(encode(audit)).hexdigest()
    assert not result["normalization"]["semantic_leakage_certified"]
    assert evaluate.validate_run(result, p, p["cases"][0], arm) is None


def test_wm_usage_optional_for_intent_to_treat():
    p, rows, audit = fixture("rpm_wm", call=False)
    result = normalize(p, rows, audit, "rpm_wm")
    assert result["status"] == "success" and result["tool_calls"] == []


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("claude_code_version", "old", "cli_version_mismatch"),
        ("model", "different", "model_mismatch"),
        ("permissionMode", "bypassPermissions", "permission_mode_mismatch"),
        ("tools", ["Bash"], "tool_inventory_mismatch"),
        ("skills", ["secret-skill"], "unexpected_skills"),
        ("plugins", [{"name": "x"}], "unexpected_plugins"),
        ("agents", ["general-purpose"], "unexpected_agents"),
        ("slash_commands", ["run"], "unexpected_slash_commands"),
        ("mcp_servers", [{"name": "evidence", "status": "failed"}], "mcp_inventory_mismatch"),
    ],
)
def test_init_contract_failure(key, value, reason):
    p, rows, audit = fixture()
    rows[0][key] = value
    out = normalize(p, rows, audit)
    assert out["status"] == "invalid"
    assert out["normalization"]["failure_reason"] == reason


def test_duplicate_init_and_terminal_and_nonfinal_terminal():
    p, rows, audit = fixture()
    for extra in (rows[0], rows[-1]):
        out = normalize(p, rows + [extra], audit)
        assert out["normalization"]["failure_reason"] == "init_or_terminal_count"
    out = normalize(p, rows + [rows[-2]], audit)
    assert out["normalization"]["failure_reason"] == "init_or_terminal_order"


def test_ancestor_outcome_is_not_certified_by_normalizer():
    # Surface compliance cannot prove source content obeys the scientific boundary.
    p, rows, audit = fixture()
    rows[2]["message"]["content"][0]["content"] = "An unapproved ancestor score was inserted."
    out = normalize(p, rows, audit)
    assert out["status"] == "success"
    assert out["normalization"]["semantic_leakage_certified"] is False


@pytest.mark.parametrize(
    "name", ["Bash", "mcp__other__read_evidence", "mcp__evidence__predict_candidate"]
)
def test_unapproved_tool_call_rejected(name):
    p, rows, audit = fixture()
    rows[1]["message"]["content"][0]["name"] = name
    assert normalize(p, rows, audit)["normalization"]["failure_reason"] == "unexpected_tool_call"


@pytest.mark.parametrize(
    "args", [{"candidate_id": "unapproved"}, {"candidate_id": "candidate-a", "payload": {}}]
)
def test_crosscandidate_or_arbitrary_wm_query_rejected(args):
    p, rows, audit = fixture("rpm_wm")
    rows[1]["message"]["content"][0]["input"] = args
    out = normalize(p, rows, audit, "rpm_wm")
    assert out["status"] == "invalid"
    assert out["normalization"]["failure_reason"] in {
        "out_of_scope_wm_query",
        "invalid_tool_arguments",
    }


@pytest.mark.parametrize("path", ["/private/labels.json", "../labels.json", "a/../b", "a\\b"])
def test_evidence_path_escape_attempt_fails(path):
    p, rows, audit = fixture()
    rows[1]["message"]["content"][0]["input"]["path"] = path
    assert normalize(p, rows, audit)["normalization"]["failure_reason"] == "invalid_evidence_path"


@pytest.mark.parametrize(
    ("where", "key", "value", "reason"),
    [
        (1, "model", "other-model", "assistant_model_mismatch"),
        (1, "role", "user", "message_role_mismatch"),
    ],
)
def test_assistant_identity(where, key, value, reason):
    p, rows, audit = fixture()
    rows[where]["message"][key] = value
    assert normalize(p, rows, audit)["normalization"]["failure_reason"] == reason


@pytest.mark.parametrize("change", ["subagent", "session", "origin", "hook", "memory"])
def test_extra_context_or_agent_events_fail_closed(change):
    p, rows, audit = fixture()
    if change == "subagent":
        rows[1]["parent_tool_use_id"] = "parent-call"
    elif change == "session":
        rows[1]["session_id"] = "other-session"
    elif change == "origin":
        rows[1]["origin"] = {"kind": "peer"}
    else:
        rows.insert(
            1,
            {
                "type": "system",
                "subtype": "hook_started" if change == "hook" else "memory_recall",
                "session_id": "session-one",
            },
        )
    assert normalize(p, rows, audit)["status"] == "invalid"


@pytest.mark.parametrize("change", ["missing", "extra", "hash", "status", "unknown-tool"])
def test_server_audit_must_match_every_dispatch(change):
    p, rows, audit = fixture()
    if change == "missing":
        audit = []
    elif change == "extra":
        audit.append(copy.deepcopy(audit[0]))
    elif change == "hash":
        audit[0]["arguments_sha256"] = "1" * 64
    elif change == "status":
        audit[0]["status"] = "error"
    else:
        audit[0]["tool"] = "[unknown]"
    assert normalize(p, rows, audit)["normalization"]["failure_reason"] == "server_audit_mismatch"


def test_parallel_calls_match_audit_in_any_completion_order():
    p, rows, audit = fixture()
    extra = copy.deepcopy(rows[1]["message"]["content"][0])
    extra["id"] = "call-two"
    extra["input"] = {"path": "recipes/b.json", "count": 16, "offset": 0}
    rows[1]["message"]["content"].append(extra)
    rows[2]["message"]["content"].append(
        {"type": "tool_result", "tool_use_id": "call-two", "content": "{}"}
    )
    audit.insert(0, {**audit[0], "arguments_sha256": sha(agent._json(extra["input"]))})
    out = normalize(p, rows, audit)
    assert out["status"] == "success" and out["normalization"]["tool_result_count"] == 2


def test_observed_cli_metadata_and_matching_tool_progress_are_accepted():
    p, rows, audit = fixture()
    rows.insert(
        2,
        {
            "type": "tool_progress",
            "session_id": "session-one",
            "tool_use_id": "call-one",
            "tool_name": "mcp__evidence__read_evidence",
            "parent_tool_use_id": None,
            "elapsed_time_seconds": 0.5,
        },
    )
    rows.insert(
        1,
        {
            "type": "system",
            "subtype": "thinking_tokens",
            "session_id": "session-one",
            "estimated_tokens": 100,
            "estimated_tokens_delta": 100,
        },
    )
    rows.insert(
        1,
        {
            "type": "rate_limit_event",
            "session_id": "session-one",
            "rate_limit_info": {"status": "allowed", "resetsAt": 1000},
        },
    )
    out = normalize(p, rows, audit)
    assert out["status"] == "success" and out["normalization"]["metadata_event_count"] == 3


@pytest.mark.parametrize("change", ["unknown-call", "extra-content", "unknown-tool"])
def test_tool_progress_cannot_introduce_foreign_calls_or_content(change):
    p, rows, audit = fixture()
    event = {
        "type": "tool_progress",
        "session_id": "session-one",
        "tool_use_id": "call-one",
        "tool_name": "mcp__evidence__read_evidence",
        "elapsed_time_seconds": 1,
    }
    if change == "unknown-call":
        event["tool_use_id"] = "other"
    elif change == "extra-content":
        event["stdout"] = "not metadata"
    else:
        event["tool_name"] = "Bash"
    rows.insert(2, event)
    assert normalize(p, rows, audit)["normalization"]["failure_reason"] == "invalid_tool_progress"


def test_legitimate_tool_error_can_be_followed_by_valid_decision():
    p, rows, audit = fixture()
    rows[2]["message"]["content"][0]["is_error"] = True
    audit[0]["status"] = "error"
    assert normalize(p, rows, audit)["status"] == "success"


@pytest.mark.parametrize("change", ["orphan", "duplicate-result", "duplicate-call", "unresolved"])
def test_call_result_bijection(change):
    p, rows, audit = fixture()
    if change == "orphan":
        rows[2]["message"]["content"][0]["tool_use_id"] = "other"
    elif change == "duplicate-result":
        rows[2]["message"]["content"].append(copy.deepcopy(rows[2]["message"]["content"][0]))
    elif change == "duplicate-call":
        rows[1]["message"]["content"].append(copy.deepcopy(rows[1]["message"]["content"][0]))
    else:
        del rows[2]
    assert normalize(p, rows, audit)["status"] == "invalid"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("subtype", "error_max_turns"),
        ("is_error", True),
        ("terminal_reason", "tool_deferred"),
        ("stop_reason", "max_tokens"),
        ("permission_denials", [{"tool_name": "Bash"}]),
        ("num_turns", 13),
        ("num_turns", True),
        ("api_error_status", 429),
    ],
)
def test_terminal_failure_never_promotes_decision(key, value):
    p, rows, audit = fixture()
    rows[-1][key] = value
    out = normalize(p, rows, audit)
    assert out["status"] == "invalid" and out["decision"] is None
    assert evaluate.validate_run(out, p, p["cases"][0], "rpm") == "runner_failure"


def test_timeout_and_nonzero_exit_preserve_known_cost():
    p, rows, audit = fixture()
    for overrides in ({"timed_out": True}, {"elapsed_seconds": 301}, {"returncode": 1}):
        out = normalize(p, rows, audit, **overrides)
        assert out["status"] != "success" and out["decision"] is None
        assert out["cost_usd"] == 0.2


def test_unknown_cost_stays_unknown_and_auxiliary_model_is_disclosed():
    p, rows, audit = fixture()
    del rows[-1]["total_cost_usd"]
    rows[-1]["modelUsage"]["helper-model"] = {"costUSD": 0.01}
    out = normalize(p, rows, audit)
    assert out["status"] == "success" and out["cost_usd"] is None
    assert out["normalization"]["auxiliary_model_usage"] == {"helper-model": {"costUSD": 0.01}}


def test_cost_above_soft_cli_cap_is_preserved_not_erased():
    p, rows, audit = fixture()
    rows[-1]["total_cost_usd"] = 1.1
    out = normalize(p, rows, audit)
    assert out["status"] == "success" and out["cost_usd"] == 1.1


@pytest.mark.parametrize("value", [-1, "0.2", True])
def test_invalid_cost_fails(value):
    p, rows, audit = fixture()
    rows[-1]["total_cost_usd"] = value
    out = normalize(p, rows, audit)
    assert out["status"] == "invalid" and out["cost_usd"] is None


@pytest.mark.parametrize(
    "value",
    [
        "```json\n{}\n```",
        '{"choice":"candidate-a"}',
        '{"choice":"candidate-a","choice":"candidate-b"}',
        '{"choice":"candidate-a","ranking":["candidate-a","candidate-a"],"confidence":0.5,"rationale":"x"}',
        '{"choice":"candidate-a","ranking":["candidate-a","candidate-b"],"confidence":NaN,"rationale":"x"}',
    ],
)
def test_strict_decision_json(value):
    p, rows, audit = fixture()
    rows[-1]["result"] = value
    rows[-2]["message"]["content"][0]["text"] = value
    assert normalize(p, rows, audit)["status"] == "invalid"


def test_final_result_must_equal_last_assistant_decision():
    p, rows, audit = fixture()
    different = json.loads(rows[-1]["result"])
    different["confidence"] = 0.3
    rows[-1]["result"] = json.dumps(different)
    assert (
        normalize(p, rows, audit)["normalization"]["failure_reason"] == "terminal_decision_mismatch"
    )


def test_lifecycle_inventory_and_call_remain_visible():
    p, rows, audit = fixture(lifecycle=True)
    rows[1]["message"]["content"].append(
        {"type": "tool_use", "name": "EndConversation", "id": "end", "input": {}}
    )
    rows[2]["message"]["content"].append(
        {"type": "tool_result", "tool_use_id": "end", "content": "ended"}
    )
    out = normalize(p, rows, audit)
    assert out["status"] == "success"
    assert "EndConversation" in out["tools_available"]
    assert out["normalization"]["lifecycle_called"]
    assert evaluate.validate_run(out, p, p["cases"][0], "rpm") is None
    del rows[-1]["result"]
    assert normalize(p, rows, audit)["status"] == "invalid"


@pytest.mark.parametrize(
    "transcript", [b"warning\n", b'{"type":"a","type":"b"}\n', b'{"x":1e999}\n', b"[]\n", b"\xff"]
)
def test_strict_stream_parser_returns_failure_not_crash(transcript):
    out = agent.normalize_stream(
        transcript,
        b"",
        protocol=protocol(),
        case_id="case-one",
        arm="rpm",
        returncode=0,
        elapsed_seconds=1,
    )
    assert out["status"] == "invalid"


def test_malformed_text_returns_failure_not_crash():
    p, rows, audit = fixture()
    rows[-2]["message"]["content"][0]["text"] = 123
    assert normalize(p, rows, audit)["normalization"]["failure_reason"] == "malformed_text"


@pytest.mark.parametrize("content", [None, 123, [{"type": "image", "source": {}}]])
def test_gateway_results_cannot_introduce_nontext_payloads(content):
    p, rows, audit = fixture()
    rows[2]["message"]["content"][0]["content"] = content
    out = normalize(p, rows, audit)
    assert out["normalization"]["failure_reason"] == "invalid_tool_result_content"


def test_mcp_text_content_array_accepted():
    p, rows, audit = fixture()
    rows[2]["message"]["content"][0]["content"] = [{"type": "text", "text": "{}"}]
    assert normalize(p, rows, audit)["status"] == "success"


def test_no_existing_model_or_cases_are_frozen_by_this_module():
    assert "data_v3" not in agent.COMMON_SYSTEM + agent.WM_INSTRUCTIONS
    with pytest.raises(ValueError, match="Unknown"):
        agent.normalize_stream(
            b"",
            b"",
            protocol=protocol(),
            case_id="unknown",
            arm="rpm",
            returncode=0,
            elapsed_seconds=1,
        )
