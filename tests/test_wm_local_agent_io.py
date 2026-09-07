import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.outcome_prediction import wm_local_agent_io as io

POLICY = {
    "model_requested": "claude-opus-5", "effort": "high", "system": "Return a JSON selection.",
    "max_output_tokens": 2048, "max_cli_network_retries": 1,
    "per_call_budget_usd": 0.5, "timeout_seconds": 240,
}


def events(response=None):
    return [
        {"type": "system", "subtype": "init", "model": "claude-opus-5",
         "tools": [], "mcp_servers": [], "plugins": [], "skills": []},
        {"type": "assistant", "message": {"model": "claude-opus-5", "content": [{"type": "text", "text": "selection"}]}},
        {"type": "result", "is_error": False, "subtype": "success",
         "result": json.dumps(response or {"model": "ridge", "ids": ["a", "b"]}),
         "total_cost_usd": 0.01, "usage": {"input_tokens": 12, "output_tokens": 15},
         "modelUsage": {"claude-opus-5": {"costUSD": 0.01}}},
    ]


def install_mock(monkeypatch, payload, *, returncode=0):
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs))
        assert list(Path(kwargs["cwd"]).iterdir()) == []
        return SimpleNamespace(stdout="\n".join(json.dumps(e) for e in payload),
                               stderr="diagnostic", returncode=returncode)

    monkeypatch.setattr(io.subprocess, "run", run)
    return seen


def test_generic_selection_and_exact_isolation(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SUBAGENT_MODEL", "forbidden-inherited-model")
    monkeypatch.setenv("CLAUDECODE", "nested")
    seen = install_mock(monkeypatch, events())
    result = io.call_json("safe selection payload", POLICY)
    assert result["response"] == {"model": "ridge", "ids": ["a", "b"]}
    assert result["error"] is None and result["models_resolved"] == ["claude-opus-5"]
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 15}
    assert result["cost_usd_reported"] == 0.01
    assert len(seen) == 1
    command, kwargs = seen[0]
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in command
    assert "--fallback-model" not in command
    assert kwargs["input"] == "safe selection payload"
    assert "agentic-world-model" not in kwargs["cwd"]
    assert not Path(kwargs["cwd"]).exists()
    assert kwargs["timeout"] == 240 and kwargs["check"] is False
    assert "CLAUDECODE" not in kwargs["env"]
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in kwargs["env"]
    assert kwargs["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "2048"
    assert kwargs["env"]["CLAUDE_CODE_MAX_RETRIES"] == "1"


def test_system_override(monkeypatch):
    seen = install_mock(monkeypatch, events())
    io.call_json("prompt", POLICY, system="Override system")
    command = seen[0][0]
    assert command[command.index("--system-prompt") + 1] == "Override system"


@pytest.mark.parametrize("payload", ["null", "[]", "0.3", '"text"',
                                     '{"a":NaN}', '{"a":[Infinity]}',
                                     '{"a":1,"a":2}', '{"a":1e999}',
                                     'Prose {"a":1}', '{"a":1}{"b":2}'])
def test_invalid_json_objects(payload):
    with pytest.raises((ValueError, TypeError)):
        io.parse_object(payload)


@pytest.mark.parametrize("payload", ['{"a":0,"enabled":false,"optional":null}',
                                     '```json\n{"a":0,"enabled":false,"optional":null}\n```'])
def test_valid_generic_json_values(payload):
    assert io.parse_object(payload) == {"a": 0, "enabled": False, "optional": None}


@pytest.mark.parametrize("field", ["tools", "mcp_servers", "plugins", "skills"])
def test_nonempty_or_missing_isolation_rejected(monkeypatch, field):
    payload = events()
    payload[0][field] = ["unexpected"]
    install_mock(monkeypatch, payload)
    result = io.call_json("prompt", POLICY)
    assert result["response"] is None and "initialization" in result["error"]
    payload[0].pop(field)
    assert io.call_json("prompt", POLICY)["response"] is None


@pytest.mark.parametrize("where", ["init", "assistant", "extra"])
def test_model_mismatch_rejected(monkeypatch, where):
    payload = events()
    if where == "init":
        payload[0]["model"] = "claude-other"
    elif where == "assistant":
        payload[1]["message"]["model"] = "claude-other"
    else:
        payload.insert(2, {"type": "assistant", "message": {"model": "claude-other", "content": []}})
    install_mock(monkeypatch, payload)
    result = io.call_json("prompt", POLICY)
    assert result["response"] is None and "model identity" in result["error"]


@pytest.mark.parametrize("block_type", ["tool_use", "server_tool_use", "mcp_tool_use"])
def test_tool_use_rejected(monkeypatch, block_type):
    payload = events()
    payload[1]["message"]["content"] = [{"type": block_type}]
    install_mock(monkeypatch, payload)
    result = io.call_json("prompt", POLICY)
    assert result["response"] is None and "tool use" in result["error"]


@pytest.mark.parametrize("message", [None, [], {"model": "claude-opus-5", "content": None},
                                      {"model": "claude-opus-5", "content": [None]}])
def test_malformed_assistant_events_do_not_crash(monkeypatch, message):
    payload = events()
    payload[1]["message"] = message
    install_mock(monkeypatch, payload)
    result = io.call_json("prompt", POLICY)
    assert result["response"] is None and result["error"]
    assert result["raw_events"] == payload


def test_missing_model_in_additional_assistant_fails_closed(monkeypatch):
    payload = events()
    payload.insert(2, {"type": "assistant", "message": {"content": []}})
    install_mock(monkeypatch, payload)
    result = io.call_json("prompt", POLICY)
    assert result["response"] is None and "model identity" in result["error"]


def test_malformed_stream_is_json_serializable_for_preservation():
    payload = io._events('null\n[]\nnot-json\nNaN\n{"a":1e999}\n')
    json.dumps(payload, allow_nan=False)
    assert payload[-2:] == [{"unparsed_stdout": "NaN"}, {"unparsed_stdout": '{"a":1e999}'}]


def test_failed_cli_preserves_error_response_and_no_retry(monkeypatch):
    payload = events()
    payload[-1].update(is_error=True, subtype="error_max_budget_usd", result=None)
    seen = install_mock(monkeypatch, payload, returncode=1)
    result = io.call_json("prompt", POLICY)
    assert len(seen) == 1 and result["response"] is None
    assert "error_max_budget_usd" in result["error"]
    assert result["cost_usd_reported"] == 0.01 and result["raw_events"] == payload


def test_timeout_retains_partial_bytes_and_does_not_retry(monkeypatch):
    seen = []
    partial = json.dumps(events()[0]).encode() + b"\n"

    def timeout(*args, **kwargs):
        seen.append(kwargs)
        raise subprocess.TimeoutExpired(args[0], 240, output=partial, stderr=b"partial diagnostic")

    monkeypatch.setattr(io.subprocess, "run", timeout)
    result = io.call_json("prompt", POLICY)
    assert len(seen) == 1 and result["error"] == "request_timeout"
    assert result["response"] is None and result["returncode"] is None
    assert result["stdout"] == partial.decode() and result["stderr"] == "partial diagnostic"
    assert result["raw_events"] == [events()[0]]


def test_unavailable_executable_returns_error(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("claude not found")

    monkeypatch.setattr(io.subprocess, "run", unavailable)
    result = io.call_json("prompt", POLICY)
    assert result["response"] is None and "invocation_error" in result["error"]


@pytest.mark.parametrize("key,bad", [("model_requested", "opus"), ("timeout_seconds", None),
                                    ("timeout_seconds", float("inf")), ("timeout_seconds", 0),
                                    ("max_output_tokens", True), ("max_cli_network_retries", -1),
                                    ("per_call_budget_usd", 0), ("effort", "unknown")])
def test_invalid_policy_before_external_invocation(monkeypatch, key, bad):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must validate before external invocation")

    monkeypatch.setattr(io.subprocess, "run", forbidden)
    policy = copy.deepcopy(POLICY)
    policy[key] = bad
    with pytest.raises((ValueError, TypeError)):
        io.call_json("prompt", policy)
