"""Generic JSON-only, tool-free Claude invocation for local-fit selection.

No model fitting or file access is exposed to Claude. A caller supplies a frozen
prompt and policy, persists the returned record, and owns any bounded retries.
Invalid policies raise before invocation; execution/response failures return
``response=None`` and an error while retaining the raw output for audit.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def parse_object(text):
    """One finite JSON object, optionally in one complete Markdown fence."""
    if not isinstance(text, str):
        raise TypeError("Response must be a JSON object encoded as text")
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    value = json.loads(cleaned, object_pairs_hook=_object_pairs)
    if not isinstance(value, dict):
        raise TypeError("Response must be one JSON object")
    _finite(value)
    return value


def _validate(prompt, policy, system):
    if not isinstance(prompt, str) or not isinstance(policy, Mapping):
        raise TypeError("A text prompt and policy mapping are required")
    required = {"model_requested", "effort", "max_output_tokens", "max_cli_network_retries",
                "per_call_budget_usd", "timeout_seconds"}
    if not required.issubset(policy):
        raise ValueError("Missing invocation policy fields: " + str(sorted(required - set(policy))))
    chosen_system = policy.get("system") if system is None else system
    if not isinstance(chosen_system, str) or not chosen_system.strip():
        raise TypeError("A nonempty explicit system prompt is required")
    model = policy["model_requested"]
    if not isinstance(model, str) or not re.fullmatch(r"claude-[A-Za-z0-9._-]+", model):
        raise ValueError("An explicit full Claude model identifier is required")
    if policy["effort"] not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Invalid effort")
    for key in ("max_output_tokens", "max_cli_network_retries"):
        if type(policy[key]) is not int or policy[key] < (1 if key == "max_output_tokens" else 0):
            raise ValueError("Invalid bounded integer policy: " + key)
    for key in ("per_call_budget_usd", "timeout_seconds"):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid positive finite policy: " + key)
    return chosen_system


def _text(value):
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _events(stdout):
    events = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            _finite(value)
            events.append(value if isinstance(value, dict) else {"unexpected_json": value})
        except (ValueError, RecursionError):
            events.append({"unparsed_stdout": line})
    return events


def _decode(events, requested, returncode, stderr):
    init = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    results = [e for e in events if e.get("type") == "result"]
    result = results[-1] if results else {}
    assistants = [e.get("message") for e in events if e.get("type") == "assistant"]
    models = sorted({m["model"] for m in assistants
                     if isinstance(m, dict) and isinstance(m.get("model"), str)})
    response, error = None, None
    try:
        if returncode or len(results) != 1 or result.get("is_error") is not False:
            raise ValueError("CLI request failed: " + str(result.get("subtype", "no_result"))
                             + ": " + _text(result.get("result") or stderr)[:400])
        if len(init) != 1 or any(init[0].get(k) != [] for k in ("tools", "mcp_servers", "plugins", "skills")):
            raise ValueError("Missing tool-free initialization or unexpected tools/customizations")
        if init[0].get("model") != requested or models != [requested]:
            raise ValueError("Requested/resolved model identity mismatch: " + str(models))
        for message in assistants:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                raise TypeError("Malformed assistant message")
            if message.get("model") != requested:
                raise ValueError("Requested/resolved model identity missing or mismatched")
            for block in message["content"]:
                if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                    raise TypeError("Malformed assistant content")
                if "tool" in block["type"]:
                    raise ValueError("Unexpected tool use in tool-free inference")
        response = parse_object(result.get("result"))
    except (ValueError, TypeError, RecursionError) as exc:
        error = str(exc)
    return response, error, models, result


def call_json(prompt, policy, *, system=None):
    """Run one bounded Claude invocation; return a JSON response plus provenance.

    Policy keys match ``wm_fresh_llm_baseline.call_model``: model_requested,
    effort, system, max_output_tokens, max_cli_network_retries,
    per_call_budget_usd, timeout_seconds. This function never retries a failed
    invocation, writes artifacts, prints credentials, or changes user settings.
    """
    chosen_system = _validate(prompt, policy, system)
    command = [
        "claude", "-p", "--safe-mode", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--no-session-persistence", "--disable-slash-commands", "--setting-sources", "",
        "--tools", "", "--output-format", "stream-json", "--verbose",
        "--effort", policy["effort"], "--max-budget-usd", str(policy["per_call_budget_usd"]),
        "--model", policy["model_requested"], "--system-prompt", chosen_system,
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_") and k != "CLAUDECODE"}
    env.update(CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(policy["max_output_tokens"]),
               CLAUDE_CODE_MAX_RETRIES=str(policy["max_cli_network_retries"]))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    stdout, stderr, returncode, execution_error = "", "", None, None
    try:
        with tempfile.TemporaryDirectory(prefix="wm-local-agent-no-data-") as cwd:
            proc = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                  cwd=cwd, env=env, timeout=policy["timeout_seconds"], check=False)
        stdout, stderr, returncode = _text(proc.stdout), _text(proc.stderr), proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _text(exc.stdout), _text(exc.stderr)
        execution_error = "request_timeout"
    except OSError as exc:
        execution_error = "invocation_error: " + str(exc)
    events = _events(stdout)
    response, error, models, result = _decode(events, policy["model_requested"], returncode, stderr)
    if execution_error:
        response, error = None, execution_error
    return {
        "started_at": started_at, "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_requested": policy["model_requested"], "models_resolved": models,
        "effort": policy["effort"], "response": response, "error": error,
        "cost_usd_reported": result.get("total_cost_usd"), "usage": result.get("usage"),
        "model_usage": result.get("modelUsage"), "returncode": returncode,
        "raw_events": events, "stdout": stdout, "stderr": stderr,
        "timeout_seconds": policy["timeout_seconds"],
        "max_output_tokens": policy["max_output_tokens"],
    }
