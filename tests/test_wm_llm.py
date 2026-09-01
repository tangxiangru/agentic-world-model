"""The autonomous WMA sees the full copied corpus and fails closed."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import pytest
import yaml

import awm.wm.agents.llm as llm_module
from awm.cli import build_parser
from awm.wm import scratch_server
from awm.wm.agents.llm import (
    ALLOWED_TOOLS,
    LLMAgent,
    _extract_tool_events,
    _response_schema,
    _validate_citations,
    _validate_grounding_references,
    _validate_raw_corpus,
    _validate_reported_models,
    _validate_reported_tools,
    _validate_server_tool_audit,
    _validate_tool_trace,
    _validation_repair_guidance,
    _vertex_subprocess_env,
)
from awm.wm.memory import Memory
from awm.wm.runtime import Session
from awm.wm.schema import WMError
from awm.wm.scratch_server import call_tool, handle_message, probe_sandbox


def _init_row(tools: tuple[str, ...] = ALLOWED_TOOLS, model: str = "claude-opus-5") -> dict:
    return {
        "type": "system",
        "subtype": "init",
        "tools": [*tools],
        "model": model,
        "apiKeySource": "none",
    }


def _read_tool_rows(
    root: Path,
    path: Path,
    *,
    tool_id: str = "read-1",
    offset: int = 0,
    limit: int = scratch_server.MAX_READ_BYTES,
) -> list[dict]:
    data = path.read_bytes()
    chunk = data[offset : offset + limit]
    more = offset + len(chunk) < len(data)
    relative = path.relative_to(root).as_posix()
    payload = json.dumps(
        {
            "root": 0,
            "path": relative,
            "offset": offset,
            "bytes": len(chunk),
            "next_offset": offset + len(chunk) if more else None,
            "content": chunk.decode(errors="replace"),
        },
        sort_keys=True,
    )
    return [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "mcp__awm_scratch__read_corpus",
                        "input": {"root": 0, "path": relative, "offset": offset, "limit": limit},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "is_error": False,
                        "content": [{"type": "text", "text": payload}],
                    }
                ]
            },
        },
    ]


def _complete_read_tool_rows(
    root: Path,
    path: Path,
    *,
    tool_id: str = "complete-read-1",
) -> list[dict]:
    data = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    payload = json.dumps(
        {
            "root": 0,
            "path": relative,
            "offset": 0,
            "bytes": len(data),
            "next_offset": None,
            "content": data.decode(errors="replace"),
        },
        sort_keys=True,
    )
    return [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "mcp__awm_scratch__read_corpus_complete",
                        "input": {"root": 0, "path": relative},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "is_error": False,
                        "content": [{"type": "text", "text": payload}],
                    }
                ]
            },
        },
    ]


def _historical_card(tag: str = "prior") -> dict:
    return {
        "schema_version": "awm-experiment-card-v1",
        "card_id": "exp-01",
        "problem": {"statement": f"{tag} arithmetic failures"},
        "hypothesis": {"claim": "worked examples help"},
        "setup": {
            "parent_checkpoint": {"path": "google/gemma-3-4b-pt", "origin": "base_model"},
            "data": [{"source": "generated arithmetic"}],
            "method": {"family": "sft", "hyperparams": {"lr": 1e-5}},
        },
        "evaluation": {"comparator": {"value": 0.3}},
        "result": {"execution": "completed", "measurements": [{"value": 0.4}]},
        "conclusion": {"verdict": "supported", "decision": "adopt"},
    }


def _current_card(session: Path) -> dict:
    return {
        "schema_version": "awm-experiment-card-v1",
        "card_id": "exp-01",
        "problem": {
            "statement": "The current model makes arithmetic slips.",
            "watch_set": {"path": str(session / "watch.jsonl"), "n": 4},
        },
        "hypothesis": {
            "claim": "SFT on worked examples will reduce slips.",
            "expected_effect": {"metric": "accuracy", "direction": "higher"},
        },
        "setup": {
            "method": {"family": "sft"},
            "data": [{"source": "current arithmetic"}],
            "progress": {"unit": "optimizer_step", "total": 100},
        },
        "evaluation": {"protocol": {"n": 10, "seed": 0}},
    }


def _seed(tmp_path: Path, *, include_test: bool = False) -> tuple[Path, Path]:
    source = tmp_path / "cards"
    train = source / "train" / "r-train000"
    train.mkdir(parents=True)
    (train / "exp-01.yaml").write_text(yaml.safe_dump(_historical_card()))
    coverage = {
        "expected_runs_by_side": {"train": 1, "test": 2},
        "runs_without_cards": {
            "by_side": {"train": [], "test": ["r-test-missing"]},
            "cause": "unknown",
            "evidence": "published corpus cannot distinguish the cause",
        },
    }
    if include_test:
        test = source / "test" / "r-test000"
        test.mkdir(parents=True)
        (test / "exp-01.yaml").write_text(yaml.safe_dump(_historical_card("test")))
    (source / "coverage.json").write_text(json.dumps(coverage))
    root = tmp_path / "memory"
    writer = Memory(root, session="seed", arm="null")
    assert writer.seed_from_exp_cards(source, side="train") == 1
    if include_test:
        assert writer.seed_from_exp_cards(source, side="test") == 1
    return source, root


def _raw_bundle(tmp_path: Path, *, include_test: bool = False) -> tuple[Path, list[Path]]:
    root = tmp_path / "prior-runs"
    rows = []
    manifest_rows = []
    traces = []
    sides = ["train", "test"] if include_test else ["train"]
    for i, side in enumerate(sides, 1):
        rel = Path(f"agent-{side}") / f"gsm8k_model_{i}"
        run = root / rel
        run.mkdir(parents=True)
        trace = run / "solve_out.txt"
        trace.write_text(f"complete {side} trajectory\nlaunch training\n")
        (run / "metrics.json").write_text(json.dumps({"accuracy": 0.3 + i / 10}))
        (run / "time_taken.txt").write_text("01:00:00\n")
        accuracy = 0.3 + i / 10
        rows.append(
            {
                "run": rel.as_posix(),
                "agent_config": f"agent-{side}",
                "run_name": f"gsm8k_model_{i}",
                "side": side,
                "base_model": "model",
                "accuracy": accuracy,
                "time_taken": "01:00:00",
                "has_trace": True,
                "trace_bytes": trace.stat().st_size,
                "path": f"/home/ben/prior_runs/{rel.as_posix()}",
            }
        )
        files = {}
        for name in ("solve_out.txt", "metrics.json", "time_taken.txt"):
            path = run / name
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest_rows.append({"run": rel.as_posix(), "side": side, "files": files})
        traces.append(trace.resolve())
    rows.sort(key=lambda row: (-row["accuracy"], row["run"]))
    (root / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    lines = [
        "# Prior runs",
        "",
        (
            f"{len(rows)} previous attempts at this task by autonomous agents, one directory each, "
            "laid out as `<agent config>/<run>/`. Each holds `solve_out.txt` (the agent's complete "
            "session trace), `metrics.json` (official accuracy), and `time_taken.txt`. "
            "No optional run artifacts or `task/` workspace snapshots are exposed."
        ),
        "",
        "Sorted by official accuracy, best first.",
        "",
        "| accuracy | base model | agent config | time | trace | path |",
        "|---:|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['accuracy']:.3f} | {row['base_model']} | {row['agent_config']} | "
            f"{row['time_taken']} | {row['trace_bytes'] // 1024} KB | `{row['path']}` |"
        )
    (root / "INDEX.md").write_text("\n".join(lines) + "\n")
    (root / "README.md").write_text(
        "Read-only copy of prior PostTrainBench runs for this task, built by "
        "tools/build_prior_runs.py. Start with INDEX.md.\n"
    )
    (root / "corpus-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "awm-prior-runs-v1",
                "split": {"id": "test/split-v1", "sides": sides},
                "dataset": {
                    "repo": "example/prior-runs",
                    "repo_type": "dataset",
                    "revision": "a" * 40,
                },
                "file_scope": ["solve_out.txt", "metrics.json", "time_taken.txt"],
                "run_count": len(manifest_rows),
                "runs": manifest_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root, traces


def test_seed_materialises_exact_idempotent_corpus_and_completeness(tmp_path: Path) -> None:
    source, root = _seed(tmp_path, include_test=True)
    copied = root / "corpus" / "train" / "r-train000" / "exp-01.yaml"
    assert copied.read_bytes() == (source / "train" / "r-train000" / "exp-01.yaml").read_bytes()
    manifest = json.loads((root / "corpus" / "test" / "manifest.json").read_text())
    assert manifest["expected_run_count"] == 2
    assert manifest["card_bearing_run_refs"] == ["r-test000"]
    assert manifest["missing_run_refs"] == ["r-test-missing"]
    assert manifest["missing_cause"] == "unknown"
    assert manifest["source_coverage_sha256"]

    writer = Memory(root, session="seed-again", arm="null")
    assert writer.seed_from_exp_cards(source, side="train") == 1
    rows = [
        json.loads(line) for line in (root / "structured" / "cards.jsonl").read_text().splitlines()
    ]
    train_rows = [row for row in rows if row["provenance"]["split_side"] == "train"]
    assert len(train_rows) == 1
    assert train_rows[0]["corpus_path"] == "corpus/train/r-train000/exp-01.yaml"


def test_llm_uses_full_visible_corpus_and_writes_audit(tmp_path: Path, monkeypatch) -> None:
    _source, root = _seed(tmp_path, include_test=True)
    session = tmp_path / "session"
    card_dir = session / "wm" / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    card = _current_card(session)
    (card_dir / "card.yaml").write_text(yaml.safe_dump(card))
    (card_dir / "launch.env.json").write_text(json.dumps({"SECRET_TOKEN": "must-not-be-readable"}))
    cited = root / "corpus" / "train" / "r-train000" / "exp-01.yaml"
    captured: dict = {}

    def fake_runner(**kwargs) -> int:
        captured.update(kwargs)
        response = {
            "claims": [
                {
                    "text": "A prior SFT card improved over its comparator.",
                    "citation_ids": ["C1", "C2"],
                }
            ],
            "citations": [
                {
                    "id": "C1",
                    "path": str(cited),
                    "locator": "result.measurements[0]",
                    "observation": "measurement value is 0.4",
                },
                {
                    "id": "C2",
                    "path": str(cited),
                    "locator": "evaluation.comparator.value",
                    "observation": "comparator is 0.3",
                },
            ],
            "objections": [],
        }
        rows = [
            _init_row(),
            *_complete_read_tool_rows(root / "corpus" / "train", cited, tool_id="tool-1"),
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "wma-session",
                "total_cost_usd": 0.12,
                "modelUsage": {"claude-opus-5": {"provider": "vertex"}},
                "structured_output": response,
            },
        ]
        kwargs["stdout_path"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        kwargs["stderr_path"].write_text("")
        return 0

    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    monkeypatch.setenv("CLAUDECODE", "parent-session")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "must-not-leak")
    memory = Memory(
        root,
        session="heldout",
        arm="llm",
        split_side="test",
        readonly=True,
        visible_sides=("train",),
    )
    agent = LLMAgent(session_dir=session, process_runner=fake_runner)
    brief = agent.on_proposal(
        card,
        [{"check": "example", "passed": True, "detail": "ok"}],
        memory,
        {"wma_model": "claude-opus-5", "wma_provider": "vertex"},
    )

    assert "[C1,C2]" in brief.summary
    assert brief.produced_by == "llm" and brief.degraded is None
    assert brief.evidence[0]["path"] == str(cited.resolve())
    assert brief.precedents == []  # no deterministic/top-k retrieval was performed
    assert captured["env"]["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert "CLAUDECODE" not in captured["env"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in captured["env"]
    tool_at = captured["argv"].index("--tools")
    assert captured["argv"][tool_at + 1] == ",".join(ALLOWED_TOOLS)
    assert "Bash" not in captured["argv"][tool_at + 1]
    assert "Read" not in ALLOWED_TOOLS
    allowed_at = captured["argv"].index("--allowedTools")
    assert captured["argv"][allowed_at + 1] == ",".join(ALLOWED_TOOLS)
    assert "--add-dir" not in captured["argv"]
    assert captured["cwd"].name == "model-workspace"
    assert list(captured["cwd"].iterdir()) == []
    assert "must-not-be-readable" not in captured["prompt"]
    audit = json.loads(Path(brief.audit["path"]).read_text())
    assert audit["model"] == "claude-opus-5" and audit["provider"] == "vertex"
    assert audit["reported_models"] == ["claude-opus-5"]
    assert audit["reported_providers"] == ["vertex"]
    assert audit["tool_event_count"] == 2 and audit["citation_count"] == 2
    assert (Path(brief.audit["path"]).parent / "stream.jsonl").is_file()
    assert (Path(brief.audit["path"]).parent / "tool-events.jsonl").is_file()
    mcp_config = json.loads((Path(brief.audit["path"]).parent / "mcp.json").read_text())
    assert mcp_config["mcpServers"]["awm_scratch"]["url"].startswith("<expired-")


def test_bounded_validation_retry_repairs_complete_card_read(tmp_path: Path, monkeypatch) -> None:
    _source, root = _seed(tmp_path)
    session = tmp_path / "retry-session"
    card_dir = session / "wm" / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    card = _current_card(session)
    (card_dir / "card.yaml").write_text(yaml.safe_dump(card))
    cited = root / "corpus" / "train" / "r-train000" / "exp-01.yaml"
    prompts: list[str] = []
    invocations: list[dict] = []

    def fake_runner(**kwargs) -> int:
        prompts.append(kwargs["prompt"])
        invocations.append(kwargs)
        response = {
            "claims": [{"text": "Prior measurement is 0.4.", "citation_ids": ["C1"]}],
            "citations": [
                {
                    "id": "C1",
                    "path": str(cited),
                    "locator": "result.measurements[0].value",
                    "observation": "measurement value is 0.4",
                }
            ],
            "objections": [],
        }
        if len(prompts) == 1:
            read_rows = [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "list-only",
                                "name": "mcp__awm_scratch__list_corpus",
                                "input": {"glob": "r-*/exp-*.yaml"},
                            }
                        ]
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "list-only",
                                "is_error": False,
                                "content": "{}",
                            }
                        ]
                    },
                },
            ]
        else:
            read_rows = _complete_read_tool_rows(root / "corpus" / "train", cited)
        rows = [
            _init_row(),
            *read_rows,
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "modelUsage": {"claude-opus-5": {"provider": "vertex"}},
                "structured_output": response,
            },
        ]
        kwargs["stdout_path"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        kwargs["stderr_path"].write_text("")
        return 0

    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    memory = Memory(
        root,
        session="retry",
        arm="llm",
        readonly=True,
        visible_sides=("train",),
    )
    brief = LLMAgent(session_dir=session, process_runner=fake_runner).on_proposal(
        card,
        [{"check": "example", "passed": True, "detail": "ok"}],
        memory,
        {
            "wma_model": "claude-opus-5",
            "wma_provider": "vertex",
            "wma_validation_attempts": 2,
            "wma_max_budget_usd": 4.0,
        },
    )

    assert len(prompts) == 2
    for invocation in invocations:
        budget_at = invocation["argv"].index("--max-budget-usd")
        assert float(invocation["argv"][budget_at + 1]) == 2.0
        assert 0 < invocation["timeout_s"] <= 900
    assert "direct StructuredOutput object" in prompts[0]
    assert "Do not wrap that object in a text field" in prompts[0]
    assert "Repair category: complete_primary_read" in prompts[1]
    audits = sorted((card_dir / "wma-calls").glob("*/audit.json"))
    assert [json.loads(path.read_text())["status"] for path in audits] == [
        "validation_error",
        "success",
    ]
    successful_request = json.loads((Path(brief.audit["path"]).parent / "request.json").read_text())
    assert successful_request["validation_attempt"] == 2
    assert successful_request["repair_code"] == "complete_primary_read"
    assert successful_request["logical_max_budget_usd"] == 4.0
    assert successful_request["attempt_max_budget_usd"] == 2.0


@pytest.mark.parametrize(
    ("expected_stage", "first_claim", "first_observation"),
    [
        (
            "citations",
            "Prior measurement is 0.4.",
            "Prior measurement is 0.4, not 0.5.",
        ),
        (
            "grounding",
            "Prior measurement is 0.4, not 0.5.",
            "Prior measurement is 0.4.",
        ),
    ],
)
def test_bounded_validation_retry_targets_unsupported_numbers_by_stage(
    tmp_path: Path,
    monkeypatch,
    expected_stage: str,
    first_claim: str,
    first_observation: str,
) -> None:
    _source, root = _seed(tmp_path)
    session = tmp_path / f"number-retry-{expected_stage}"
    card_dir = session / "wm" / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    card = _current_card(session)
    (card_dir / "card.yaml").write_text(yaml.safe_dump(card))
    cited = root / "corpus" / "train" / "r-train000" / "exp-01.yaml"
    prompts: list[str] = []

    def fake_runner(**kwargs) -> int:
        prompts.append(kwargs["prompt"])
        first_attempt = len(prompts) == 1
        response = {
            "claims": [
                {
                    "text": first_claim if first_attempt else "Prior measurement is 0.4.",
                    "citation_ids": ["C1"],
                }
            ],
            "citations": [
                {
                    "id": "C1",
                    "path": str(cited),
                    "locator": "result.measurements[0].value",
                    "observation": (
                        first_observation
                        if first_attempt
                        else "Prior measurement is 0.4."
                    ),
                }
            ],
            "objections": [],
        }
        rows = [
            _init_row(),
            *_complete_read_tool_rows(root / "corpus" / "train", cited),
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "modelUsage": {"claude-opus-5": {"provider": "vertex"}},
                "structured_output": response,
            },
        ]
        kwargs["stdout_path"].write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        kwargs["stderr_path"].write_text("")
        return 0

    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    memory = Memory(
        root,
        session=f"number-retry-{expected_stage}",
        arm="llm",
        readonly=True,
        visible_sides=("train",),
    )
    brief = LLMAgent(session_dir=session, process_runner=fake_runner).on_proposal(
        card,
        [{"check": "example", "passed": True, "detail": "ok"}],
        memory,
        {
            "wma_model": "claude-opus-5",
            "wma_provider": "vertex",
            "wma_validation_attempts": 2,
            "wma_max_budget_usd": 4.0,
        },
    )

    assert len(prompts) == 2
    assert "Repair category: citation_grounding" in prompts[1]
    assert "add an exact locator that contains it or remove the number" in prompts[1]
    audits = sorted((card_dir / "wma-calls").glob("*/audit.json"))
    failed_audit = json.loads(audits[0].read_text())
    assert failed_audit["status"] == "validation_error"
    assert failed_audit["validation_stage"] == expected_stage
    assert "adds numbers absent" in failed_audit["error"]
    successful_request = json.loads(
        (Path(brief.audit["path"]).parent / "request.json").read_text()
    )
    assert successful_request["validation_attempt"] == 2
    assert successful_request["repair_code"] == "citation_grounding"


@pytest.mark.parametrize(
    "failure,match",
    [
        ("nonzero", "Claude exited"),
        ("provider", "provider is not exactly Vertex"),
        ("model", "model does not exactly match"),
        ("tool_policy", "disallowed tools"),
        ("unexpected", "invalid WMA output"),
    ],
)
def test_nonrepairable_wma_failures_are_not_retried(
    tmp_path: Path, monkeypatch, failure: str, match: str
) -> None:
    _source, root = _seed(tmp_path)
    session = tmp_path / f"fatal-{failure}"
    card_dir = session / "wm" / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    card = _current_card(session)
    (card_dir / "card.yaml").write_text(yaml.safe_dump(card))
    cited = root / "corpus" / "train" / "r-train000" / "exp-01.yaml"
    calls = 0

    def fake_runner(**kwargs) -> int:
        nonlocal calls
        calls += 1
        kwargs["stderr_path"].write_text("")
        if failure == "nonzero":
            kwargs["stdout_path"].write_text("")
            return 1
        response = {
            "claims": [{"text": "Prior measurement is 0.4.", "citation_ids": ["C1"]}],
            "citations": [
                {
                    "id": "C1",
                    "path": str(cited),
                    "locator": "result.measurements[0].value",
                    "observation": "measurement value is 0.4",
                }
            ],
            "objections": [],
        }
        actual_model = "claude-opus-4-6" if failure == "model" else "claude-opus-5"
        provider = "anthropic" if failure == "provider" else "vertex"
        extra_rows = []
        if failure == "tool_policy":
            extra_rows = [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "bad", "name": "Read", "input": {}}
                        ]
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "bad",
                                "is_error": False,
                                "content": "forbidden",
                            }
                        ]
                    },
                },
            ]
        rows = [
            _init_row(model=actual_model),
            *_complete_read_tool_rows(root / "corpus" / "train", cited),
            *extra_rows,
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "modelUsage": {actual_model: {"provider": provider}},
                "structured_output": response,
            },
        ]
        kwargs["stdout_path"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return 0

    if failure == "unexpected":
        def fail_unexpected(*_args, **_kwargs):
            raise RuntimeError("validator bug")

        monkeypatch.setattr(llm_module, "_validate_citations", fail_unexpected)
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    memory = Memory(
        root,
        session=f"fatal-{failure}",
        arm="llm",
        readonly=True,
        visible_sides=("train",),
    )
    with pytest.raises(WMError, match=match):
        LLMAgent(session_dir=session, process_runner=fake_runner).on_proposal(
            card,
            [{"check": "example", "passed": True, "detail": "ok"}],
            memory,
            {
                "wma_model": "claude-opus-5",
                "wma_provider": "vertex",
                "wma_validation_attempts": 3,
            },
        )
    assert calls == 1
    assert len(list((card_dir / "wma-calls").glob("*/audit.json"))) == 1


def test_llm_init_and_hidden_side_fail_closed(tmp_path: Path, monkeypatch) -> None:
    _source, root = _seed(tmp_path, include_test=True)
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    monkeypatch.delenv("AWM_WMA_MODEL", raising=False)

    with pytest.raises(WMError, match="read-only"):
        Session.init(
            tmp_path / "writable",
            arm="llm",
            memory_root=str(root),
            memory_readonly=False,
            wma_model="claude-opus-5",
        )
    with pytest.raises(WMError, match="requires --wma-model"):
        Session.init(
            tmp_path / "model-less",
            arm="llm",
            memory_root=str(root),
            memory_readonly=True,
        )
    with pytest.raises(WMError, match="explicit version"):
        Session.init(
            tmp_path / "moving-alias",
            arm="llm",
            memory_root=str(root),
            memory_readonly=True,
            wma_model="opus",
        )
    with pytest.raises(WMError, match="wma_validation_attempts"):
        Session.init(
            tmp_path / "invalid-attempts",
            arm="llm",
            memory_root=str(root),
            memory_readonly=True,
            wma_model="claude-opus-5",
            wma_validation_attempts=0,
        )
    session = Session.init(
        tmp_path / "valid",
        arm="llm",
        memory_root=str(root),
        memory_readonly=True,
        memory_sides=["train"],
        wma_model="claude-opus-5",
    )
    assert session.agent.session_dir == (tmp_path / "valid").resolve()
    assert session.config["wma_model"] == "claude-opus-5"

    args = build_parser().parse_args(
        [
            "wm",
            "init",
            "--arm",
            "llm",
            "--wma-model",
            "claude-opus-5",
            "--wma-validation-attempts",
            "3",
        ]
    )
    assert args.wma_model == "claude-opus-5"
    assert args.wma_validation_attempts == 3


def test_card_corpus_integrity_failure_blocks_init(tmp_path: Path, monkeypatch) -> None:
    _source, root = _seed(tmp_path)
    (root / "corpus" / "train" / "r-train000" / "exp-01.yaml").unlink()
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    with pytest.raises(WMError, match="inventory mismatch|unexpected directory"):
        Session.init(
            tmp_path / "corrupt",
            arm="llm",
            memory_root=str(root),
            memory_readonly=True,
            wma_model="claude-opus-5",
        )

    _source2, root2 = _seed(tmp_path / "extra-yml")
    extra = root2 / "corpus" / "train" / "r-train000" / "exp-unattested.yml"
    extra.write_text(yaml.safe_dump(_historical_card("unattested")))
    with pytest.raises(WMError, match="unexpected file"):
        Session.init(
            tmp_path / "extra-yml-session",
            arm="llm",
            memory_root=str(root2),
            memory_readonly=True,
            wma_model="claude-opus-5",
        )


def test_raw_corpus_is_full_tool_source_with_grounded_read(tmp_path: Path, monkeypatch) -> None:
    raw, traces = _raw_bundle(tmp_path, include_test=True)
    session = tmp_path / "raw-session"
    card_dir = session / "wm" / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    card = _current_card(session)
    (card_dir / "card.yaml").write_text(yaml.safe_dump(card))
    captured: dict = {}

    def fake_runner(**kwargs) -> int:
        captured.update(kwargs)
        response = {
            "claims": [{"text": "The prior trace launched training.", "citation_ids": ["C1"]}],
            "citations": [
                {
                    "id": "C1",
                    "path": str(traces[1]),
                    "locator": "line 2",
                    "observation": "the trace says launch training",
                }
            ],
            "objections": [],
        }
        rows = [
            _init_row(),
            *_read_tool_rows(raw, traces[1], tool_id="raw-read"),
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "modelUsage": {"claude-opus-5": {"provider": "vertex"}},
                "structured_output": response,
            },
        ]
        kwargs["stdout_path"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        kwargs["stderr_path"].write_text("")
        return 0

    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    memory = Memory(
        tmp_path / "empty-memory",
        session="heldout",
        arm="llm",
        split_side="test",
        readonly=True,
        visible_sides=("train", "test"),
    )
    agent = LLMAgent(session_dir=session, process_runner=fake_runner)
    brief = agent.on_proposal(
        card,
        [{"check": "example", "passed": True, "detail": "ok"}],
        memory,
        {
            "wma_model": "claude-opus-5",
            "wma_provider": "vertex",
            "wma_corpus_kind": "raw",
            "wma_corpus_root": str(raw),
        },
    )
    assert brief.evidence[0]["path"] == str(traces[1])
    request = json.loads((Path(brief.audit["path"]).parent / "request.json").read_text())
    assert request["corpus_kind"] == "raw"
    assert request["corpus_metadata"]["run_count"] == 2
    assert "--add-dir" not in captured["argv"]


@pytest.mark.parametrize("failed_read,cite_unread", [(True, False), (False, True)])
def test_failed_or_unread_historical_citation_fails_closed(
    tmp_path: Path, monkeypatch, failed_read: bool, cite_unread: bool
) -> None:
    raw, traces = _raw_bundle(tmp_path, include_test=True)
    session = tmp_path / "bad-read-session"
    card_dir = session / "wm" / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    card = _current_card(session)
    (card_dir / "card.yaml").write_text(yaml.safe_dump(card))
    read_path = traces[0]
    cited_path = traces[1] if cite_unread else read_path

    def fake_runner(**kwargs) -> int:
        response = {
            "claims": [{"text": "Historical claim.", "citation_ids": ["C1"]}],
            "citations": [
                {
                    "id": "C1",
                    "path": str(cited_path),
                    "locator": "line 1",
                    "observation": "claimed evidence",
                }
            ],
            "objections": [],
        }
        read_rows = _read_tool_rows(raw, read_path)
        if failed_read:
            read_rows[1]["message"]["content"][0].update(
                {"is_error": True, "content": "read failed"}
            )
        rows = [
            _init_row(),
            *read_rows,
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": response,
            },
        ]
        kwargs["stdout_path"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        kwargs["stderr_path"].write_text("")
        return 0

    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")
    memory = Memory(
        tmp_path / "empty-memory",
        session="heldout",
        arm="llm",
        readonly=True,
        visible_sides=("train", "test"),
    )
    agent = LLMAgent(session_dir=session, process_runner=fake_runner)
    match = "successfully Read" if failed_read else "without a successful corpus read"
    with pytest.raises(WMError, match=match):
        agent.on_proposal(
            card,
            [{"check": "example", "passed": True, "detail": "ok"}],
            memory,
            {
                "wma_model": "claude-opus-5",
                "wma_provider": "vertex",
                "wma_corpus_kind": "raw",
                "wma_corpus_root": str(raw),
            },
        )
    audit = next((card_dir / "wma-calls").glob("*/audit.json"))
    assert json.loads(audit.read_text())["status"] == "validation_error"


def test_vertex_subprocess_env_is_not_nested_or_oauth() -> None:
    filtered = _vertex_subprocess_env(
        {
            "PATH": "/bin",
            "HOME": "/tmp/home",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": "project",
            "ANTHROPIC_VERTEX_REGION": "us-east5",
            "VERTEX_REGION_CLAUDE_4_8_OPUS": "us-east5",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/adc.json",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "pinned-opus",
            "CLAUDECODE": "nested",
            "AWM_SESSION_DIR": "/scientist",
            "CLAUDE_CODE_OAUTH_TOKEN": "secret",
            "ANTHROPIC_API_KEY": "secret",
            "GOOGLE_UNRELATED_TOKEN": "secret",
            "VERTEX_UNRELATED_TOKEN": "secret",
            "NO_PROXY": "internal.example",
            "UNRELATED_SECRET": "secret",
        }
    )
    assert filtered["ANTHROPIC_VERTEX_PROJECT_ID"] == "project"
    assert filtered["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/adc.json"
    assert filtered["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "pinned-opus"
    assert filtered["ANTHROPIC_VERTEX_REGION"] == "us-east5"
    assert filtered["VERTEX_REGION_CLAUDE_4_8_OPUS"] == "us-east5"
    for host in ("127.0.0.1", "localhost", "::1", "metadata.google.internal", "169.254.169.254"):
        assert host in filtered["NO_PROXY"].split(",")
        assert host in filtered["no_proxy"].split(",")
    assert not (
        {
            "CLAUDECODE",
            "AWM_SESSION_DIR",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "GOOGLE_UNRELATED_TOKEN",
            "VERTEX_UNRELATED_TOKEN",
            "UNRELATED_SECRET",
        }
        & set(filtered)
    )


def test_cli_tool_inventory_and_tool_paths_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(WMError, match="did not expose required tools"):
        _validate_reported_tools([_init_row(ALLOWED_TOOLS[:-1])])
    with pytest.raises(WMError, match="outside the fixed policy"):
        _validate_reported_tools([_init_row((*ALLOWED_TOOLS, "Read"))])

    raw, traces = _raw_bundle(tmp_path)
    call_dir = tmp_path / "call"
    scratch = call_dir / "scratch"
    scratch.mkdir(parents=True)
    (call_dir / "input.json").write_text("{}\n")
    rel = traces[0].relative_to(raw).as_posix()
    stream_rows = _read_tool_rows(raw, traces[0], tool_id="read")
    content_text = stream_rows[1]["message"]["content"][0]["content"][0]["text"]
    good = [
        {
            "event": "tool_use",
            "id": "read",
            "name": "mcp__awm_scratch__read_corpus",
            "input": {
                "root": 0,
                "path": rel,
                "offset": 0,
                "limit": scratch_server.MAX_READ_BYTES,
            },
        },
        {
            "event": "tool_result",
            "tool_use_id": "read",
            "is_error": False,
            "content_text": content_text,
        },
    ]
    assert _validate_tool_trace(good, "raw", [raw.resolve()], scratch) == {
        traces[0]: [(0, traces[0].stat().st_size)]
    }
    default_root = json.loads(json.dumps(good))
    del default_root[0]["input"]["root"]
    assert _validate_tool_trace(default_root, "raw", [raw.resolve()], scratch) == {
        traces[0]: [(0, traces[0].stat().st_size)]
    }
    rejected_schema_mistake = good + [
        {
            "event": "tool_use",
            "id": "bad-list",
            "name": "mcp__awm_scratch__list_corpus",
            "input": {"text": "mistaken structured output"},
        },
        {
            "event": "tool_result",
            "tool_use_id": "bad-list",
            "is_error": True,
            "content_text": "glob must be non-empty",
        },
    ]
    assert _validate_tool_trace(rejected_schema_mistake, "raw", [raw.resolve()], scratch) == {
        traces[0]: [(0, traces[0].stat().st_size)]
    }
    rejected_write_schema = good + [
        {
            "event": "tool_use",
            "id": "bad-write",
            "name": "mcp__awm_scratch__write_file",
            "input": {"text": "mistaken structured output", "citation_ids": ["C1"]},
        },
        {
            "event": "tool_result",
            "tool_use_id": "bad-write",
            "is_error": True,
            "content_text": "content must be a string",
        },
    ]
    assert _validate_tool_trace(rejected_write_schema, "raw", [raw.resolve()], scratch) == {
        traces[0]: [(0, traces[0].stat().st_size)]
    }
    with_structured_output = good + [
        {
            "event": "tool_use",
            "id": "schema",
            "name": "StructuredOutput",
            "input": {"claims": [], "citations": [], "objections": []},
        },
        {
            "event": "tool_result",
            "tool_use_id": "schema",
            "is_error": False,
            "content_text": "Structured output provided successfully",
        },
    ]
    assert _validate_tool_trace(with_structured_output, "raw", [raw.resolve()], scratch) == {
        traces[0]: [(0, traces[0].stat().st_size)]
    }

    escaping = good + [
        {
            "event": "tool_use",
            "id": "write",
            "name": "mcp__awm_scratch__write_file",
            "input": {"path": "../outside.py", "content": "print('bad')"},
        },
        {"event": "tool_result", "tool_use_id": "write", "is_error": True},
    ]
    with pytest.raises(WMError, match="attempted to escape"):
        _validate_tool_trace(escaping, "raw", [raw.resolve()], scratch)
    rejected_escape = good + [
        {
            "event": "tool_use",
            "id": "bad-read",
            "name": "mcp__awm_scratch__read_corpus",
            "input": {"path": "../scientist-secret"},
        },
        {
            "event": "tool_result",
            "tool_use_id": "bad-read",
            "is_error": True,
            "content_text": "path must stay inside its declared root",
        },
    ]
    with pytest.raises(WMError, match="attempted to escape"):
        _validate_tool_trace(rejected_escape, "raw", [raw.resolve()], scratch)


def test_partial_reads_bogus_locators_and_ungrounded_observations_fail_closed(
    tmp_path: Path,
) -> None:
    raw, traces = _raw_bundle(tmp_path)
    call_dir = tmp_path / "partial-call"
    scratch = call_dir / "scratch"
    scratch.mkdir(parents=True)
    (call_dir / "input.json").write_text("{}\n")

    one_byte_events = _extract_tool_events(_read_tool_rows(raw, traces[0], limit=1))
    with pytest.raises(WMError, match="solve_out.txt trajectory"):
        _validate_tool_trace(one_byte_events, "raw", [raw.resolve()], scratch)

    first_line_end = len(traces[0].read_bytes().splitlines(keepends=True)[0])
    only_first_line = {traces[0]: [(0, first_line_end)]}
    base = {
        "claims": [{"text": "claim", "citation_ids": ["C1"]}],
        "citations": [
            {
                "id": "C1",
                "path": str(traces[0]),
                "locator": "line 2",
                "observation": "launch training",
            }
        ],
        "objections": [],
    }
    with pytest.raises(WMError, match="not covered"):
        _validate_citations(base, "raw", [raw.resolve()], call_dir, only_first_line)

    whole = {traces[0]: [(0, traces[0].stat().st_size)]}
    bogus = json.loads(json.dumps(base))
    bogus["citations"][0]["locator"] = "line 999"
    with pytest.raises(WMError, match="outside"):
        _validate_citations(bogus, "raw", [raw.resolve()], call_dir, whole)

    unrelated = json.loads(json.dumps(base))
    unrelated["citations"][0]["observation"] = "unrelated optimizer collapse 9.9"
    with pytest.raises(WMError, match="no checkable token|adds numbers"):
        _validate_citations(unrelated, "raw", [raw.resolve()], call_dir, whole)


def test_claim_and_objection_prose_cannot_add_uncited_values() -> None:
    response = {
        "claims": [
            {
                "text": "Qwen/Qwen3-4B measured 0.4 on r-train000.",
                "citation_ids": ["C1"],
            }
        ],
        "citations": [{"id": "C1"}],
        "objections": [
            {
                "field": "setup.method",
                "severity": "advisory",
                "fix": "Keep Qwen/Qwen3-4B because its cited value is 0.4.",
                "citation_ids": ["C1"],
            }
        ],
    }
    material = {
        "C1": "path: r-train000/exp-01.yaml\n"
        "locator: result.measurements[0]\n"
        '{"model": "Qwen/Qwen3-4B", "value": 0.4}'
    }
    _validate_grounding_references(response, material)

    invented_number = json.loads(json.dumps(response))
    invented_number["claims"][0]["text"] = "The cited run achieved 99% accuracy."
    with pytest.raises(WMError, match="adds numbers"):
        _validate_grounding_references(invented_number, material)

    invented_identifier = json.loads(json.dumps(response))
    invented_identifier["objections"][0]["fix"] = "Switch to model-X9."
    with pytest.raises(WMError, match="adds checkable identifiers"):
        _validate_grounding_references(invented_identifier, material)


def test_identifier_grounding_prompt_and_repair_cover_task_context_names() -> None:
    system = llm_module._system_prompt("cards", [Path("/historical/cards")])
    assert "benchmark, dataset, model, run, experiment, or path" in system
    assert "current task context" in system
    assert "benchmark and base model" in system

    code, guidance = _validation_repair_guidance(
        "WMA claims[0] adds checkable identifiers absent from its cited locators: "
        "['gsm8k', 'google/gemma-3-4b-pt']",
        "cards",
        validation_stage="grounding",
    )
    assert code == "citation_grounding"
    assert "benchmark, dataset, model, run, experiment, and path" in guidance
    assert "Task context is not historical evidence" in guidance
    assert "add the exact field or line containing it, or remove it" in guidance


def test_actual_model_and_server_audit_must_match(tmp_path: Path) -> None:
    with pytest.raises(WMError, match="does not exactly match"):
        _validate_reported_models(
            [_init_row(model="claude-opus-4-8")],
            {"modelUsage": {"claude-opus-4-8": {}}},
            "claude-opus-5",
        )

    events = [
        {
            "event": "tool_use",
            "id": "t1",
            "name": "mcp__awm_scratch__list_corpus",
            "input": {"root": 0, "glob": "**/*"},
        },
        {"event": "tool_result", "tool_use_id": "t1", "is_error": False, "content_text": "{}"},
    ]
    audit = tmp_path / "scratch-tools.jsonl"
    audit.write_text(
        json.dumps(
            {
                "tool": "list_corpus",
                "arguments": {"root": 0, "glob": "**/*"},
                "result": {"isError": False, "content": [{"type": "text", "text": "tampered"}]},
            }
        )
        + "\n"
    )
    with pytest.raises(WMError, match="audit/CLI trace mismatch"):
        _validate_server_tool_audit(audit, events)


def test_server_audit_reconciliation_is_order_independent_and_counts_duplicates(
    tmp_path: Path,
) -> None:
    first_arguments = {"root": 0, "glob": "**/*"}
    first_result = {
        "isError": False,
        "content": [{"type": "text", "text": '{"files": ["one"]}'}],
    }
    second_arguments = {"root": 0, "glob": "**/*.txt", "pattern": "needle"}
    second_result = {
        "isError": False,
        "content": [{"type": "text", "text": '{"matches": []}'}],
    }
    audit = tmp_path / "scratch-tools.jsonl"
    audit.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "tool": "search_corpus",
                    "arguments": second_arguments,
                    "result": second_result,
                },
                {
                    "tool": "list_corpus",
                    "arguments": first_arguments,
                    "result": first_result,
                },
                {
                    "tool": "list_corpus",
                    "arguments": first_arguments,
                    "result": first_result,
                },
            )
        )
        + "\n"
    )
    events = []
    for tool_id, name, arguments, result in (
        ("list-1", "list_corpus", first_arguments, first_result),
        ("list-2", "list_corpus", first_arguments, first_result),
        ("search-1", "search_corpus", second_arguments, second_result),
    ):
        events.extend(
            [
                {
                    "event": "tool_use",
                    "id": tool_id,
                    "name": f"mcp__awm_scratch__{name}",
                    "input": arguments,
                },
                {
                    "event": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": result["isError"],
                    "content_text": result["content"][0]["text"],
                },
            ]
        )

    _validate_server_tool_audit(audit, events)

    events[-1]["content_text"] = "different but private result"
    with pytest.raises(WMError, match="multiset mismatch") as caught:
        _validate_server_tool_audit(audit, events)
    assert "different but private result" not in str(caught.value)

    events[-1]["content_text"] = second_result["content"][0]["text"]
    events[2]["id"] = "list-1"
    with pytest.raises(WMError, match="correlation is not one-to-one"):
        _validate_server_tool_audit(audit, events)


def test_concurrent_calls_reconcile_when_arrival_order_is_reversed(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "evidence.txt").write_text("needle\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    audit = tmp_path / "scratch-tools.jsonl"
    logical_calls = {
        "first": ("list_corpus", {"glob": "**/*"}),
        "second": ("search_corpus", {"glob": "**/*", "pattern": "needle"}),
    }
    responses: dict[str, dict] = {}
    errors: list[BaseException] = []
    release_first = threading.Event()

    def invoke(label: str) -> None:
        try:
            if label == "first" and not release_first.wait(timeout=5):
                raise TimeoutError("second request did not complete")
            name, arguments = logical_calls[label]
            responses[label] = call_tool(
                name,
                arguments,
                scratch=scratch,
                roots=[corpus],
                audit_path=audit,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            if label == "second":
                release_first.set()

    first = threading.Thread(target=invoke, args=("first",))
    second = threading.Thread(target=invoke, args=("second",))
    first.start()
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert not errors

    server_calls = [
        row
        for row in (json.loads(line) for line in audit.read_text().splitlines())
        if row.get("tool")
    ]
    assert [row["tool"] for row in server_calls] == ["search_corpus", "list_corpus"]

    events = []
    for label in ("first", "second"):
        name, arguments = logical_calls[label]
        result = responses[label]
        events.extend(
            [
                {
                    "event": "tool_use",
                    "id": label,
                    "name": f"mcp__awm_scratch__{name}",
                    "input": arguments,
                },
                {
                    "event": "tool_result",
                    "tool_use_id": label,
                    "is_error": result["isError"],
                    "content_text": result["content"][0]["text"],
                },
            ]
        )
    _validate_server_tool_audit(audit, events)


def test_complete_card_read_is_exact_bounded_and_trace_validated(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    card_dir = corpus / "r-production"
    card_dir.mkdir(parents=True)
    card = card_dir / "exp-01.yaml"
    card.write_text(yaml.safe_dump({"problem": {"statement": "x" * 10_000}}))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    audit = tmp_path / "tools.jsonl"
    arguments = {"root": 0, "path": "r-production/exp-01.yaml"}
    result = call_tool(
        "read_corpus_complete",
        arguments,
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["content"].encode() == card.read_bytes()
    assert payload["bytes"] == card.stat().st_size
    assert payload["offset"] == 0 and payload["next_offset"] is None

    events = [
        {
            "event": "tool_use",
            "id": "complete",
            "name": "mcp__awm_scratch__read_corpus_complete",
            "input": arguments,
        },
        {
            "event": "tool_result",
            "tool_use_id": "complete",
            "is_error": False,
            "content_text": result["content"][0]["text"],
        },
    ]
    _validate_server_tool_audit(audit, events)
    successful_reads = _validate_tool_trace(events, "cards", [corpus.resolve()], scratch)
    assert successful_reads == {
        card.resolve(): [(0, card.stat().st_size)]
    }

    # A corpus mutation after tool-trace reconciliation must not make citation
    # validation allocate or parse an unbounded replacement file.
    card.write_bytes(b"x" * (llm_module.MAX_STRUCTURED_CITATION_BYTES + 1))
    with pytest.raises(WMError, match="exceeds citation source cap"):
        _validate_citations(
            {
                "citations": [
                    {
                        "id": "C1",
                        "path": str(card),
                        "locator": "problem.statement",
                        "observation": "statement x",
                    }
                ]
            },
            "cards",
            [corpus.resolve()],
            tmp_path / "current-card",
            successful_reads,
        )

    hostile = card_dir / "exp-02.yaml"
    hostile.write_text("payload: \"" + ("\\\\" * 13_000) + "\"\n")
    oversized = call_tool(
        "read_corpus_complete",
        {"path": "r-production/exp-02.yaml"},
        scratch=scratch,
        roots=[corpus],
    )
    assert oversized["isError"] is True
    assert "use paged read_corpus" in oversized["content"][0]["text"]
    page = call_tool(
        "read_corpus",
        {"path": "r-production/exp-02.yaml", "offset": 0, "limit": 512},
        scratch=scratch,
        roots=[corpus],
    )
    assert page["isError"] is False
    assert json.loads(page["content"][0]["text"])["bytes"] == 512

    # The complete-read transport cap is smaller than the independently
    # bounded structured-citation cap. Exact pages through EOF can therefore
    # support a citation, while partial page coverage remains rejected.
    paged_events: list[dict] = []
    offset = 0
    page_number = 0
    while True:
        page_number += 1
        arguments = {
            "path": "r-production/exp-02.yaml",
            "offset": offset,
            "limit": scratch_server.MAX_READ_BYTES,
        }
        result = call_tool(
            "read_corpus",
            arguments,
            scratch=scratch,
            roots=[corpus],
        )
        assert result["isError"] is False
        tool_id = f"page-{page_number}"
        paged_events.extend(
            [
                {
                    "event": "tool_use",
                    "id": tool_id,
                    "name": "mcp__awm_scratch__read_corpus",
                    "input": arguments,
                },
                {
                    "event": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": False,
                    "content_text": result["content"][0]["text"],
                },
            ]
        )
        payload = json.loads(result["content"][0]["text"])
        if payload["next_offset"] is None:
            break
        offset = payload["next_offset"]

    paged_reads = _validate_tool_trace(
        paged_events, "cards", [corpus.resolve()], scratch
    )
    response = {
        "citations": [
            {
                "id": "C1",
                "path": str(hostile),
                "locator": "payload",
                "observation": "payload field",
            }
        ]
    }
    evidence, _material = _validate_citations(
        response,
        "cards",
        [corpus.resolve()],
        tmp_path / "current-card",
        paged_reads,
    )
    assert evidence[0]["locator"] == "payload"
    with pytest.raises(WMError, match="not covered"):
        _validate_citations(
            response,
            "cards",
            [corpus.resolve()],
            tmp_path / "current-card",
            {hostile.resolve(): paged_reads[hostile.resolve()][:-1]},
        )


def test_exact_line_read_is_audited_byte_covering_and_citation_covering(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "raw-corpus"
    run = corpus / "agent-train" / "gsm8k-run"
    run.mkdir(parents=True)
    trace = run / "solve_out.txt"
    lines = [
        f"event {index:03d} π exact historical evidence {'x' * 48}\n"
        for index in range(1, 91)
    ]
    trace.write_text("".join(lines))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    audit = tmp_path / "line-tools.jsonl"
    relative = trace.relative_to(corpus).as_posix()
    primary_arguments = {
        "root": 0,
        "path": relative,
        "start_line": 1,
        "end_line": 55,
    }
    primary_result = call_tool(
        "read_corpus_lines",
        primary_arguments,
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert primary_result["isError"] is False
    primary_payload = json.loads(primary_result["content"][0]["text"])
    primary_end = sum(len(line.encode()) for line in lines[:55])
    assert primary_payload == {
        "root": 0,
        "path": relative,
        "start_line": 1,
        "end_line": 55,
        "line_count": 55,
        "offset": 0,
        "end_offset": primary_end,
        "bytes": primary_end,
        "scanned_bytes": primary_end,
        "content": "".join(lines[:55]),
    }
    assert primary_payload["bytes"] >= 4096
    assert len(
        json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": primary_result}, sort_keys=True
        ).encode()
    ) <= scratch_server.MAX_MCP_RESPONSE_BYTES

    citation_arguments = {
        "root": 0,
        "path": relative,
        "start_line": 60,
        "end_line": 61,
    }
    citation_result = call_tool(
        "read_corpus_lines",
        citation_arguments,
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert citation_result["isError"] is False
    citation_payload = json.loads(citation_result["content"][0]["text"])
    line_60_start = sum(len(line.encode()) for line in lines[:59])
    line_61_end = sum(len(line.encode()) for line in lines[:61])
    assert citation_payload["offset"] == line_60_start
    assert citation_payload["end_offset"] == line_61_end
    assert citation_payload["bytes"] == line_61_end - line_60_start
    assert citation_payload["content"] == "".join(lines[59:61])

    events = [
        {
            "event": "tool_use",
            "id": "primary-lines",
            "name": "mcp__awm_scratch__read_corpus_lines",
            "input": primary_arguments,
        },
        {
            "event": "tool_result",
            "tool_use_id": "primary-lines",
            "is_error": False,
            "content_text": primary_result["content"][0]["text"],
        },
        {
            "event": "tool_use",
            "id": "citation-lines",
            "name": "mcp__awm_scratch__read_corpus_lines",
            "input": citation_arguments,
        },
        {
            "event": "tool_result",
            "tool_use_id": "citation-lines",
            "is_error": False,
            "content_text": citation_result["content"][0]["text"],
        },
    ]
    _validate_server_tool_audit(audit, events)
    successful_reads = _validate_tool_trace(events, "raw", [corpus.resolve()], scratch)
    assert successful_reads == {
        trace.resolve(): [(0, primary_end), (line_60_start, line_61_end)]
    }

    response = {
        "claims": [{"text": "Historical evidence was recorded.", "citation_ids": ["C1"]}],
        "citations": [
            {
                "id": "C1",
                "path": str(trace),
                "locator": "lines 60-61",
                "observation": "exact historical evidence was recorded",
            }
        ],
        "objections": [],
    }
    with pytest.raises(WMError, match="not covered"):
        _validate_citations(
            response,
            "raw",
            [corpus.resolve()],
            tmp_path / "current-card",
            {trace.resolve(): [(0, primary_end)]},
        )
    evidence, _material = _validate_citations(
        response,
        "raw",
        [corpus.resolve()],
        tmp_path / "current-card",
        successful_reads,
    )
    assert evidence[0]["locator"] == "lines 60-61"

    short_arguments = {**primary_arguments, "end_line": 20}
    short_result = call_tool(
        "read_corpus_lines",
        short_arguments,
        scratch=scratch,
        roots=[corpus],
    )
    short_events = [
        {
            "event": "tool_use",
            "id": "short-lines",
            "name": "mcp__awm_scratch__read_corpus_lines",
            "input": short_arguments,
        },
        {
            "event": "tool_result",
            "tool_use_id": "short-lines",
            "is_error": False,
            "content_text": short_result["content"][0]["text"],
        },
    ]
    with pytest.raises(WMError, match="solve_out.txt trajectory"):
        _validate_tool_trace(
            [*short_events, *events[2:]], "raw", [corpus.resolve()], scratch
        )

    forged = json.loads(citation_result["content"][0]["text"])
    forged["offset"] += 1
    forged_events = json.loads(json.dumps(events))
    forged_events[3]["content_text"] = json.dumps(forged, sort_keys=True)
    with pytest.raises(WMError, match="does not match the requested file lines"):
        _validate_tool_trace(forged_events, "raw", [corpus.resolve()], scratch)

    malformed_trace = json.loads(json.dumps(events))
    malformed_trace[2]["input"]["start_line"] = True
    with pytest.raises(WMError, match="range must contain"):
        _validate_tool_trace(malformed_trace, "raw", [corpus.resolve()], scratch)


def test_exact_line_read_rejects_malformed_ranges_and_all_resource_caps(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    trace = corpus / "trace.txt"
    trace.write_text("".join(f"line {index}\n" for index in range(1, 601)))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    malformed = [
        {"path": "trace.txt", "start_line": 1},
        {"path": "trace.txt", "start_line": True, "end_line": 1},
        {"path": "trace.txt", "start_line": 2, "end_line": 1},
        {"path": "trace.txt", "start_line": 1, "end_line": 501},
        {"path": "trace.txt", "start_line": 600, "end_line": 601},
        {"path": "trace.txt", "start_line": 1, "end_line": 1, "offset": 0},
    ]
    for arguments in malformed:
        result = call_tool(
            "read_corpus_lines",
            arguments,
            scratch=scratch,
            roots=[corpus],
        )
        assert result["isError"] is True
        assert len(
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}, sort_keys=True).encode()
        ) <= scratch_server.MAX_MCP_RESPONSE_BYTES

    material_cap = corpus / "material-cap.txt"
    material_cap.write_bytes(b"x" * (scratch_server.MAX_LINE_READ_MATERIAL_BYTES + 1) + b"\n")
    material_result = call_tool(
        "read_corpus_lines",
        {"path": material_cap.name, "start_line": 1, "end_line": 1},
        scratch=scratch,
        roots=[corpus],
    )
    assert material_result["isError"] is True
    assert "material cap" in material_result["content"][0]["text"]

    response_cap = corpus / "response-cap.txt"
    response_cap.write_text(('\\"' * 5_000) + "\n")
    response_result = call_tool(
        "read_corpus_lines",
        {"path": response_cap.name, "start_line": 1, "end_line": 1},
        scratch=scratch,
        roots=[corpus],
    )
    assert response_result["isError"] is True
    assert "response budget" in response_result["content"][0]["text"]

    line_cap = corpus / "line-cap.txt"
    line_cap.write_bytes(b"x" * (scratch_server.MAX_LINE_READ_LINE_BYTES + 1))
    line_result = call_tool(
        "read_corpus_lines",
        {"path": line_cap.name, "start_line": 1, "end_line": 1},
        scratch=scratch,
        roots=[corpus],
    )
    assert line_result["isError"] is True
    assert "line over" in line_result["content"][0]["text"]

    scan_cap = corpus / "scan-cap.txt"
    with scan_cap.open("wb") as file:
        file.seek(scratch_server.MAX_LINE_READ_SCAN_BYTES)
        file.write(b"x")
    scan_result = call_tool(
        "read_corpus_lines",
        {"path": scan_cap.name, "start_line": 1, "end_line": 1},
        scratch=scratch,
        roots=[corpus],
    )
    assert scan_result["isError"] is True
    assert "scan bytes" in scan_result["content"][0]["text"]


def test_historical_text_locator_streams_with_line_and_byte_caps(tmp_path: Path) -> None:
    trace = tmp_path / "solve_out.txt"
    trace.write_bytes(b"first fact\nsecond fact\n" + b"tail\n" * 1_000_000)
    material, start, end = llm_module._resolve_locator(
        trace,
        "lines 1-2",
        bounded_historical_source=True,
    )
    assert material == "first fact\nsecond fact\n"
    assert (start, end) == (0, len(material.encode()))

    with pytest.raises(WMError, match="over-broad"):
        llm_module._resolve_locator(
            trace,
            "lines 1-501",
            bounded_historical_source=True,
        )

    huge_line = tmp_path / "huge-line.txt"
    huge_line.write_bytes(b"x" * (llm_module.MAX_LINE_LOCATOR_MATERIAL_BYTES + 1))
    with pytest.raises(WMError, match="line exceeds locator byte cap"):
        llm_module._resolve_locator(
            huge_line,
            "line 1",
            bounded_historical_source=True,
        )


def test_locator_schema_rejects_prose_annotations() -> None:
    pattern = _response_schema("brief")["properties"]["citations"]["items"][
        "properties"
    ]["locator"]["pattern"]
    assert re.fullmatch(pattern, "problem.statement; hypothesis.claim")
    assert re.fullmatch(pattern, "card.setup.command.argv[2]")
    assert re.fullmatch(pattern, "lines 12-18")
    assert not re.fullmatch(pattern, "card.setup.command.argv (--max-steps 1")
    assert not re.fullmatch(pattern, "problem.statement = arithmetic errors")

    raw_code, raw_guidance = _validation_repair_guidance(
        "WMA citation C1 locator was not covered", "raw"
    )
    assert raw_code == "complete_raw_range"
    assert "solve_out.txt" in raw_guidance
    assert "exp-*.yaml" not in raw_guidance


def test_oversized_scratch_results_are_paged_bounded_and_exactly_audited(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    listed_root = corpus / "listed"
    listed_root.mkdir(parents=True)
    expected_files = []
    for index in range(1_000):
        path = listed_root / f"trajectory-{index:04d}-with-a-descriptive-name.txt"
        path.write_text("x")
        expected_files.append(path.relative_to(corpus).as_posix())

    search_path = corpus / "large-search.txt"
    search_path.write_text("".join(f"needle {index:03d} {'x' * 3_000}\n" for index in range(50)))
    read_path = corpus / "large-read.bin"
    read_path.write_bytes(b"\x00" * 10_000)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    audit = tmp_path / "scratch-tools.jsonl"
    calls: list[tuple[str, dict, dict]] = []

    def checked_call(name: str, arguments: dict) -> dict:
        result = call_tool(
            name,
            arguments,
            scratch=scratch,
            roots=[corpus],
            audit_path=audit,
        )
        response = {"jsonrpc": "2.0", "id": len(calls) + 1, "result": result}
        assert len(json.dumps(response, sort_keys=True).encode()) <= (
            scratch_server.MAX_MCP_RESPONSE_BYTES
        )
        calls.append((name, arguments, result))
        return json.loads(result["content"][0]["text"])

    listed: list[str] = []
    offset = 0
    while True:
        arguments = {"glob": "listed/*.txt", "offset": offset, "limit": 10_000}
        payload = checked_call("list_corpus", arguments)
        listed.extend(payload["files"])
        if payload["next_offset"] is None:
            assert not payload["truncated"]
            break
        assert payload["truncated"]
        assert payload["next_offset"] > offset
        offset = payload["next_offset"]
    assert listed == expected_files

    found_lines: list[int] = []
    cursor = None
    while True:
        arguments = {
            "glob": "large-search.txt",
            "pattern": "needle",
            "limit": scratch_server.MAX_SEARCH_MATCHES,
        }
        if cursor is not None:
            arguments["cursor"] = cursor
        payload = checked_call("search_corpus", arguments)
        found_lines.extend(match["line"] for match in payload["matches"])
        cursor = payload["next_cursor"]
        if cursor is None:
            assert not payload["truncated"]
            break
        assert payload["truncated"]
    assert found_lines == list(range(1, 51))

    total_read = 0
    offset = 0
    while True:
        arguments = {
            "path": read_path.relative_to(corpus).as_posix(),
            "offset": offset,
            "limit": scratch_server.MAX_READ_BYTES,
        }
        payload = checked_call("read_corpus", arguments)
        total_read += payload["bytes"]
        if payload["next_offset"] is None:
            break
        assert payload["next_offset"] > offset
        offset = payload["next_offset"]
    assert total_read == read_path.stat().st_size

    class FakePopen:
        returncode = 0
        pid = 12345

        def __init__(self, _command, *, stdout, stderr, start_new_session) -> None:
            assert start_new_session
            stdout.write(b"\x00" * 100_000)
            stderr.write(b"\xff" * 100_000)

        def wait(self, timeout=None) -> int:
            assert timeout is not None
            return self.returncode

    monkeypatch.setattr(scratch_server.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(scratch_server.subprocess, "Popen", FakePopen)
    run_payload = checked_call("run", {"argv": ["python3", "tool.py"], "timeout_s": 5})
    assert run_payload["stdout_truncated"]
    assert run_payload["stderr_truncated"]
    assert run_payload["stdout_bytes"] == 100_000
    assert run_payload["stderr_bytes"] == 100_000
    assert run_payload["truncation_guidance"]

    missing_arguments = {"glob": "*"}
    missing_result = call_tool(
        "list_corpus",
        missing_arguments,
        scratch=None,
        roots=[corpus],
        audit_path=audit,
    )
    assert missing_result["isError"]
    assert (
        len(
            json.dumps(
                {"jsonrpc": "2.0", "id": len(calls) + 1, "result": missing_result},
                sort_keys=True,
            ).encode()
        )
        <= scratch_server.MAX_MCP_RESPONSE_BYTES
    )
    calls.append(("list_corpus", missing_arguments, missing_result))

    unknown_audit = tmp_path / "unknown-tool.jsonl"
    unknown_result = call_tool(
        "unknown-" + "x" * 20_000,
        {},
        scratch=scratch,
        roots=[corpus],
        audit_path=unknown_audit,
    )
    assert unknown_result["isError"]
    assert (
        len(
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "result": unknown_result}, sort_keys=True
            ).encode()
        )
        <= scratch_server.MAX_MCP_RESPONSE_BYTES
    )
    unknown_row = json.loads(unknown_audit.read_text())
    assert unknown_row["result"] == unknown_result

    events: list[dict] = []
    for index, (name, arguments, result) in enumerate(calls, 1):
        tool_id = f"tool-{index}"
        events.extend(
            [
                {
                    "event": "tool_use",
                    "id": tool_id,
                    "name": f"mcp__awm_scratch__{name}",
                    "input": arguments,
                },
                {
                    "event": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": bool(result.get("isError")),
                    "content_text": result["content"][0]["text"],
                },
            ]
        )
    _validate_server_tool_audit(audit, events)


def test_scratch_tools_are_auditable_and_execution_is_confined(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    evidence = corpus / "evidence.txt"
    evidence.write_text("alpha\nbeta historical fact\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    audit = tmp_path / "tools.jsonl"

    listed = call_tool(
        "list_corpus",
        {"glob": "**/*"},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert not listed["isError"] and "evidence.txt" in listed["content"][0]["text"]
    searched = call_tool(
        "search_corpus",
        {"root": 0, "glob": "**/*", "pattern": "historical"},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert not searched["isError"] and "beta historical fact" in searched["content"][0]["text"]
    assert not call_tool(
        "write_file",
        {"path": "tool.py", "content": "print('ok')\n"},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )["isError"]
    assert call_tool(
        "write_file",
        {"path": "../escape", "content": "bad"},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )["isError"]

    probe_sandbox(scratch, [corpus])
    run = call_tool(
        "run",
        {
            "argv": [
                "python3",
                "-c",
                "import pathlib; print(pathlib.Path('/corpus/0/evidence.txt').read_text()); "
                "print(pathlib.Path('/home').exists())",
            ]
        },
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert not run["isError"]
    assert "historical fact" in run["content"][0]["text"]
    assert "False" in run["content"][0]["text"]
    forbidden = call_tool(
        "run",
        {
            "argv": [
                "python3",
                "-c",
                "import pathlib; pathlib.Path('/corpus/0/forbidden').write_text('x')",
            ]
        },
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert forbidden["isError"] and not (corpus / "forbidden").exists()
    audit_rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert {row.get("tool") for row in audit_rows} >= {
        "list_corpus",
        "search_corpus",
        "write_file",
        "run",
    }

    init = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert init["result"]["serverInfo"]["name"] == "awm_scratch"
    assert init["result"]["protocolVersion"] == "2025-06-18"
    malformed = handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": []},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert malformed["error"]["code"] == -32602
    for invalid_version in (None, "1.0", 2.0):
        invalid = handle_message(
            {
                "jsonrpc": invalid_version,
                "id": "must-not-be-reflected",
                "method": "ping",
            },
            scratch=scratch,
            roots=[corpus],
            audit_path=audit,
        )
        assert invalid == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "request must declare jsonrpc 2.0"},
        }


def test_directory_jail_mounts_clone_and_lock_the_complete_tree(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    calls: list[list[str]] = []
    verified: list[Path] = []

    def fake_run(argv, *, check):
        assert check is True
        calls.append(argv)

    monkeypatch.setattr(scratch_server.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scratch_server, "_make_mount_tree_readonly", lambda path: verified.append(path)
    )

    scratch_server._mount(source, target, readonly=True)

    assert calls == [
        ["mount", "--rbind", str(source), str(target)],
    ]
    assert verified == [target]


def test_recursive_readonly_fallback_is_still_verified(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    calls: list[list[str]] = []
    verified: list[Path] = []

    def unavailable(_target):
        raise OSError("mount_setattr unavailable")

    def fake_run(argv, *, check):
        assert check is True
        calls.append(argv)

    monkeypatch.setattr(scratch_server, "_mount_setattr_readonly_recursive", unavailable)
    monkeypatch.setattr(scratch_server.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scratch_server,
        "_require_readonly_mount_tree",
        lambda path: verified.append(path),
    )

    scratch_server._make_mount_tree_readonly(target)

    assert calls == [
        ["mount", "-R", "-o", "remount,ro,bind", str(target)],
    ]
    assert verified == [target]


def test_readonly_mount_verification_checks_nested_and_escaped_mounts(
    tmp_path: Path,
) -> None:
    target = tmp_path / "jail root"
    target.mkdir()
    child = target / "usr" / "injected library"
    child.mkdir(parents=True)

    def escaped(path: Path) -> str:
        return str(path).replace("\\", r"\134").replace(" ", r"\040")

    root_row = f"10 1 0:1 / {escaped(target)} ro,nosuid - tmpfs tmpfs rw"
    child_ro = f"11 10 0:2 / {escaped(child)} ro,nodev - tmpfs tmpfs rw"
    scratch_server._require_readonly_mount_tree(target, mountinfo=f"{root_row}\n{child_ro}\n")

    child_rw = f"11 10 0:2 / {escaped(child)} rw,nodev - tmpfs tmpfs rw"
    with pytest.raises(RuntimeError, match=r"writable mount.*injected library"):
        scratch_server._require_readonly_mount_tree(target, mountinfo=f"{root_row}\n{child_rw}\n")

    with pytest.raises(RuntimeError, match="absent from mountinfo"):
        scratch_server._require_readonly_mount_tree(target, mountinfo=f"{child_ro}\n")


def test_scratch_usage_caps_entries_and_depth(tmp_path: Path, monkeypatch) -> None:
    scratch = tmp_path / "scratch-limits"
    scratch.mkdir()
    for name in ("one", "two", "three"):
        (scratch / name).touch()
    monkeypatch.setattr(scratch_server, "MAX_SCRATCH_ENTRIES", 2)
    with pytest.raises(ValueError, match="filesystem entries"):
        scratch_server._scratch_usage(scratch)

    monkeypatch.setattr(scratch_server, "MAX_SCRATCH_ENTRIES", 100)
    monkeypatch.setattr(scratch_server, "MAX_SCRATCH_DEPTH", 1)
    (scratch / "level-one" / "level-two").mkdir(parents=True)
    with pytest.raises(ValueError, match="directory depth"):
        scratch_server._scratch_usage(scratch)


def test_raw_manifest_and_every_exposed_file_are_fail_closed(tmp_path: Path) -> None:
    raw, _traces = _raw_bundle(tmp_path)
    assert _validate_raw_corpus(raw, ("train",))["run_count"] == 1

    metrics = next(raw.glob("*/*/metrics.json"))
    outside = tmp_path / "outside.json"
    outside.write_text(metrics.read_text())
    metrics.unlink()
    metrics.symlink_to(outside)
    with pytest.raises(WMError, match="lacks metrics|symlink|manifest"):
        _validate_raw_corpus(raw, ("train",))

    raw2, _ = _raw_bundle(tmp_path / "optional")
    next(raw2.glob("*/*/solve_out.txt")).with_name("solve_parsed.txt").write_text("unpinned\n")
    with pytest.raises(WMError, match="unexpected file"):
        _validate_raw_corpus(raw2, ("train",))

    raw3, _ = _raw_bundle(tmp_path / "mutated-metadata")
    index = raw3 / "index.jsonl"
    row = json.loads(index.read_text())
    row["accuracy"] = 0.99
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(WMError, match="not the deterministic view"):
        _validate_raw_corpus(raw3, ("train",))


def test_sandbox_self_test_failure_blocks_real_llm_init(tmp_path: Path, monkeypatch) -> None:
    _source, root = _seed(tmp_path)
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "vertex-project")

    def fail_probe(*_args, **_kwargs):
        raise RuntimeError("unsafe namespace")

    monkeypatch.setattr("awm.wm.scratch_server.probe_sandbox", fail_probe)
    with pytest.raises(WMError, match="cannot prove.*corpus-read-only"):
        Session.init(
            tmp_path / "unsafe",
            arm="llm",
            memory_root=str(root),
            memory_readonly=True,
            wma_model="claude-opus-5",
        )


@pytest.mark.skipif(
    os.environ.get("AWM_RUN_VERTEX_WMA_PROBE") != "1",
    reason="explicit opt-in: makes one real Vertex sidecar call",
)
def test_vertex_cli_loopback_mcp_end_to_end(tmp_path: Path) -> None:
    model = os.environ.get("AWM_WMA_MODEL")
    if not model:
        pytest.fail("set an explicit AWM_WMA_MODEL for the Vertex compatibility probe")
    _source, root = _seed(tmp_path)
    session = tmp_path / "vertex-session"
    card_dir = session / "wm" / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    card = _current_card(session)
    (card_dir / "card.yaml").write_text(yaml.safe_dump(card))
    memory = Memory(
        root,
        session="vertex-probe",
        arm="llm",
        readonly=True,
        visible_sides=("train",),
    )
    brief = LLMAgent(session_dir=session).on_proposal(
        card,
        [{"check": "probe", "passed": True, "detail": "ok"}],
        memory,
        {
            "wma_model": model,
            "wma_provider": "vertex",
            "wma_max_budget_usd": 0.75,
            "wma_timeout_s": 540,
            "wma_validation_attempts": 3,
            "wma_effort": "low",
        },
    )
    assert brief.produced_by == "llm"
    assert set(ALLOWED_TOOLS) <= set(brief.audit["reported_tools"])
    assert brief.evidence
