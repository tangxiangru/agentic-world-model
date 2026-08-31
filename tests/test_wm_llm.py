"""The autonomous WMA sees the full copied corpus and fails closed."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

import awm.wm.scratch_server as scratch_server
from awm.cli import build_parser
from awm.wm.agents.llm import (
    ALLOWED_TOOLS,
    LLMAgent,
    _extract_tool_events,
    _validate_server_tool_audit,
    _validate_citations,
    _validate_grounding_references,
    _validate_raw_corpus,
    _validate_reported_models,
    _validate_reported_tools,
    _validate_tool_trace,
    _vertex_subprocess_env,
)
from awm.wm.memory import Memory
from awm.wm.runtime import Session
from awm.wm.schema import WMError
from awm.wm.scratch_server import call_tool, handle_message, probe_sandbox


def _init_row(tools: tuple[str, ...] = ALLOWED_TOOLS, model: str = "claude-opus-5") -> dict:
    return {"type": "system", "subtype": "init", "tools": [*tools],
            "model": model, "apiKeySource": "none"}


def _read_tool_rows(
    root: Path,
    path: Path,
    *,
    tool_id: str = "read-1",
    offset: int = 0,
    limit: int = 200_000,
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
        {"type": "assistant", "message": {"content": [{
            "type": "tool_use",
            "id": tool_id,
            "name": "mcp__awm_scratch__read_corpus",
            "input": {"root": 0, "path": relative, "offset": offset, "limit": limit},
        }]}},
        {"type": "user", "message": {"content": [{
            "type": "tool_result",
            "tool_use_id": tool_id,
            "is_error": False,
            "content": [{"type": "text", "text": payload}],
        }]}},
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
        rows.append({
            "run": rel.as_posix(), "agent_config": f"agent-{side}",
            "run_name": f"gsm8k_model_{i}", "side": side,
            "base_model": "model", "accuracy": accuracy, "time_taken": "01:00:00",
            "has_trace": True, "trace_bytes": trace.stat().st_size,
            "path": f"/home/ben/prior_runs/{rel.as_posix()}",
        })
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
        "# Prior runs", "",
        (f"{len(rows)} previous attempts at this task by autonomous agents, one directory each, "
         "laid out as `<agent config>/<run>/`. Each holds `solve_out.txt` (the agent's complete "
         "session trace), `metrics.json` (official accuracy), and `time_taken.txt`. "
         "No optional run artifacts or `task/` workspace snapshots are exposed."),
        "", "Sorted by official accuracy, best first.", "",
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
    (root / "corpus-manifest.json").write_text(json.dumps({
        "schema_version": "awm-prior-runs-v1",
        "split": {"id": "test/split-v1", "sides": sides},
        "dataset": {"repo": "example/prior-runs", "repo_type": "dataset",
                    "revision": "a" * 40},
        "file_scope": ["solve_out.txt", "metrics.json", "time_taken.txt"],
        "run_count": len(manifest_rows),
        "runs": manifest_rows,
    }, indent=2, sort_keys=True) + "\n")
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
    rows = [json.loads(line) for line in (root / "structured" / "cards.jsonl").read_text().splitlines()]
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
            "claims": [{"text": "A prior SFT card improved over its comparator.",
                        "citation_ids": ["C1", "C2"]}],
            "citations": [
                {"id": "C1", "path": str(cited),
                 "locator": "result.measurements[0]",
                 "observation": "measurement value is 0.4"},
                {"id": "C2", "path": str(cited),
                 "locator": "evaluation.comparator.value",
                 "observation": "comparator is 0.3"},
            ],
            "objections": [],
        }
        rows = [
            _init_row(),
            *_read_tool_rows(root / "corpus" / "train", cited, tool_id="tool-1"),
            {"type": "result", "subtype": "success", "is_error": False,
             "session_id": "wma-session", "total_cost_usd": 0.12,
             "modelUsage": {"claude-opus-5": {"provider": "vertex"}},
             "structured_output": response},
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
        ["wm", "init", "--arm", "llm", "--wma-model", "claude-opus-5"]
    )
    assert args.wma_model == "claude-opus-5"


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
            "citations": [{
                "id": "C1", "path": str(traces[1]), "locator": "line 2",
                "observation": "the trace says launch training",
            }],
            "objections": [],
        }
        rows = [
            _init_row(),
            *_read_tool_rows(raw, traces[1], tool_id="raw-read"),
            {"type": "result", "subtype": "success", "is_error": False,
             "modelUsage": {"claude-opus-5": {"provider": "vertex"}},
             "structured_output": response},
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
            "citations": [{"id": "C1", "path": str(cited_path), "locator": "line 1",
                           "observation": "claimed evidence"}],
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
            {"type": "result", "subtype": "success", "is_error": False,
             "structured_output": response},
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
    match = "successfully Read" if failed_read else "without a successful read_corpus"
    with pytest.raises(WMError, match=match):
        agent.on_proposal(
            card,
            [{"check": "example", "passed": True, "detail": "ok"}],
            memory,
            {"wma_model": "claude-opus-5", "wma_provider": "vertex",
             "wma_corpus_kind": "raw", "wma_corpus_root": str(raw)},
        )
    audit = next((card_dir / "wma-calls").glob("*/audit.json"))
    assert json.loads(audit.read_text())["status"] == "validation_error"


def test_vertex_subprocess_env_is_not_nested_or_oauth() -> None:
    filtered = _vertex_subprocess_env({
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
    })
    assert filtered["ANTHROPIC_VERTEX_PROJECT_ID"] == "project"
    assert filtered["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/adc.json"
    assert filtered["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "pinned-opus"
    assert filtered["ANTHROPIC_VERTEX_REGION"] == "us-east5"
    assert filtered["VERTEX_REGION_CLAUDE_4_8_OPUS"] == "us-east5"
    for host in ("127.0.0.1", "localhost", "::1", "metadata.google.internal",
                 "169.254.169.254"):
        assert host in filtered["NO_PROXY"].split(",")
        assert host in filtered["no_proxy"].split(",")
    assert not ({"CLAUDECODE", "AWM_SESSION_DIR", "CLAUDE_CODE_OAUTH_TOKEN",
                 "ANTHROPIC_API_KEY", "GOOGLE_UNRELATED_TOKEN",
                 "VERTEX_UNRELATED_TOKEN", "UNRELATED_SECRET"} & set(filtered))


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
        {"event": "tool_use", "id": "read", "name": "mcp__awm_scratch__read_corpus",
         "input": {"root": 0, "path": rel, "offset": 0, "limit": 200_000}},
        {"event": "tool_result", "tool_use_id": "read", "is_error": False,
         "content_text": content_text},
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
        {"event": "tool_use", "id": "bad-list",
         "name": "mcp__awm_scratch__list_corpus",
         "input": {"text": "mistaken structured output"}},
        {"event": "tool_result", "tool_use_id": "bad-list", "is_error": True,
         "content_text": "glob must be non-empty"},
    ]
    assert _validate_tool_trace(
        rejected_schema_mistake, "raw", [raw.resolve()], scratch
    ) == {traces[0]: [(0, traces[0].stat().st_size)]}
    rejected_write_schema = good + [
        {"event": "tool_use", "id": "bad-write",
         "name": "mcp__awm_scratch__write_file",
         "input": {"text": "mistaken structured output", "citation_ids": ["C1"]}},
        {"event": "tool_result", "tool_use_id": "bad-write", "is_error": True,
         "content_text": "content must be a string"},
    ]
    assert _validate_tool_trace(
        rejected_write_schema, "raw", [raw.resolve()], scratch
    ) == {traces[0]: [(0, traces[0].stat().st_size)]}
    with_structured_output = good + [
        {"event": "tool_use", "id": "schema", "name": "StructuredOutput",
         "input": {"claims": [], "citations": [], "objections": []}},
        {"event": "tool_result", "tool_use_id": "schema", "is_error": False,
         "content_text": "Structured output provided successfully"},
    ]
    assert _validate_tool_trace(
        with_structured_output, "raw", [raw.resolve()], scratch
    ) == {traces[0]: [(0, traces[0].stat().st_size)]}

    escaping = good + [
        {"event": "tool_use", "id": "write", "name": "mcp__awm_scratch__write_file",
         "input": {"path": "../outside.py", "content": "print('bad')"}},
        {"event": "tool_result", "tool_use_id": "write", "is_error": True},
    ]
    with pytest.raises(WMError, match="attempted to escape"):
        _validate_tool_trace(escaping, "raw", [raw.resolve()], scratch)
    rejected_escape = good + [
        {"event": "tool_use", "id": "bad-read",
         "name": "mcp__awm_scratch__read_corpus",
         "input": {"path": "../scientist-secret"}},
        {"event": "tool_result", "tool_use_id": "bad-read", "is_error": True,
         "content_text": "path must stay inside its declared root"},
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
        "citations": [{"id": "C1", "path": str(traces[0]), "locator": "line 2",
                       "observation": "launch training"}],
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
        "claims": [{
            "text": "Qwen/Qwen3-4B measured 0.4 on r-train000.",
            "citation_ids": ["C1"],
        }],
        "citations": [{"id": "C1"}],
        "objections": [{
            "field": "setup.method",
            "severity": "advisory",
            "fix": "Keep Qwen/Qwen3-4B because its cited value is 0.4.",
            "citation_ids": ["C1"],
        }],
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


def test_actual_model_and_server_audit_must_match(tmp_path: Path) -> None:
    with pytest.raises(WMError, match="does not exactly match"):
        _validate_reported_models(
            [_init_row(model="claude-opus-4-8")],
            {"modelUsage": {"claude-opus-4-8": {}}},
            "claude-opus-5",
        )

    events = [
        {"event": "tool_use", "id": "t1", "name": "mcp__awm_scratch__list_corpus",
         "input": {"root": 0, "glob": "**/*"}},
        {"event": "tool_result", "tool_use_id": "t1", "is_error": False,
         "content_text": "{}"},
    ]
    audit = tmp_path / "scratch-tools.jsonl"
    audit.write_text(json.dumps({
        "tool": "list_corpus", "arguments": {"root": 0, "glob": "**/*"},
        "result": {"isError": False, "content": [{"type": "text", "text": "tampered"}]},
    }) + "\n")
    with pytest.raises(WMError, match="audit/CLI trace mismatch"):
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
        "list_corpus", {"glob": "**/*"}, scratch=scratch,
        roots=[corpus], audit_path=audit,
    )
    assert not listed["isError"] and "evidence.txt" in listed["content"][0]["text"]
    searched = call_tool(
        "search_corpus", {"root": 0, "glob": "**/*", "pattern": "historical"},
        scratch=scratch, roots=[corpus], audit_path=audit,
    )
    assert not searched["isError"] and "beta historical fact" in searched["content"][0]["text"]
    assert not call_tool(
        "write_file", {"path": "tool.py", "content": "print('ok')\n"},
        scratch=scratch, roots=[corpus], audit_path=audit,
    )["isError"]
    assert call_tool(
        "write_file", {"path": "../escape", "content": "bad"},
        scratch=scratch, roots=[corpus], audit_path=audit,
    )["isError"]

    probe_sandbox(scratch, [corpus])
    run = call_tool(
        "run",
        {"argv": ["python3", "-c",
                  "import pathlib; print(pathlib.Path('/corpus/0/evidence.txt').read_text()); "
                  "print(pathlib.Path('/home').exists())"]},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert not run["isError"]
    assert "historical fact" in run["content"][0]["text"]
    assert "False" in run["content"][0]["text"]
    forbidden = call_tool(
        "run",
        {"argv": ["python3", "-c",
                  "import pathlib; pathlib.Path('/corpus/0/forbidden').write_text('x')"]},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert forbidden["isError"] and not (corpus / "forbidden").exists()
    audit_rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert {row.get("tool") for row in audit_rows} >= {
        "list_corpus", "search_corpus", "write_file", "run"
    }

    init = handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18"}},
        scratch=scratch,
        roots=[corpus],
        audit_path=audit,
    )
    assert init["result"]["serverInfo"]["name"] == "awm_scratch"
    assert init["result"]["protocolVersion"] == "2025-06-18"
    malformed = handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": []},
        scratch=scratch, roots=[corpus], audit_path=audit,
    )
    assert malformed["error"]["code"] == -32602


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


def test_sandbox_self_test_failure_blocks_real_llm_init(
    tmp_path: Path, monkeypatch
) -> None:
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
            "wma_max_budget_usd": 0.25,
            "wma_timeout_s": 180,
            "wma_effort": "low",
        },
    )
    assert brief.produced_by == "llm"
    assert set(ALLOWED_TOOLS) <= set(brief.audit["reported_tools"])
    assert brief.evidence
