"""No model calls: exercise the evidence gateway's actual security boundary."""

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.outcome_prediction import wm_evidence as gateway
from tools.outcome_prediction.wm_evidence import (
    EvidenceError,
    EvidenceStore,
    MCPServer,
    UnknownTool,
)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    files = {"a.txt": "α🙂\r\nhello .* hello\n", "nested/b.txt": "banana hello"}
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content.encode())
    (root / "not-allowed.txt").write_text("SECRET")
    return root, {name: digest(content) for name, content in files.items()}, files


@pytest.fixture
def store(corpus, tmp_path):
    root, files, _ = corpus
    with EvidenceStore(root, files, audit_log=tmp_path / "audit.jsonl") as value:
        yield value


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "..",
        "/etc/passwd",
        "../secret",
        "a/../secret",
        "a/./b",
        "a//b",
        "a/",
        "a\\b",
        "C:/secret",
        "a\x00b",
        "a\nb",
        "a\tb",
        "a\x7fb",
    ],
)
def test_allowlist_rejects_unsafe_paths(corpus, tmp_path, path):
    with pytest.raises(EvidenceError, match="logical"):
        EvidenceStore(corpus[0], {path: digest("")}, audit_log=tmp_path / "audit")


@pytest.mark.parametrize("sha", ["", "0" * 63, "x" * 64, 123, None])
def test_allowlist_rejects_bad_hash(corpus, tmp_path, sha):
    with pytest.raises(EvidenceError, match="SHA256"):
        EvidenceStore(corpus[0], {"a.txt": sha}, audit_log=tmp_path / "audit")


def test_allowlist_copied_and_unlisted_never_enumerated(corpus, store):
    corpus[1]["not-allowed.txt"] = digest("SECRET")
    paths = [item["path"] for item in store.handle_call("list_evidence")["files"]]
    assert paths == ["a.txt", "nested/b.txt"]
    for path in ["not-allowed.txt", "does-not-exist.txt"]:
        with pytest.raises(EvidenceError, match="not allowlisted"):
            store.handle_call("read_evidence", {"path": path})
    with pytest.raises(TypeError):
        store.allowlist["x"] = digest("")


def test_root_symlink_rejected(corpus, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(corpus[0], target_is_directory=True)
    with pytest.raises(EvidenceError, match="root.*symlink"):
        EvidenceStore(alias, corpus[1], audit_log=tmp_path / "audit")


@pytest.mark.parametrize("inside", [False, True])
def test_leaf_symlink_rejected_even_with_matching_hash(corpus, store, tmp_path, inside):
    target = corpus[0] / "a.txt"
    outside = (corpus[0] if inside else tmp_path) / "linked-content.txt"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(EvidenceError, match="symlink"):
        store.handle_call("read_evidence", {"path": "a.txt"})


def test_directory_replaced_by_symlink_after_initialization(corpus, store, tmp_path):
    directory = corpus[0] / "nested"
    moved = tmp_path / "moved"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(EvidenceError, match="symlink"):
        store.handle_call("read_evidence", {"path": "nested/b.txt"})


def test_root_fd_remains_pinned_if_root_path_replaced(corpus, store, tmp_path):
    root = corpus[0]
    root.rename(tmp_path / "original-root")
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    (attacker / "a.txt").write_text("untrusted replacement")
    root.symlink_to(attacker, target_is_directory=True)
    assert store.handle_call("read_evidence", {"path": "a.txt"})["text"] == corpus[2]["a.txt"]


def test_hash_reverified_after_success_and_same_size_tampering(corpus, store):
    assert store.handle_call("read_evidence", {"path": "a.txt"})["text"]
    target = corpus[0] / "a.txt"
    target.write_bytes(target.read_bytes().replace(b"hello", b"HELLO"))
    for name, arguments in [
        ("read_evidence", {"path": "a.txt"}),
        ("search_evidence", {"query": "HELLO"}),
    ]:
        with pytest.raises(EvidenceError, match="integrity"):
            store.handle_call(name, arguments)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_non_regular_evidence_never_read(corpus, store, kind):
    target = corpus[0] / "a.txt"
    target.unlink()
    if kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    with pytest.raises(EvidenceError, match="regular"):
        store.handle_call("read_evidence", {"path": "a.txt"})


def test_binary_text_fails_without_lossy_replacement(corpus, tmp_path):
    (corpus[0] / "bad.txt").write_bytes(b"\xff")
    with (
        EvidenceStore(corpus[0], {"bad.txt": digest(b"\xff")}, audit_log=tmp_path / "audit") as s,
        pytest.raises(EvidenceError, match="UTF-8"),
    ):
        s.handle_call("read_evidence", {"path": "bad.txt"})


def test_audit_log_outside_evidence_and_not_symlink_or_hardlink(corpus, tmp_path):
    with pytest.raises(EvidenceError, match="outside"):
        EvidenceStore(corpus[0], corpus[1], audit_log=corpus[0] / "audit")
    alias = tmp_path / "audit-alias"
    alias.symlink_to(corpus[0] / "a.txt")
    with pytest.raises(EvidenceError, match="symlink"):
        EvidenceStore(corpus[0], corpus[1], audit_log=alias)
    hard = tmp_path / "audit-hard"
    os.link(corpus[0] / "a.txt", hard)
    with pytest.raises(EvidenceError, match="hardlinked"):
        EvidenceStore(corpus[0], corpus[1], audit_log=hard)
    assert (corpus[0] / "a.txt").read_bytes().decode() == corpus[2]["a.txt"]


def test_audit_records_success_failure_without_content_or_query(store, tmp_path):
    store.handle_call("read_evidence", {"path": "a.txt"})
    store.handle_call("search_evidence", {"query": "private query value"})
    with pytest.raises(UnknownTool):
        store.handle_call("/private/filesystem", {"token": "private token value"})
    text = (tmp_path / "audit.jsonl").read_text()
    assert all(secret not in text for secret in ["private", "hello", "a.txt", "α"])
    records = [json.loads(line) for line in text.splitlines()]
    assert [r["status"] for r in records] == ["ok", "ok", "error"]
    assert [r["tool"] for r in records] == ["read_evidence", "search_evidence", "[unknown]"]
    assert all(len(r["arguments_sha256"]) == 64 for r in records)


def test_failed_audit_withholds_tool_result(store):
    os.close(store._audit_fd)
    store._audit_fd = -1
    with pytest.raises(EvidenceError, match="Audit logging failed"):
        store.handle_call("list_evidence")


def test_list_pagination_full_access(store):
    page = store.handle_call("list_evidence", {"limit": 1})
    assert page["files"][0]["path"] == "a.txt"
    assert page["next_offset"] == 1
    page2 = store.handle_call("list_evidence", {"offset": page["next_offset"], "limit": 1})
    assert page2["files"][0]["path"] == "nested/b.txt"
    assert page2["next_offset"] is None
    assert page2["total_files"] == 2
    assert store.handle_call("list_evidence", {"offset": 2})["files"] == []


def test_read_character_pagination_preserves_unicode_crlf_and_complete_text(store, corpus):
    offset, pages = 0, []
    while offset is not None:
        result = store.handle_call("read_evidence", {"path": "a.txt", "offset": offset, "count": 3})
        assert len(result["text"]) <= 3
        assert result["offset"] == offset
        pages.append(result["text"])
        offset = result["next_offset"]
    assert "".join(pages) == corpus[2]["a.txt"]
    assert result["total_chars"] == len(corpus[2]["a.txt"])
    end = store.handle_call("read_evidence", {"path": "a.txt", "offset": result["total_chars"]})
    assert end["text"] == "" and end["next_offset"] is None


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_evidence", {"path": "a.txt", "count": 16001}),
        ("read_evidence", {"path": "a.txt", "offset": -1}),
        ("read_evidence", {"path": "a.txt", "offset": True}),
        ("read_evidence", {"path": "a.txt", "offset": 999}),
        ("read_evidence", {"path": "a.txt", "count": 0}),
        ("list_evidence", {"limit": 101}),
        ("list_evidence", {"offset": 0.0}),
        ("search_evidence", {"query": ""}),
        ("search_evidence", {"query": "a" * 513}),
        ("search_evidence", {"query": "a", "context_chars": 201}),
        ("search_evidence", {"query": "a", "limit": 51}),
        ("read_evidence", {"path": "a.txt", "command": "ls"}),
        ("read_evidence", {}),
    ],
)
def test_invalid_tool_arguments_rejected(store, name, args):
    with pytest.raises(EvidenceError):
        store.handle_call(name, args)


def test_literal_search_and_overlapping_matches(store):
    matches = store.handle_call("search_evidence", {"query": ".*"})["matches"]
    assert len(matches) == 1
    assert matches[0]["path"] == "a.txt"
    assert ".*" in matches[0]["snippet"]
    matches = store.handle_call("search_evidence", {"query": "ana", "context_chars": 0})["matches"]
    assert [(m["offset"], m["snippet"]) for m in matches] == [(1, "ana"), (3, "ana")]
    assert store.handle_call("search_evidence", {"query": "SECRET"})["matches"] == []


def test_search_pagination_has_no_lost_or_repeated_matches(store):
    cursor, matches = None, []
    for _ in range(10):
        result = store.handle_call(
            "search_evidence", {"query": "hello", "limit": 1, "cursor": cursor}
        )
        assert len(result["matches"]) <= 1
        matches.extend(result["matches"])
        cursor = result["next_cursor"]
        if cursor is None:
            break
    assert cursor is None
    assert matches == store.handle_call("search_evidence", {"query": "hello"})["matches"]
    assert len(matches) == 3


def test_search_resumes_empty_pages_and_finds_boundary_spanning_query(store, monkeypatch):
    monkeypatch.setattr(gateway, "SEARCH_SCAN_CHARS", 5)
    cursor, matches = None, []
    empty_with_cursor = False
    for _ in range(30):
        page = store.handle_call("search_evidence", {"query": "hello", "cursor": cursor})
        empty_with_cursor |= not page["matches"] and page["next_cursor"] is not None
        matches.extend((m["path"], m["offset"]) for m in page["matches"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert cursor is None and empty_with_cursor
    assert matches == [("a.txt", 4), ("a.txt", 13), ("nested/b.txt", 7)]


def test_search_file_limit_preserves_access(store, monkeypatch):
    monkeypatch.setattr(gateway, "SEARCH_SCAN_FILES", 1)
    page = store.handle_call("search_evidence", {"query": "banana"})
    assert not page["matches"] and page["next_cursor"] is not None
    page2 = store.handle_call("search_evidence", {"query": "banana", "cursor": page["next_cursor"]})
    assert page2["matches"][0]["path"] == "nested/b.txt"
    assert page2["next_cursor"] is None


def test_search_cursor_bound_to_query_and_manifest(store):
    cursor = store.handle_call("search_evidence", {"query": "hello", "limit": 1})["next_cursor"]
    with pytest.raises(EvidenceError, match="cursor"):
        store.handle_call("search_evidence", {"query": "different", "cursor": cursor})
    decoded = json.loads(base64.urlsafe_b64decode(cursor))
    decoded[3] = "0" * 64
    bad = base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode()
    with pytest.raises(EvidenceError, match="cursor"):
        store.handle_call("search_evidence", {"query": "hello", "cursor": bad})
    with pytest.raises(EvidenceError, match="cursor"):
        store.handle_call("search_evidence", {"query": "hello", "cursor": "not-base64"})


def test_baseline_tools_identical_to_wm_evidence_tools(corpus, store, tmp_path):
    baseline = store.tool_specs()
    assert [s["name"] for s in baseline] == ["list_evidence", "read_evidence", "search_evidence"]
    with pytest.raises(UnknownTool):
        store.handle_call("predict_candidate", {"candidate_id": "c1"})
    with EvidenceStore(
        corpus[0],
        corpus[1],
        audit_log=tmp_path / "wm-audit",
        candidates=["c1"],
        predict=lambda _: {"predicted_official_accuracy": 0.4},
    ) as wm:
        assert wm.tool_specs()[:3] == baseline
        assert wm.tool_specs()[3]["name"] == "predict_candidate"
        assert wm.tool_specs()[3]["inputSchema"]["properties"]["candidate_id"]["enum"] == ["c1"]


def test_callback_receives_only_registered_id(corpus, tmp_path):
    calls = []

    def predict(candidate_id):
        calls.append(candidate_id)
        return {"predicted_official_accuracy": 0.7, "diagnostics": {"method": "train-only"}}

    with EvidenceStore(
        corpus[0],
        corpus[1],
        audit_log=tmp_path / "audit",
        candidates={"c1": {"not": "passed to callback"}},
        predict=predict,
    ) as wm:
        assert (
            wm.handle_call("predict_candidate", {"candidate_id": "c1"})[
                "predicted_official_accuracy"
            ]
            == 0.7
        )
        for candidate in ["c2", "../secrets", "__import__('os')", None, {"id": "c1"}]:
            with pytest.raises(EvidenceError, match="not registered"):
                wm.handle_call("predict_candidate", {"candidate_id": candidate})
        with pytest.raises(EvidenceError, match="unrecognized"):
            wm.handle_call("predict_candidate", {"candidate_id": "c1", "query": "arbitrary"})
    assert calls == ["c1"]


@pytest.mark.parametrize("result", [None, "answer", {"x": float("nan")}, {"x": "a" * 32001}])
def test_callback_invalid_or_unbounded_output_rejected(corpus, tmp_path, result):
    with (
        EvidenceStore(
            corpus[0],
            corpus[1],
            audit_log=tmp_path / "audit",
            candidates=["c"],
            predict=lambda _: result,
        ) as wm,
        pytest.raises(EvidenceError, match="invalid data"),
    ):
        wm.handle_call("predict_candidate", {"candidate_id": "c"})


def test_callback_errors_do_not_reveal_server_secrets(corpus, tmp_path):
    def predict(_):
        raise RuntimeError("token=TOPSECRET filesystem=/private/server")

    with EvidenceStore(
        corpus[0], corpus[1], audit_log=tmp_path / "audit", candidates=["c"], predict=predict
    ) as wm:
        with pytest.raises(EvidenceError) as error:
            wm.handle_call("predict_candidate", {"candidate_id": "c"})
        assert "TOPSECRET" not in str(error.value)
    assert "TOPSECRET" not in (tmp_path / "audit").read_text()


def request(method, params=None, request_id=1):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def initialize(server):
    return server.handle_message(
        request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
            request_id=0,
        )
    )


def test_mcp_lifecycle_and_static_tools(store):
    server = MCPServer(store)
    assert server.handle_message(request("ping", request_id="ping"))["result"] == {}
    assert "error" in server.handle_message(request("tools/list"))
    response = initialize(server)
    assert response["id"] == 0
    assert response["result"]["protocolVersion"] == "2025-11-25"
    assert response["result"]["capabilities"] == {"tools": {}}
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle_message(request("tools/list"))["result"]["tools"] == store.tool_specs()
    assert "error" in initialize(server)
    assert "error" in server.handle_message(request("tools/list", {"cursor": "bogus"}))


def test_mcp_tool_content_escaping_and_error_categories(store):
    server = MCPServer(store)
    initialize(server)
    response = server.handle_message(
        request(
            "tools/call",
            {
                "name": "read_evidence",
                "arguments": {"path": "a.txt"},
            },
        )
    )
    result = response["result"]
    assert not result["isError"]
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    assert "\r\n" in result["structuredContent"]["text"]
    bad_path = server.handle_message(
        request(
            "tools/call",
            {
                "name": "read_evidence",
                "arguments": {"path": "/etc/passwd"},
            },
        )
    )
    assert bad_path["result"]["isError"] is True
    assert "/etc/passwd" not in json.dumps(bad_path)
    unknown = server.handle_message(request("tools/call", {"name": "execute_code"}))
    assert unknown["error"]["code"] == -32602
    malformed = server.handle_message(
        request("tools/call", {"name": "read_evidence", "arguments": []})
    )
    assert malformed["error"]["code"] == -32602


def test_notifications_never_execute_tool_or_generate_response(store, tmp_path):
    server = MCPServer(store)
    initialize(server)
    for method in ["tools/call", "unknown", "notifications/cancelled"]:
        assert (
            server.handle_message(
                {"jsonrpc": "2.0", "method": method, "params": {"name": "list_evidence"}}
            )
            is None
        )
    assert (tmp_path / "audit.jsonl").read_text() == ""


@pytest.mark.parametrize(
    "message",
    [
        None,
        [],
        {},
        {"jsonrpc": "1.0", "method": "ping", "id": 1},
        request("ping", request_id=None),
        request("ping", request_id=True),
    ],
)
def test_malformed_rpc_envelopes_rejected(store, message):
    assert MCPServer(store).handle_message(message)["error"]["code"] == -32600


def test_stdio_newline_delimited_json_and_no_notification_output(store):
    messages = [
        request(
            "initialize",
            {
                "protocolVersion": "unsupported",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
            0,
        ),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        request("tools/call", {"name": "read_evidence", "arguments": {"path": "a.txt"}}, 2),
    ]
    source = io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n")
    sink = io.StringIO()
    gateway.serve_stdio(store, stdin=source, stdout=sink)
    output = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [m["id"] for m in output] == [0, 2]
    assert output[0]["result"]["protocolVersion"] == gateway.PROTOCOL_VERSION
    assert output[1]["result"]["structuredContent"]["text"].startswith("α🙂\r\n")


@pytest.mark.parametrize("raw", ["{not json}\n", '{"x":NaN}\n', '{"x":1,"x":2}\n'])
def test_stdio_invalid_json(store, raw):
    sink = io.StringIO()
    gateway.serve_stdio(store, stdin=io.StringIO(raw), stdout=sink)
    result = json.loads(sink.getvalue())
    assert result["error"]["code"] == -32700
    assert "id" not in result


def test_surrogate_request_id_cannot_crash_utf8_transport(store):
    source = io.StringIO(json.dumps(request("ping", request_id="\ud800")) + "\n")
    raw_sink = io.BytesIO()
    sink = io.TextIOWrapper(raw_sink, encoding="utf-8")
    gateway.serve_stdio(store, stdin=source, stdout=sink)
    assert json.loads(raw_sink.getvalue())["id"] == "\ud800"


def test_surrogate_logical_path_rejected(corpus, tmp_path):
    with pytest.raises(EvidenceError, match="logical"):
        EvidenceStore(corpus[0], {"\ud800.txt": digest("")}, audit_log=tmp_path / "audit")


def test_audit_fifo_fails_without_blocking(corpus, tmp_path):
    audit = tmp_path / "audit-fifo"
    os.mkfifo(audit)
    with pytest.raises(EvidenceError):
        EvidenceStore(corpus[0], corpus[1], audit_log=audit)


def test_stdio_oversized_line_drained_then_next_message_processed(store):
    source = io.StringIO(
        "x" * (gateway.MAX_REQUEST_CHARS + 10) + "\n" + json.dumps(request("ping")) + "\n"
    )
    sink = io.StringIO()
    gateway.serve_stdio(store, stdin=source, stdout=sink)
    outputs = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(outputs) == 2 and outputs[0]["error"]["code"] == -32600
    assert outputs[1]["result"] == {}


def test_baseline_cli_config_relative_paths_and_clean_stdout(corpus, tmp_path):
    config = tmp_path / "server.json"
    config.write_text(
        json.dumps({"evidence_root": "evidence", "files": corpus[1], "audit_log": "audit"})
    )
    result = subprocess.run(
        [sys.executable, "-m", "tools.outcome_prediction.wm_evidence", "--config", str(config)],
        input=json.dumps(request("ping")) + "\n",
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["result"] == {}
    assert (tmp_path / "audit").exists()
    config.write_text(
        json.dumps(
            {
                "evidence_root": "evidence",
                "files": corpus[1],
                "audit_log": "audit",
                "hidden_labels": {"c1": 0.99},
            }
        )
    )
    assert gateway.main(["--config", str(config)]) == 2
