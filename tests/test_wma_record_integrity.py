"""Container identity, crash-safe records and stale-reply regressions."""

from __future__ import annotations

import gzip
import json
import shutil
from types import SimpleNamespace

import pytest

from awm import paths, ptb_ops, wma_client
from awm.exp_protocol import cli, decisions, lock, schema
from awm.exp_protocol.run import _choice
from awm.wma import ledger


def test_record_publication_failure_never_leaves_partial_final_json(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    original = decisions.os.link

    def interrupted(*_args):
        raise OSError("interrupted before publication")

    monkeypatch.setattr(decisions.os, "link", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        decisions.write_once(path, {"state": "complete"})
    assert not path.exists() and not list(tmp_path.glob(".record-*"))
    monkeypatch.setattr(decisions.os, "link", original)
    decisions.write_once(path, {"state": "complete"})
    with pytest.raises(FileExistsError):
        decisions.write_once(path, {"state": "overwritten"})
    assert json.loads(path.read_text()) == {"state": "complete"}


def test_unreadable_old_record_is_retained_without_poisoning_valid_records(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"partial":')
    decisions.write_once(tmp_path / "good.json", {"created_at": "2026-09-04", "action": "proceed"})
    with pytest.warns(RuntimeWarning, match="retained"):
        rows = list(decisions.read_records(tmp_path.glob("*.json")))
    assert len(rows) == 1 and rows[0][1]["action"] == "proceed"
    assert bad.read_text() == '{"partial":'


def test_fingerprint_maps_scientist_paths_into_the_sidecar_mount(tmp_path):
    scientist = tmp_path / "scientist"
    cards = scientist / "memory/cards"
    cards.mkdir(parents=True)
    script = scientist / "train.py"
    script.write_text("print('training')\n")
    card = {"setup": {"command": {"script": "/home/ben/task/train.py", "cwd": "/home/ben/task"}}}
    schema.dump_card(cards / "exp-01.yaml", card)
    (cards / "exp-01.lock.json").write_text(json.dumps({
        "lock_id": "fixed", "plan_sha256": schema.plan_hash(card),
        "script": {"path": "/home/ben/task/train.py", "sha256": schema.sha256_file(script)},
        "data": [], "configs": [],
    }))
    sidecar = tmp_path / "sidecar-session"
    shutil.copytree(scientist, sidecar)
    left = decisions.card_fingerprint(scientist, "exp-01")
    right = decisions.card_fingerprint(sidecar, "exp-01")
    assert left == right
    assert right["files"] == [{"path": "/home/ben/task/train.py", "sha256": schema.sha256_file(script)}]
    (sidecar / "train.py").write_text("changed\n")
    assert decisions.card_fingerprint(sidecar, "exp-01") != left


def test_final_choice_binding_tracks_the_formal_plan_and_script_bytes(tmp_path):
    session = tmp_path / "session"
    cards = session / "memory/cards"
    cards.mkdir(parents=True)
    script = session / "train.py"
    script.write_text("recipe B\n")
    card = {"hypothesis": {"claim": "B"}, "setup": {"command": {"script": str(script), "cwd": str(session)}}}
    schema.dump_card(cards / "exp-01.yaml", card)
    proposal = json.loads((paths.REPO_ROOT / "skills/exp_protocol/proposal.example.json").read_text())
    did = proposal["decision_id"]
    decisions.write_once(decisions.proposal_path(session, did), proposal)
    wma_client.compare_and_wait(session, did, out=lambda _: None)
    wma_client.record_choice(session, did, "B", "B selected", "exp-01")
    assert _choice(session, "exp-01", card)["selected"] == "B"
    script.write_text("different recipe\n")
    with pytest.raises(ValueError, match="choice binding"):
        _choice(session, "exp-01", card)
    wma_client.record_choice(session, did, "B", "explicitly confirm implementation repair", "exp-01")
    assert _choice(session, "exp-01", card)["selected"] == "B"
    card["hypothesis"]["claim"] = "a different intervention"
    with pytest.raises(ValueError, match="choice binding"):
        _choice(session, "exp-01", card)
    schema.dump_card(cards / "exp-01.yaml", card)
    wma_client.record_choice(session, did, "B", "bind changed plan explicitly", "exp-01")
    wma_client.record_choice(session, did, None, "none of these candidates should run")
    with pytest.raises(ValueError, match="superseded or declined"):
        _choice(session, "exp-01", card)


def test_old_review_cannot_overwrite_a_newer_lock_annotation(tmp_path, monkeypatch):
    session = tmp_path / "session"
    path = session / "memory/cards/exp-01.yaml"
    path.parent.mkdir(parents=True)
    card = {"card_id": "exp-01", "setup": {"command": {"cwd": str(session)}}}
    schema.dump_card(path, card)
    original = lock.write_lock(path, card, {})
    old_fp = decisions.card_fingerprint(session, "exp-01")
    monkeypatch.setattr(wma_client, "sidecar_attached", lambda _session: True)

    def newer_review(*_args, **_kwargs):
        newer = lock.write_lock(path, card, {}, relock_reason="new version")
        lock.annotate_lock(path, "wma", {"state": "delivered", "request_id": "new"},
                           expected_lock_id=newer["lock_id"])
        return {"state": "delivered", "request_id": "old", "fingerprint": old_fp}

    monkeypatch.setattr(wma_client, "review_and_wait", newer_review)
    result = cli._wma_gate(SimpleNamespace(dir=str(session)), path, "exp-01", expected_lock_id=original["lock_id"])
    assert result["state"] == "superseded"
    assert lock.read_lock(path)["wma"]["request_id"] == "new"


def test_harvest_retains_nested_review_and_comparison_evidence(tmp_path):
    result, out = tmp_path / "result", tmp_path / "out"
    private = result / "wma_private"
    review = private / "reviews/request-A"
    review.mkdir(parents=True)
    (review / "exp-01.transcript.jsonl").write_text('{"type":"result"}\n')
    (review / "exp-01.result.json").write_text('{"verdict":{}}\n')
    compare = private / "comparisons/invocation"
    compare.mkdir(parents=True)
    (compare / "comparison.json").write_text('{"schema_version":"awm-wma-comparison-v1"}\n')
    out.mkdir()
    status = ptb_ops._copy_sidecar(result, out, [])
    assert status["transcripts"][0]["name"] == "reviews/request-A/exp-01.transcript.jsonl"
    with gzip.open(out / "wma_private/reviews/request-A/exp-01.transcript.jsonl.gz", "rt") as f:
        assert json.loads(f.readline())["type"] == "result"
    assert (out / "wma_private/reviews/request-A/exp-01.result.json").is_file()
    assert (out / "wma_private/comparisons/invocation/comparison.json").is_file()
    assert ledger.rows([out]) == []  # archived revisions never inflate the legacy ledger
