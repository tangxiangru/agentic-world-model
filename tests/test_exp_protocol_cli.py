"""One card, start to finish, through the CLI: new -> check -> preflight -> lock -> close -> index."""

from __future__ import annotations

import json

import pytest

from awm import paths
from awm.cli import main
from awm.exp_protocol import lineage, schema


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(paths.REPO_ROOT / "skills" / "exp_protocol"))
    (tmp_path / "train.py").write_text("pass\n")
    write_jsonl(tmp_path / "data" / "train.jsonl",
                [{"prompt": "q", "completion": "a #### 1<|im_end|>"} for _ in range(20)])
    (tmp_path / "ckpts").mkdir()
    return tmp_path


def fill_plan(card_path, root):
    card = schema.load_card(card_path)
    card["situation"].update({"elapsed_h": 0.5, "trigger": "base scores 0.33",
                              "alternatives_rejected": [{"option": "dpo", "reason": "no prefs"}]})
    card["problem"]["statement"] = "arithmetic slips"
    card["hypothesis"]["claim"] = "sft on filtered samples cuts slips"
    card["hypothesis"]["falsified_if"] = "watch set fixed < 10%"
    card["setup"]["parent_checkpoint"].update({"path": "google/gemma-3-4b-pt", "origin": "base_model"})
    card["setup"]["data"] = [{"path": str(root / "data" / "train.jsonl"), "source": "synthetic:self",
                              "n_examples": 20}]
    card["setup"]["method"].update({"family": "sft", "stop_token": "<|im_end|>", "answer_marker": "#### ",
                                    "hyperparams": {"max_seq_len": 512}})
    card["setup"]["command"].update({"argv": ["python", "train.py"], "cwd": str(root),
                                     "script": str(root / "train.py")})
    card["setup"]["output_dir"] = str(root / "ckpts" / "exp-01")
    card["setup"]["checkpoints"]["keep"] = "last"
    card["evaluation"]["protocol"]["n"] = 150
    schema.dump_card(card_path, card)


def fill_result(card_path, root):
    card = schema.load_card(card_path)
    (root / "ckpts" / "exp-01" / "final").mkdir(parents=True)
    card["result"] = {"execution": "completed", "output_checkpoint": str(root / "ckpts" / "exp-01" / "final"),
                      "measurements": [{"metric": "accuracy", "value": 0.4, "n": 150, "path": str(root / "e.json")}]}
    card["conclusion"] = {"verdict": "supported", "mechanism_verdict": "not_tested",
                          "summary": "+7 on dev-150", "decision": "adopt"}
    schema.dump_card(card_path, card)


def test_the_whole_protocol_for_one_card(session, capsys) -> None:
    root = session
    d = str(root)
    assert main(["exp_protocol", "new", "--dir", d]) == 0
    card_path = lineage.cards_dir(root) / "exp-01.yaml"
    assert card_path.is_file()
    out = capsys.readouterr().out
    assert "situation.trigger" in out                      # questions printed on new

    assert main(["exp_protocol", "check", "--dir", d, "exp-01"]) == 1   # unfilled: errors
    fill_plan(card_path, root)
    assert main(["exp_protocol", "check", "--dir", d, "exp-01"]) == 0

    assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 0   # runs preflight itself
    assert (lineage.cards_dir(root) / "exp-01.lock.json").is_file()
    assert (lineage.cards_dir(root) / "exp-01.preflight.json").is_file()

    assert main(["exp_protocol", "close", "--dir", d, "exp-01"]) == 1  # no result yet
    fill_result(card_path, root)
    assert main(["exp_protocol", "close", "--dir", d, "exp-01"]) == 0
    index = (root / "memory" / "index.md").read_text()
    assert "| exp-01 |" in index and "adopt" in index

    assert main(["exp_protocol", "chain", "--dir", d, "exp-01"]) == 0
    assert "base_model" in capsys.readouterr().out

    assert main(["exp_protocol", "new", "--dir", d]) == 0                # next id
    assert (lineage.cards_dir(root) / "exp-02.yaml").is_file()

    assert main(["exp_protocol", "collect", d]) == 0
    assert "exp_protocol" not in capsys.readouterr().err


def test_lock_refuses_when_preflight_fails(session, capsys) -> None:
    root = session
    d = str(root)
    main(["exp_protocol", "new", "--dir", d])
    card_path = lineage.cards_dir(root) / "exp-01.yaml"
    fill_plan(card_path, root)
    card = schema.load_card(card_path)
    card["setup"]["data"][0]["n_examples"] = 19   # off by one: data_n_examples_match fails
    schema.dump_card(card_path, card)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 1
    assert not (lineage.cards_dir(root) / "exp-01.lock.json").exists()
    assert "data_n_examples_match" in capsys.readouterr().out


def test_close_refuses_a_plan_edited_after_lock(session, capsys) -> None:
    root = session
    d = str(root)
    main(["exp_protocol", "new", "--dir", d])
    card_path = lineage.cards_dir(root) / "exp-01.yaml"
    fill_plan(card_path, root)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 0
    fill_result(card_path, root)
    card = schema.load_card(card_path)
    card["hypothesis"]["claim"] = "a story that fits the result"
    schema.dump_card(card_path, card)
    assert main(["exp_protocol", "close", "--dir", d, "exp-01"]) == 1
    assert "differ from what was locked" in capsys.readouterr().out


def test_unknown_card_id_is_a_usage_error(session) -> None:
    assert main(["exp_protocol", "check", "--dir", str(session), "exp-99"]) == 2


# ---- review findings (2026-09-01) --------------------------------------------

def test_relocking_needs_a_reason_and_leaves_a_trace(session, capsys) -> None:
    root = session
    d = str(root)
    main(["exp_protocol", "new", "--dir", d])
    card_path = lineage.cards_dir(root) / "exp-01.yaml"
    fill_plan(card_path, root)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 0
    card = schema.load_card(card_path)
    card["hypothesis"]["claim"] = "a story that fits the result"
    schema.dump_card(card_path, card)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 1
    assert "--relock" in capsys.readouterr().out
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01", "--relock", "typo in the claim"]) == 0
    info = json.loads((lineage.cards_dir(root) / "exp-01.lock.json").read_text())
    assert info["relocked_from"][0]["reason"] == "typo in the claim"
    fill_result(card_path, root)
    assert main(["exp_protocol", "close", "--dir", d, "exp-01"]) == 0
    assert "re-locked 1 time" in capsys.readouterr().out


def test_an_override_lets_a_failing_check_through_and_is_recorded(session, capsys) -> None:
    root = session
    d = str(root)
    main(["exp_protocol", "new", "--dir", d])
    card_path = lineage.cards_dir(root) / "exp-01.yaml"
    fill_plan(card_path, root)
    card = schema.load_card(card_path)
    card["setup"]["data"][0]["n_examples"] = 19
    schema.dump_card(card_path, card)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01", "--override", "no_such_check=why"]) == 2
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01",
                 "--override", "data_n_examples_match=one row is a header line"]) == 0
    info = json.loads((lineage.cards_dir(root) / "exp-01.lock.json").read_text())
    assert info["overrides"] == {"data_n_examples_match": "one row is a header line"}
    assert "overridden" in capsys.readouterr().out


# ---- the verdict is part of the lock (2026-09-03) ----

def _locked_card(session):
    d = str(session)
    main(["exp_protocol", "new", "--dir", d])
    card_path = lineage.cards_dir(session) / "exp-01.yaml"
    fill_plan(card_path, session)
    return d, card_path


def test_lock_records_that_no_world_model_agent_was_attached(session, capsys) -> None:
    d, card_path = _locked_card(session)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 0
    info = json.loads((lineage.cards_dir(session) / "exp-01.lock.json").read_text())
    assert info["wma"] == {"state": "not_attached", "waited_s": 0.0, "verdict_path": None, "error": None,
                           "request_id": None, "requested_at": None, "fingerprint": info["wma"]["fingerprint"]}
    assert info["wma"]["fingerprint"]["plan_sha256"] == info["plan_sha256"]
    out = capsys.readouterr().out
    assert "locked exp-01" in out and "verdict" not in out
    # the annotation does not disturb what the lock pins
    from awm.exp_protocol import lock
    assert lock.verify_lock(card_path, schema.load_card(card_path)).ok


def test_lock_waits_for_the_attached_agent_and_records_the_delivered_verdict(session, capsys, monkeypatch) -> None:
    import threading

    from awm import wma_client
    from awm.wma import backends, sidecar
    d, card_path = _locked_card(session)
    (session / ".wma" / "requests").mkdir(parents=True)          # the sidecar opened its queue first
    skill = session / "private-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("private WMA policy\n")
    config = sidecar.Config(session_dir=session, skill_dir=skill, history_dir=None, backend="heuristic",
                            model="claude-opus-5", effort="high", budget=backends.Budget(), jobs=1)
    # the sidecar can only review once the lock file exists: answer shortly after lock writes it
    monkeypatch.setattr(wma_client, "DEFAULT_WAIT_MIN", 0.5)
    original = wma_client.wait_for_verdict
    monkeypatch.setattr(wma_client, "wait_for_verdict",
                        lambda *a, **k: original(*a, **{**k, "poll_s": 0.05, "heartbeat_s": 0.1}))
    answered = threading.Timer(0.3, lambda: sidecar.run(config, once=True))
    answered.start()
    try:
        assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 0
    finally:
        answered.join()
    info = json.loads((lineage.cards_dir(session) / "exp-01.lock.json").read_text())
    assert info["wma"]["state"] == "delivered" and info["wma"]["request_id"]
    assert (lineage.cards_dir(session) / "exp-01.verdict.json").is_file()
    out = capsys.readouterr().out
    assert "waiting up to" in out and "verdict: L0_runs=" in out and "read it before launching" in out
    from awm.exp_protocol import lock
    assert lock.verify_lock(card_path, schema.load_card(card_path)).ok
    assert not (session / "skills" / "wma").exists()


def test_lock_can_skip_the_wait_only_with_a_recorded_reason(session, capsys) -> None:
    d, _ = _locked_card(session)
    (session / ".wma" / "requests").mkdir(parents=True)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01", "--no-wma-wait", "sidecar known dead"]) == 0
    info = json.loads((lineage.cards_dir(session) / "exp-01.lock.json").read_text())
    assert info["wma"]["state"] == "skipped" and info["wma"]["reason"] == "sidecar known dead"
    assert not list((session / ".wma" / "requests").glob("*.json"))
    assert "skipped by request" in capsys.readouterr().out


def test_a_relock_keeps_the_earlier_verdict_wait_in_the_lock_history(session) -> None:
    """Answering a verdict's preconditions means re-locking, and every lock waits again. The cost of
    the gate is only recoverable if the earlier wait survives the new lock (2026-09-03)."""
    from awm.exp_protocol import lock
    d, card_path = _locked_card(session)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01"]) == 0
    lock.annotate_lock(card_path, "wma", {"state": "delivered", "waited_s": 390.0,
                                          "verdict_path": "memory/cards/exp-01.verdict.json",
                                          "error": None, "request_id": "r1", "requested_at": None})
    card = schema.load_card(card_path)
    card["hypothesis"]["claim"] = "the answer the WMA asked us to check first"
    schema.dump_card(card_path, card)
    assert main(["exp_protocol", "lock", "--dir", d, "exp-01", "--relock", "answering the precondition"]) == 0

    info = json.loads((lineage.cards_dir(session) / "exp-01.lock.json").read_text())
    assert info["relocked_from"][0]["wma"]["waited_s"] == 390.0
    assert info["relocked_from"][0]["reason"] == "answering the precondition"
    assert info["wma"]["state"] == "not_attached"          # this lock asked again; no sidecar here
    assert lock.verify_lock(card_path, schema.load_card(card_path)).ok
