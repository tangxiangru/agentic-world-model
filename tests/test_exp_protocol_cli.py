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
