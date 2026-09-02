"""awm wma end to end: a card is locked by the protocol, reviewed, closed, reconciled, counted."""

from __future__ import annotations

import json

import pytest

from awm import paths
from awm.cli import main
from awm.exp_protocol import lineage, schema as cards
from awm.wma import schema


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(paths.REPO_ROOT / "skills" / "exp_protocol"))
    skill = tmp_path / "wma-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: wma\n---\n")
    monkeypatch.setenv("AWM_WMA_SKILL_DIR", str(skill))
    root = tmp_path / "task"
    (root / "ckpts").mkdir(parents=True)
    (root / "train.py").write_text("pass\n")
    write_jsonl(root / "data" / "train.jsonl", [{"prompt": "q", "completion": "a #### 1<|im_end|>"} for _ in range(10)])
    main(["exp_protocol", "new", "--dir", str(root)])
    p = lineage.cards_dir(root) / "exp-01.yaml"
    card = cards.load_card(p)
    card["situation"].update({"elapsed_h": 1.0, "trigger": "t", "alternatives_rejected": [{"option": "o", "reason": "r"}]})
    card["problem"]["statement"] = "s"
    card["hypothesis"].update({"claim": "c", "falsified_if": "f"})
    card["setup"]["parent_checkpoint"].update({"path": "google/gemma-3-4b-pt", "origin": "base_model"})
    card["setup"]["data"] = [{"path": str(root / "data" / "train.jsonl"), "source": "local", "n_examples": 10}]
    card["setup"]["method"].update({"family": "sft", "stop_token": "<|im_end|>", "answer_marker": "#### ",
                                    "hyperparams": {"max_seq_len": 512}})
    card["setup"]["command"].update({"argv": ["python", "train.py"], "cwd": str(root), "script": str(root / "train.py")})
    card["setup"]["output_dir"] = str(root / "ckpts" / "exp-01")
    card["setup"]["checkpoints"]["keep"] = "last"
    card["evaluation"]["protocol"]["n"] = 150
    cards.dump_card(p, card)
    assert main(["exp_protocol", "lock", "--dir", str(root), "exp-01"]) == 0
    return root


def test_review_close_reconcile_ledger(session, capsys) -> None:
    d = str(session)
    assert main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic"]) == 0
    vp = schema.verdict_path(lineage.cards_dir(session) / "exp-01.yaml")
    out = capsys.readouterr().out
    assert vp.is_file()
    assert out.splitlines()[0].startswith("exp-01: worth running now = ")   # the scientist's one line comes first
    assert "levels:" in out                                                  # the decomposition stays secondary

    p = lineage.cards_dir(session) / "exp-01.yaml"
    card = cards.load_card(p)
    (session / "ckpts" / "exp-01" / "final").mkdir(parents=True)
    card["result"] = {"execution": "completed", "output_checkpoint": str(session / "ckpts" / "exp-01" / "final"),
                      "wall_h": 1.2, "measurements": [{"metric": "accuracy", "value": 0.4, "n": 150,
                                                       "path": str(session / "e.json"), "delta_vs_comparator": 0.02}]}
    card["conclusion"] = {"verdict": "supported", "mechanism_verdict": "not_tested", "summary": "ok", "decision": "adopt"}
    cards.dump_card(p, card)
    assert main(["exp_protocol", "close", "--dir", d, "exp-01"]) == 0
    assert main(["wma", "reconcile", "--dir", d, "exp-01"]) == 0
    assert schema.load_verdict(vp)["scored"]["L2"] == "in_interval"

    assert main(["wma", "ledger", d]) == 0
    out = capsys.readouterr().out
    assert "heuristic" in out and "| 1 | 1 |" in out
    assert main(["wma", "ledger", d, "--csv"]) == 0
    assert capsys.readouterr().out.startswith("wma_skill,")


def test_review_refuses_post_hoc_and_reports_usage_errors(session, capsys) -> None:
    d = str(session)
    assert main(["wma", "review", "--dir", d, "exp-99", "--backend", "heuristic"]) == 2
    assert main(["wma", "reconcile", "--dir", d, "exp-01"]) == 1   # no verdict yet
    assert main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic", "--budget", "wall=abc"]) == 2


def test_review_with_an_unknown_backend_is_a_usage_error(session) -> None:
    with pytest.raises(SystemExit):
        main(["wma", "review", "--dir", str(session), "exp-01", "--backend", "oracle"])
