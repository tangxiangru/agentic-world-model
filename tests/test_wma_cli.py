"""awm wma end to end: a card is locked by the protocol, reviewed, closed by the protocol, counted."""

from __future__ import annotations

import json
import time

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


def test_review_close_ledger(session, capsys) -> None:
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
    assert "scored" not in schema.load_verdict(vp)          # the verdict file is never touched after review

    assert main(["wma", "ledger", d]) == 0
    out = capsys.readouterr().out
    assert "heuristic" in out and "| 1 | 1 |" in out
    assert main(["wma", "ledger", d, "--csv"]) == 0
    assert capsys.readouterr().out.startswith("wma_skill,")


def test_review_reports_usage_errors(session, capsys) -> None:
    d = str(session)
    assert main(["wma", "review", "--dir", d, "exp-99", "--backend", "heuristic"]) == 2
    assert main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic", "--budget", "wall=abc"]) == 2


def test_review_with_an_unknown_backend_is_a_usage_error(session) -> None:
    with pytest.raises(SystemExit):
        main(["wma", "review", "--dir", str(session), "exp-01", "--backend", "oracle"])


# ---- non-blocking, batch, tagged (2026-09-02) ---------------------------------


def _more_cards(session, ids):
    p0 = lineage.cards_dir(session) / "exp-01.yaml"
    base = cards.load_card(p0)
    for cid in ids:
        c = json.loads(json.dumps(base))
        c["card_id"] = cid
        cards.dump_card(lineage.cards_dir(session) / f"{cid}.yaml", c)


def test_several_cards_are_reviewed_in_one_call_and_ranked(session, capsys) -> None:
    _more_cards(session, ["exp-02", "exp-03"])
    d = str(session)
    assert main(["wma", "review", "--dir", d, "exp-01", "exp-02", "exp-03", "--backend", "heuristic", "--jobs", "3"]) == 0
    for cid in ("exp-01", "exp-02", "exp-03"):
        assert schema.verdict_path(lineage.cards_dir(session) / f"{cid}.yaml").is_file()
    out = capsys.readouterr().out
    assert out.count("worth running now") == 3 and "ranking" in out


def test_background_review_returns_at_once_and_the_verdict_arrives_later(session, capsys) -> None:
    d = str(session)
    assert main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic", "--background"]) == 0
    out = capsys.readouterr().out
    assert "background" in out
    vp = schema.verdict_path(lineage.cards_dir(session) / "exp-01.yaml")
    for _ in range(100):
        if vp.is_file():
            break
        time.sleep(0.1)
    assert vp.is_file(), "the detached review never wrote the verdict"
    assert main(["wma", "status", "--dir", d]) == 0
    assert "exp-01" in capsys.readouterr().out


def test_a_tag_lets_two_agents_review_the_same_card(session, capsys) -> None:
    d = str(session)
    assert main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic", "--tag", "a"]) == 0
    assert main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic", "--tag", "b"]) == 0
    cdir = lineage.cards_dir(session)
    assert (cdir / "exp-01.verdict.a.json").is_file() and (cdir / "exp-01.verdict.b.json").is_file()
    from awm.wma import ledger
    rows = ledger.rows([session])
    assert len(rows) == 2 and all(r["card_id"] == "exp-01" for r in rows)
    assert main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic", "--tag", "bad tag!"]) == 2


def test_status_reports_pending_and_done(session, capsys) -> None:
    _more_cards(session, ["exp-02"])
    d = str(session)
    main(["wma", "review", "--dir", d, "exp-01", "--backend", "heuristic"])
    capsys.readouterr()
    assert main(["wma", "status", "--dir", d]) == 0
    out = capsys.readouterr().out
    assert "exp-01" in out and "exp-02" in out and "no verdict" in out


def test_the_cli_builds_and_serves_the_protocol_without_the_wma_package(monkeypatch) -> None:
    """The ablation sandbox ships a checkout with no awm/wma; every other command must still work."""
    import awm.cli as cli

    real = cli.importlib.util.find_spec
    monkeypatch.setattr(cli.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "awm.wma" else real(name, *a, **k))
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["wma", "status", "--dir", "x"])
    assert parser.parse_args(["exp_protocol", "index", "--dir", "x"]).cmd == "index"


def test_budget_accepts_turns() -> None:
    from awm.wma.cli import _budget
    b = _budget("wall=8,turns=25")
    assert b.wall_min == 8 and b.max_turns == 25
    with pytest.raises(ValueError):
        _budget("steps=3")
