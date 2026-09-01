"""The world-model agent's toolbelt: the consult contract, the ledger, drafting, search, CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from awm.wm import consult, intake
from awm.wm.memory import Memory
from awm.wm.schema import WMError

REPO = Path(__file__).resolve().parent.parent

PLAN = """We want to fix multi-step arithmetic: the base model sets problems up correctly and slips on the numbers.
We expect SFT on worked solutions to raise dev accuracy over the base model.

```bash
python train.py --model_name_or_path google/gemma-3-4b-pt --train_file data/train.jsonl --output_dir runs/exp01 --max_steps 1000
```

Evaluate with `python evaluate.py --limit 150` at each checkpoint.
"""


def good_response(card: dict, *, stage: str = "plan", label: str = "CANNOT_DECIDE", conf: float = 0.5,
                  based_on: list | None = None, reasons: list | None = None) -> dict:
    return {
        "schema_version": consult.RESPONSE_SCHEMA, "stage": stage, "card": {**card, "gaps": card.get("gaps", [])},
        "verdict": {"label": label, "confidence": conf,
                    "prediction": {"metric": "accuracy", "horizon": "final", "delta_mean": 0.06, "delta_sd": 0.04, "basis": "2 runs"},
                    "based_on": based_on or []},
        "eval_plan": consult.default_eval_plan(1000, n=150, parent_value=0.30),
        "suggestion": {"label": "KEEP_RUNNING", "reason": "the first evaluation at step 250 will separate the cases"},
        "reasons": reasons or [],
    }


def session(tmp_path: Path) -> tuple[Path, Path, dict]:
    sd = tmp_path / "task"; (sd / "data").mkdir(parents=True); (sd / "wm" / "tmp").mkdir(parents=True)
    (sd / "data" / "train.jsonl").write_text("\n".join(json.dumps({"q": i}) for i in range(10)) + "\n")
    (sd / "train.py").write_text("print('train')\n")
    prior = tmp_path / "prior_runs"; (prior / "cfg" / "run_a").mkdir(parents=True)
    (prior / "INDEX.md").write_text("| 0.70 | google/gemma-3-4b-pt | cfg | run_a |\n")
    (prior / "cfg" / "run_a" / "metrics.json").write_text('{"accuracy": 0.70}')
    cfg = {"schema_version": "awm-wm-config-v2", "session_id": "t-1", "session_dir": str(sd), "arm": "traj",
           "prior_runs_root": str(prior), "memory_root": str(tmp_path / "mem"), "memory_sides": ["train"],
           "memory_readonly": False, "split_side": "train", "wma_model": "claude-opus-4-8", "base_model": "google/gemma-3-4b-pt"}
    (sd / "wm" / "config.json").write_text(json.dumps(cfg))
    return sd, prior, cfg


def test_draft_card_reads_the_plan_and_lists_gaps(tmp_path: Path) -> None:
    sd, _, _ = session(tmp_path)
    card, questions = intake.draft_card("exp-01", PLAN, {}, sd, base_model="google/gemma-3-4b-pt")
    assert card["problem"]["statement"].startswith("We want to fix multi-step arithmetic")
    assert "expect SFT" in card["hypothesis"]["claim"]
    assert card["setup"]["output_dir"] == str(sd / "runs" / "exp01")
    assert card["setup"]["progress"]["total"] == 1000 and card["evaluation"]["protocol"]["n"] == 150
    assert card["setup"]["data"][0]["n_examples"] == 10
    assert card["setup"]["parent_checkpoint"]["path"] == "google/gemma-3-4b-pt"
    assert [q["field"] for q in questions] == ["setup.resume_argv"]
    # a vague plan leaves most fields as gaps, none guessed
    card2, q2 = intake.draft_card("exp-02", "Train something better.", {}, sd)
    assert len(q2) >= 6 and card2["setup"].get("output_dir") is None


def test_response_contract_and_thresholds(tmp_path: Path) -> None:
    sd, prior, _ = session(tmp_path)
    card, _ = intake.draft_card("exp-01", PLAN, {}, sd)
    assert consult.validate_response(good_response(card)) == []
    # SURE_* needs confidence >= threshold and citations
    r = good_response(card, label="SURE_WILL_WORK", conf=0.6)
    probs = consult.validate_response(r)
    assert any("confidence >=" in p for p in probs) and any("must cite" in p for p in probs)
    r = good_response(card, label="SURE_WILL_WORK", conf=0.9,
                      based_on=[{"path": str(prior / "cfg/run_a/metrics.json"), "locator": "accuracy", "observation": "0.70"}])
    assert consult.validate_response(r) == []
    # bad shapes are named
    r = good_response(card); r["suggestion"] = {"label": "ADJUST", "reason": "x"}
    assert any("ADJUST needs" in p for p in consult.validate_response(r))
    r = good_response(card); r["stage"] = "later"
    assert any("stage" in p for p in consult.validate_response(r))
    r = good_response(card); r["verdict"]["prediction"]["delta_sd"] = -1
    assert any("delta_sd" in p for p in consult.validate_response(r))


def test_lint_downgrades_uncited_sure_verdicts(tmp_path: Path) -> None:
    sd, prior, _ = session(tmp_path)
    card, _ = intake.draft_card("exp-01", PLAN, {}, sd)
    r = good_response(card, label="SURE_WONT_WORK", conf=0.9,
                      based_on=[{"path": "/etc/passwd", "locator": "1", "observation": "outside roots"},
                                {"path": str(prior / "missing.json"), "locator": "x", "observation": "does not exist"}],
                      reasons=[{"claim": "ok", "path": str(prior / "INDEX.md"), "locator": "1"},
                               {"claim": "bad", "path": "/nowhere"}])
    linted, dropped = consult.lint_citations(r, [sd, prior])
    assert len(dropped) == 3
    assert linted["verdict"]["label"] == "CANNOT_DECIDE" and linted["verdict"]["confidence"] < consult.SURE_THRESHOLD
    assert [x["claim"] for x in linted["reasons"]] == ["ok"]


def test_default_eval_plan() -> None:
    plan = consult.default_eval_plan(1200, n=100, parent_value=0.31)
    assert [p["step"] for p in plan["points"]] == [300, 600, 900]
    assert plan["protocol"]["n"] == 100 and plan["comparator"]["value"] == 0.31
    assert consult.default_eval_plan(None)["points"] == []


def test_log_consult_and_outcome(tmp_path: Path) -> None:
    sd, prior, _cfg = session(tmp_path)
    card, _ = intake.draft_card("exp-01", PLAN, {}, sd)
    wm = sd / "wm"
    roots = [sd, prior]
    e1 = consult.log_consult(wm, good_response(card), request="planning SFT", roots=roots, arm="traj", model="claude-opus-4-8")
    assert e1["card_id"] == "exp-01" and e1["consult_n"] == 1 and e1["verdict"] == "CANNOT_DECIDE"
    assert (wm / "cards" / "exp-01" / "consult-01.json").is_file() and (wm / "cards" / "exp-01" / "card.json").is_file()
    r2 = good_response(card, stage="running", label="SURE_WILL_WORK", conf=0.8,
                       based_on=[{"path": str(prior / "cfg/run_a/metrics.json"), "locator": "accuracy", "observation": "0.70"}])
    r2["card"]["results"] = [{"step": 250, "metric": "accuracy", "value": 0.34, "n": 150}]
    e2 = consult.log_consult(wm, r2, request="step 250: 0.34", roots=roots, arm="traj", model="claude-opus-4-8")
    assert e2["consult_n"] == 2 and e2["n_based_on"] == 1 and e2["stage"] == "running"
    with pytest.raises(WMError, match="contract"):
        consult.log_consult(wm, {"schema_version": "x"}, request="", roots=roots, arm="traj", model=None)
    out = consult.record_outcome(wm, "exp-01", final_value=0.42, shipped=str(sd / "runs/exp01/checkpoint-750"))
    assert out["predictions_made"] == 2 and out["last_prediction"]["delta_mean"] == 0.06
    card_json = json.loads((wm / "cards" / "exp-01" / "card.json").read_text())
    assert card_json["outcome"]["final_value"] == 0.42
    rows = consult.ConsultLedger(wm / "consults.jsonl").rows()
    assert [r.get("event", "consult") for r in rows] == ["consult", "consult", "outcome"]


def test_memory_precedents_respect_visible_sides(tmp_path: Path) -> None:
    sd, _, _ = session(tmp_path)
    card, _ = intake.draft_card("exp-01", PLAN, {}, sd, base_model="google/gemma-3-4b-pt")
    root = tmp_path / "mem"
    writer = Memory(root, session="x", arm="null", split_side="test")
    writer._append("cards", {"card_id": "exp-09", "base_model": "google/gemma-3-4b-pt", "method_family": "sft",
                             "data_sources": ["local"], "problem": card["problem"]["statement"],
                             "claim": card["hypothesis"]["claim"], "best_selection_value": 0.5, "parent_value": 0.3})
    assert Memory(root, session="y", arm="retrieval").precedents(card) == []
    both = Memory(root, session="y", arm="retrieval", visible_sides=("train", "test")).precedents(card)
    assert [p["card_id"] for p in both] == ["exp-09"] and both[0]["delta_best_vs_parent"] == pytest.approx(0.2)


def test_cli_toolbelt_end_to_end(tmp_path: Path) -> None:
    sd = tmp_path / "task"; (sd / "data").mkdir(parents=True)
    (sd / "data" / "train.jsonl").write_text('{"q": 1}\n{"q": 2}\n')
    prior = tmp_path / "prior_runs"; (prior / "cfg" / "run_a").mkdir(parents=True)
    (prior / "INDEX.md").write_text("x\n"); (prior / "cfg/run_a/metrics.json").write_text('{"accuracy": 0.7}')
    env = {**os.environ, "AWM_SESSION_DIR": str(sd), "PYTHONPATH": str(REPO)}

    def wm(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "awm.cli", "wm", *args], capture_output=True, text=True, env=env, check=False)

    r = wm("init", "--arm", "traj", "--prior-runs", str(prior), "--memory-root", str(tmp_path / "mem"),
           "--wma-model", "claude-opus-4-8", "--base-model", "google/gemma-3-4b-pt")
    assert r.returncode == 0, r.stderr
    cfg = json.loads((sd / "wm" / "config.json").read_text())
    assert cfg["arm"] == "traj" and cfg["prior_runs_root"] == str(prior)

    r = wm("draft-card", "--text", PLAN)
    assert r.returncode == 0, r.stderr
    card = yaml.safe_load(r.stdout)
    assert card["card_id"] == "exp-01" and card["gaps"] and card["setup"]["progress"]["total"] == 1000

    r = wm("eval-plan", "--steps", "1000", "--parent", "0.30")
    assert json.loads(r.stdout)["points"][0]["step"] == 250

    (sd / "eval_step250.json").write_text(json.dumps({"accuracy": 0.34, "stderr": 0.039}))
    r = wm("read-eval", str(sd / "eval_step250.json"))
    assert json.loads(r.stdout)["value"] == 0.34
    assert wm("read-eval", "/etc/hosts").returncode != 0

    resp = good_response(card, label="SURE_WILL_WORK", conf=0.8,
                         based_on=[{"path": str(prior / "cfg/run_a/metrics.json"), "locator": "accuracy", "observation": "0.70"}])
    (sd / "wm" / "tmp" / "response.json").write_text(json.dumps(resp))
    (sd / "wm" / "tmp" / "request.txt").write_text(PLAN)
    r = wm("log", "--response", str(sd / "wm/tmp/response.json"), "--request", str(sd / "wm/tmp/request.txt"))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["verdict"] == "SURE_WILL_WORK"
    r = wm("outcome", "--card", "exp-01", "--final", "0.71", "--shipped", "runs/exp01/checkpoint-750")
    assert r.returncode == 0, r.stderr
    st = json.loads(wm("status").stdout)
    assert st["cards"]["exp-01"] == {"consults": 2, "last_verdict": "SURE_WILL_WORK", "last_suggestion": "KEEP_RUNNING", "outcome": 0.71}
    # search on a traj arm has no memory to search and says so
    assert json.loads(wm("search", "--text", PLAN).stdout)["precedents"] == []
