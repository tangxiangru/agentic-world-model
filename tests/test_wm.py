"""End-to-end conformance of the world-model runtime with a fake trainer and fake graders.

No GPU, no model: ``evaluate.py`` and the custom scorer read a ``score.txt``
inside each checkpoint directory. The loop exercised:

propose -> brief -> accept -> freeze (parent scored) -> checkpoints at the
standing fractions -> yields -> observations -> notices -> final -> decision
(timeout: select best) -> seal -> finalize adopt -> {submission} symlink ->
memory rows -> a second session in the retrieval arm sees the precedent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from awm.wm.runtime import Session
from awm.wm.schema import HOOK_CONTINUE, HOOK_YIELD, WMError

REPO = Path(__file__).resolve().parent.parent

FAKE_EVALUATE = textwrap.dedent('''
    import argparse, json, pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path"); ap.add_argument("--limit", type=int); ap.add_argument("--json-output-file")
    a = ap.parse_args()
    score_file = pathlib.Path(a.model_path) / "score.txt"
    acc = float(score_file.read_text()) if score_file.is_file() else 0.30
    pathlib.Path(a.json_output_file).write_text(json.dumps({"accuracy": acc, "stderr": 0.02, "n": a.limit}))
''')

FAKE_SCORER = textwrap.dedent('''
    import json, pathlib, sys
    ckpt, items, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    acc = float((ckpt / "score.txt").read_text()) if (ckpt / "score.txt").is_file() else 0.30
    rows = [json.loads(l) for l in items.read_text().splitlines() if l.strip()]
    k = round(acc * len(rows))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"id": r["id"], "correct": i < k}) + "\\n")
    (out / "metrics.json").write_text(json.dumps({"value": k / len(rows), "n": len(rows)}))
''')


def make_session(tmp_path: Path, *, parent_score: float, arm: str = "null", memory_root: Path | None = None) -> tuple[Session, Path]:
    sd = tmp_path / "session"
    sd.mkdir(parents=True)
    (sd / "evaluate.py").write_text(FAKE_EVALUATE)
    scorer = sd / "score_fake.py"
    scorer.write_text(FAKE_SCORER)
    parent = sd / "parent"
    parent.mkdir()
    (parent / "config.json").write_text(json.dumps({"model_type": "fake", "architectures": ["Fake"]}))
    (parent / "score.txt").write_text(str(parent_score))
    (sd / "data").mkdir()
    (sd / "data" / "train.jsonl").write_text("\n".join(json.dumps({"q": i}) for i in range(10)) + "\n")
    (sd / "evals").mkdir()
    (sd / "evals" / "dev.jsonl").write_text(json.dumps({"id": "d1", "correct": False}) + "\n")
    watch = [{"id": f"w{i}", "question": f"q{i}", "gold": str(i)} for i in range(1, 5)]
    (sd / "evals" / "watch.jsonl").write_text("\n".join(json.dumps(w) for w in watch) + "\n")
    (sd / "runs").mkdir()
    mem = memory_root or (tmp_path / "wm-memory")
    session = Session.init(
        sd, arm=arm, memory_root=str(mem), submission=str(sd / "submission"),
        official_argv=[sys.executable, "evaluate.py", "--model-path", "{checkpoint}", "--limit", "{n}",
                       "--json-output-file", "{out}/metrics.json"],
        custom_argv=[sys.executable, str(scorer), "{checkpoint}", "{items}", "{out}"],
        spawn_worker=False, auto_relaunch=False,
        timeouts_s={"brief": 60, "yield_request": 1, "decision": 1}, poll_s=0.05,
    )
    return session, sd


def write_card(sd: Path, card_id: str = "exp-01") -> Path:
    card = {
        "schema_version": "awm-experiment-card-v1", "card_id": card_id, "created_at": "2026-08-31T00:00:00Z", "elapsed_h": 0.5,
        "problem": {
            "statement": "The base model drops intermediate arithmetic on multi-step word problems.",
            "evidence": [{"path": str(sd / "evals" / "dev.jsonl"), "locator": "d1", "observation": "1 of 1 fails"}],
            "affected_share": 0.4,
            "failure_examples": [
                {"id": f"w{i}", "source": str(sd / "evals" / "watch.jsonl"), "question": f"q{i}", "gold": str(i),
                 "model_output": "wrong", "failure": "arithmetic slip"} for i in (1, 2, 3)
            ],
            "watch_set": {"path": str(sd / "evals" / "watch.jsonl"), "n": 4, "selection": "all dev failures"},
        },
        "hypothesis": {
            "claim": "SFT on 10 worked solutions will reduce arithmetic slips on the watch set.",
            "mechanism": "explicit intermediate steps in the target",
            "expected_effect": {"metric": "accuracy", "direction": "higher", "against": "base_model", "magnitude": None},
            "falsified_if": "watch fixed < 2 and dev delta <= 0 after training",
        },
        "setup": {
            "parent_checkpoint": {"path": str(sd / "parent"), "origin": "base_model", "hash": None},
            "base_model": "fake/base-1b",
            "data": [{"path": str(sd / "data" / "train.jsonl"), "source": "local", "n_examples": 10, "built_by": None,
                      "build_command": [], "selection": "all", "contamination_check": "passed", "mixture_weight": 1.0}],
            "method": {"family": "sft", "framework": "fake", "peft": "none",
                       "hyperparams": {"lr": 1e-5, "epochs": 1, "seed": 0}, "target_format": "text"},
            "command": {"argv": ["python", "train.py"], "cwd": str(sd), "script": str(sd / "train.py"), "env": {},
                        "log": str(sd / "logs" / "train.log")},
            "resume_argv": ["python", "train.py", "--resume_from_checkpoint", "{checkpoint}"],
            "output_dir": str(sd / "runs" / card_id),
            "progress": {"unit": "optimizer_step", "total": 1000},
            "budget": {"gpu": "fake", "planned_h": 0.1},
        },
        "evaluation": {
            "protocol": {"command": ["python", "evaluate.py", "--limit", "150"], "dev_set": "official --limit 150",
                         "n": 150, "seed": 0},
            "comparator": {"ref": "base_model", "value": None, "path": None},
        },
    }
    path = sd / "memory" / "cards" / f"{card_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(card, sort_keys=False))
    return path


def make_ckpt(sd: Path, card_id: str, step: int, score: float) -> Path:
    ckpt = sd / "runs" / card_id / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps({"model_type": "fake"}))
    (ckpt / "score.txt").write_text(str(score))
    return ckpt


def test_full_loop_select_best_and_adopt(tmp_path: Path) -> None:
    s, sd = make_session(tmp_path, parent_score=0.30)
    card_path = write_card(sd)

    brief = s.propose(card_path)
    assert brief["kind"] == "brief" and brief["reply_required"]
    assert s.status("exp-01")["status"] == "draft"
    assert [p["ping_id"] for p in s.pending_replies()] == ["p-1"]
    with pytest.raises(WMError):  # cannot re-propose while the brief is unanswered
        s.propose(card_path)

    out = s.reply("exp-01/p-1", "accept")
    assert out["status"] == "frozen"
    assert out["parent"]["dev150"] == pytest.approx(0.30)
    assert out["parent"]["watch"] == pytest.approx(0.25)  # round(0.3*4)=1 of 4
    contract = s.contract("exp-01")
    assert contract["standing_yields"]["evaluators"] == ["dev150", "watch"]
    assert (s.card_dir("exp-01") / "evaluators" / "watch" / "items.jsonl").is_file()

    # an early save that hits no fraction continues
    assert s.checkpoint("exp-01", make_ckpt(sd, "exp-01", 100, 0.31), step=100) == HOOK_CONTINUE
    assert s.status("exp-01")["status"] == "running"

    scores = {250: 0.34, 500: 0.40, 750: 0.42, 1000: 0.41}
    for step, score in scores.items():
        code = s.checkpoint("exp-01", make_ckpt(sd, "exp-01", step, score), step=step, final=(step == 1000))
        assert code == HOOK_YIELD
        # a second hook call while the yield is pending does not stack
        assert s.checkpoint("exp-01", make_ckpt(sd, "exp-01", step, score), step=step) == HOOK_CONTINUE
        out = s.run_worker("exp-01")
        if step < 1000:
            assert out["resume"] == "manual"
            assert s.mailbox("exp-01").pings()[-1]["kind"] == "notice"
        else:
            # completion decision, no reply -> timeout selects the best observation
            assert out["status"] == "awaiting_review"
            assert out["sealed"] == "obs-3"  # step 750, 0.42

    st = s.status("exp-01")
    assert st["status"] == "awaiting_review"
    assert [o["obs_id"] for o in st["observations"]] == ["obs-1", "obs-2", "obs-3", "obs-4"]
    obs3 = json.loads((s.card_dir("exp-01") / "observations" / "obs-3" / "observation.json").read_text())
    assert obs3["evaluators"]["dev150"]["delta_vs_parent"] == pytest.approx(0.12)
    assert obs3["watch"]["fixed"] == 1 and obs3["watch"]["regressions"] == 0
    decision = [p for p in s.mailbox("exp-01").pings() if p["kind"] == "decision"]
    assert len(decision) == 1 and decision[0]["raised_by"] == "completion"
    reply = s.mailbox("exp-01").reply(decision[0]["ping_id"])
    assert reply["by"] == "timeout" and reply["choice"] == "select:obs-3"
    seal = json.loads((s.card_dir("exp-01") / "seal.json").read_text())
    assert seal["checkpoint"]["path"].endswith("checkpoint-750")
    assert seal["decision_ping"] == decision[0]["ping_id"]

    # sections 5-6 go into the same card file; sections 1-4 must be unchanged
    completed = yaml.safe_load(card_path.read_text())
    completed["result"] = {"execution": "completed", "wall_h": 0.1, "output_checkpoint": seal["checkpoint"]["path"],
                           "measurements": [{"metric": "accuracy", "value": 0.42, "n": 150,
                                             "path": str(s.card_dir("exp-01") / "observations" / "obs-3" / "dev150" / "normalized.json"),
                                             "delta_vs_comparator": 0.12}]}
    completed["conclusion"] = {"verdict": "supported", "decision": "adopt", "summary": "dev150 +0.12 vs parent; watch fixed 1."}
    tampered = dict(completed)
    tampered["hypothesis"] = {**completed["hypothesis"], "claim": "rewritten after the fact"}
    tpath = sd / "memory" / "cards" / "tampered.yaml"
    tpath.write_text(yaml.safe_dump(tampered))
    with pytest.raises(WMError, match="sections 1-4 differ"):
        s.finalize("exp-01", tpath)
    card_path.write_text(yaml.safe_dump(completed, sort_keys=False))
    out = s.finalize("exp-01", card_path)
    assert out["status"] == "closed" and out["decision"] == "adopt"
    sub = Path(s.config["submission"])
    assert sub.is_symlink() and sub.resolve() == Path(seal["checkpoint"]["path"]).resolve()
    assert s.memory.stats()["cards"] == 1 and s.memory.stats()["observations"] == 4
    assert s.pending_replies() == []

    # a second session in the retrieval arm sees the precedent
    s2, sd2 = make_session(tmp_path / "two", parent_score=0.30, arm="retrieval", memory_root=tmp_path / "wm-memory")
    brief2 = s2.propose(write_card(sd2, "exp-01"))
    assert "1 precedents in memory" in brief2["summary"]
    assert any(e["locator"] == "exp-01" for e in brief2["evidence"])


def test_regress_rule_fires_and_defaults_to_abort(tmp_path: Path) -> None:
    s, sd = make_session(tmp_path, parent_score=0.50)
    s.propose(write_card(sd))
    s.reply("exp-01/p-1", "accept")
    for step, score in {250: 0.44, 500: 0.43}.items():
        assert s.checkpoint("exp-01", make_ckpt(sd, "exp-01", step, score), step=step) == HOOK_YIELD
        out = s.run_worker("exp-01")
    assert out["status"] == "awaiting_review" and out.get("aborted") is True
    decision = next(p for p in s.mailbox("exp-01").pings() if p["kind"] == "decision")
    assert decision["raised_by"] == "rule:regress"
    assert decision["timeout_action"]["action"] == "abort"
    # a later hook call from a straggling trainer is told to abort
    assert s.checkpoint("exp-01", make_ckpt(sd, "exp-01", 600, 0.43), step=600) == 4


def test_withdraw_closes_without_running(tmp_path: Path) -> None:
    s, sd = make_session(tmp_path, parent_score=0.30)
    s.propose(write_card(sd))
    out = s.reply("exp-01/p-1", "withdraw", why="not worth the GPU")
    assert out["status"] == "closed" and out["decision"] == "abandon_line"
    result = yaml.safe_load((s.card_dir("exp-01") / "card.yaml").read_text())
    assert result["result"]["execution"] == "not_run" and result["conclusion"]["decision"] == "abandon_line"
    assert s.memory.stats()["cards"] == 1


def test_grounding_rejects_bad_cards(tmp_path: Path) -> None:
    s, sd = make_session(tmp_path, parent_score=0.30)
    path = write_card(sd)
    card = yaml.safe_load(path.read_text())
    card["problem"]["watch_set"]["n"] = 7
    path.write_text(yaml.safe_dump(card))
    with pytest.raises(WMError, match="watch_set.count"):
        s.propose(path)
    card["problem"]["watch_set"]["n"] = 4
    card["setup"]["output_dir"] = "/tmp/elsewhere"
    path.write_text(yaml.safe_dump(card))
    with pytest.raises(WMError, match="output_dir"):
        s.propose(path)


def test_replies_are_idempotent_and_immutable(tmp_path: Path) -> None:
    s, sd = make_session(tmp_path, parent_score=0.30)
    s.propose(write_card(sd))
    with pytest.raises(WMError, match="override requires"):
        s.reply("exp-01/p-1", "override")
    s.reply("exp-01/p-1", "accept")
    s.reply("exp-01/p-1", "accept")  # same choice: no-op
    with pytest.raises(WMError, match="immutable"):
        s.reply("exp-01/p-1", "withdraw", why="changed my mind")


def test_cli_and_stop_hook(tmp_path: Path) -> None:
    _s, sd = make_session(tmp_path, parent_score=0.30)
    env = {**os.environ, "AWM_SESSION_DIR": str(sd), "PYTHONPATH": str(REPO)}
    proc = subprocess.run([sys.executable, "-m", "awm.cli", "wm", "propose", str(write_card(sd))],
                          capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "BRIEF" in proc.stdout and "awm wm reply p-1" in proc.stdout
    hook = subprocess.run([sys.executable, str(REPO / ".claude/hooks/wm_pending_reply.py")],
                          input=json.dumps({"cwd": str(sd)}), capture_output=True, text=True, env=env, check=False)
    decision = json.loads(hook.stdout)
    assert decision.get("decision") == "block" and "exp-01/p-1" in decision["reason"]
    pending = subprocess.run([sys.executable, "-m", "awm.cli", "wm", "pending"], capture_output=True, text=True,
                             env=env, check=False)
    assert pending.returncode == 1
    proc = subprocess.run([sys.executable, "-m", "awm.cli", "wm", "reply", "exp-01/p-1", "--choose", "accept"],
                          capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    hook = subprocess.run([sys.executable, str(REPO / ".claude/hooks/wm_pending_reply.py")],
                          input=json.dumps({"cwd": str(sd)}), capture_output=True, text=True, env=env, check=False)
    assert json.loads(hook.stdout) == {}
    ckpt = make_ckpt(sd, "exp-01", 250, 0.35)
    proc = subprocess.run([sys.executable, "-m", "awm.cli", "wm", "checkpoint", "exp-01", str(ckpt), "--step", "250"],
                          capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == HOOK_YIELD and "YIELD" in proc.stdout


def test_adopt_copy_mode_and_memory_sides(tmp_path: Path) -> None:
    s, sd = make_session(tmp_path, parent_score=0.30)
    s.config["submission_mode"] = "copy"
    ckpt = make_ckpt(sd, "exp-01", 750, 0.42)
    (sd / "submission").mkdir()  # a stale directory from an earlier adopt
    (sd / "submission" / "old.txt").write_text("stale")
    s._adopt({"card_id": "exp-01", "seal": {"checkpoint": str(ckpt), "obs_id": "obs-3"}})
    sub = sd / "submission"
    assert sub.is_dir() and not sub.is_symlink()
    assert (sub / "config.json").is_file() and not (sub / "old.txt").exists()
    assert json.loads((s.wm / "incumbent.json").read_text())["obs_id"] == "obs-3"

    # memory sides: a test-side row is invisible by default, visible when asked for
    from awm.wm.memory import Memory
    mem_root = tmp_path / "mem2"
    writer = Memory(mem_root, session="x", arm="null", split_side="test")
    card = yaml.safe_load(write_card(sd).read_text())
    writer._append("cards", {"card_id": "exp-09", "base_model": "fake/base-1b", "method_family": "sft",
                             "data_sources": ["local"], "problem": card["problem"]["statement"],
                             "claim": card["hypothesis"]["claim"], "best_selection_value": 0.5, "parent_value": 0.3})
    assert Memory(mem_root, session="y", arm="retrieval").precedents(card) == []
    both = Memory(mem_root, session="y", arm="retrieval", visible_sides=("train", "test")).precedents(card)
    assert [p["card_id"] for p in both] == ["exp-09"] and both[0]["delta_best_vs_parent"] == pytest.approx(0.2)


def test_traj_arm_with_fake_backend_grounds_and_lints(tmp_path: Path) -> None:
    prior = tmp_path / "prior_runs"
    (prior / "cfg" / "run_a" / "task").mkdir(parents=True)
    (prior / "INDEX.md").write_text("| 0.7 | fake/base-1b | cfg | run_a |\n")
    (prior / "cfg" / "run_a" / "metrics.json").write_text('{"accuracy": 0.7}')
    s, sd = make_session(tmp_path, parent_score=0.30, arm="traj")
    s.config["prior_runs_root"] = str(prior)
    s.config["wma_backend"] = "fake"
    s.config["wma_fake"] = {
        "brief": {"summary": "Two prior runs on this base with the same recipe gained +0.10 to +0.14.",
                  "objections": [{"field": "evaluation.protocol.n", "severity": "advisory", "fix": "n=150 resolves ±0.04; expected effect is larger, fine"},
                                 {"field": "bogus", "severity": "fatal", "fix": "x"}],
                  "prediction": {"metric": "accuracy", "horizon": "final", "delta_mean": 0.12, "delta_sd": 0.03, "basis": "2 runs"},
                  "evidence": [{"path": str(prior / "cfg" / "run_a" / "metrics.json"), "locator": "accuracy", "observation": "0.70 final"},
                               {"path": "/etc/passwd", "locator": "1", "observation": "outside allowed roots"},
                               {"path": str(prior / "missing.json"), "locator": "x", "observation": "does not exist"}],
                  "recommendation": "run"},
        "observation": {"kind": "yield_request", "summary": "dev150 tracks run_a's curve at 25%.",
                        "evidence": [{"path": str(prior / "INDEX.md"), "locator": "run_a", "observation": "0.70"}],
                        "prediction": {"metric": "accuracy", "horizon": "final", "delta_mean": 0.11, "delta_sd": 0.02, "basis": "run_a"},
                        "request_evaluators": ["diag", "not_in_contract"], "recommendation": None},
    }
    brief = s.propose(write_card(sd))
    assert brief["summary"].startswith("Two prior runs") and brief["summary"].endswith("Recommendation: run.")
    assert brief["prediction"]["delta_mean"] == pytest.approx(0.12)
    paths = [e["path"] for e in brief["evidence"]]
    assert str(prior / "cfg" / "run_a" / "metrics.json") in paths
    assert "/etc/passwd" not in paths and str(prior / "missing.json") not in paths
    assert sum(1 for e in brief["evidence"] if e["locator"] == "evaluation.protocol.n") == 1  # the fatal one was dropped
    assert any(r["event"] == "lint" and r.get("where") == "brief" and r["dropped"] == 2 for r in s.ledger.rows())
    assert (s.wm / "agent-calls").is_dir() and list((s.wm / "agent-calls").glob("*-brief.prompt.md"))
    s.reply("exp-01/p-1", "accept")
    # the contract has no on_request evaluators, so the agent's yield_request degrades to a notice
    assert s.checkpoint("exp-01", make_ckpt(sd, "exp-01", 250, 0.34), step=250) == HOOK_YIELD
    s.run_worker("exp-01")
    last = s.mailbox("exp-01").pings()[-1]
    assert last["kind"] == "notice" and last["prediction"]["delta_mean"] == pytest.approx(0.11)
    assert any(e["path"] == str(prior / "INDEX.md") for e in last["evidence"])


def test_agent_degradation_is_recorded_and_strict_mode_refuses(tmp_path: Path) -> None:
    prior = tmp_path / "prior_runs"; prior.mkdir(); (prior / "INDEX.md").write_text("x\n")
    s, sd = make_session(tmp_path, parent_score=0.30, arm="traj")
    s.config.update({"prior_runs_root": str(prior), "wma_backend": "fake", "wma_fake": {}})  # no canned answers -> fallback
    brief = s.propose(write_card(sd))
    assert brief["agent"]["produced_by"] == "deterministic" and "no canned answer" in brief["agent"]["degraded"]
    assert brief["summary"].startswith("[agent degraded:")
    assert brief["agent"]["arm"] == "traj" and brief["agent"]["sources"] == ["prior_runs"]
    assert any(r["event"] == "agent_degraded" and r["where"] == "brief" for r in s.ledger.rows())
    assert s.status()["cards"][0]["degraded_calls"] == 1
    s.reply("exp-01/p-1", "accept")
    assert s.checkpoint("exp-01", make_ckpt(sd, "exp-01", 250, 0.34), step=250) == HOOK_YIELD
    s.run_worker("exp-01")
    last = s.mailbox("exp-01").pings()[-1]
    assert last["kind"] == "notice" and last["agent"]["produced_by"] == "deterministic" and last["agent"]["degraded"]
    assert s.status("exp-01")["degraded_calls"] == 2

    # strict: the brief refuses instead of falling back; the card stays a draft
    s2, sd2 = make_session(tmp_path / "strict", parent_score=0.30, arm="traj")
    s2.config.update({"prior_runs_root": str(tmp_path / "nope"), "wma_backend": "fake", "wma_strict": True})
    with pytest.raises(WMError, match="wma_strict"):
        s2.propose(write_card(sd2))
    assert s2.status("exp-01")["status"] == "draft"
    assert any(r["event"] == "agent_failed" for r in s2.ledger.rows())

    # a healthy call is stamped llm, and the retrieval arm records its k
    s3, sd3 = make_session(tmp_path / "ok", parent_score=0.30, arm="traj")
    s3.config.update({"prior_runs_root": str(prior), "wma_backend": "fake",
                      "wma_fake": {"brief": {"summary": "fine", "evidence": [], "prediction": None}}})
    b3 = s3.propose(write_card(sd3))
    assert b3["agent"]["produced_by"] == "llm" and b3["agent"]["degraded"] is None
    s4, sd4 = make_session(tmp_path / "ret", parent_score=0.30, arm="retrieval")
    assert s4.propose(write_card(sd4))["agent"]["retrieval_k"] == 5
