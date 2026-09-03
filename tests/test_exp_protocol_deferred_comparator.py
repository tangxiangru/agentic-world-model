"""K's opt-in lifecycle through real CPU CLI calls; no model or Slurm execution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from awm import paths
from awm.ptb_experiments import EXP_PROTOCOL_SHIP
from awm.cli import main
from awm.exp_protocol import collect, comparator, install, lineage, lock, preflight, schema
from exp_protocol_cards import plan_card


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(paths.REPO_ROOT / "skills/exp_protocol"))
    root = tmp_path / "session"
    root.mkdir()
    (root / "train.py").write_text("pass\n")
    (root / "data.jsonl").write_text(' {"completion":"ok<|im_end|>"}\n' * 20)
    (root / "eval").mkdir()
    (root / "ckpts/final").mkdir(parents=True)
    card = plan_card()
    card["setup"]["command"] = {"argv": ["python", "train.py"], "cwd": str(root),
                                  "script": str(root / "train.py")}
    card["setup"]["data"] = [{"path": str(root / "data.jsonl"), "source": "local", "n_examples": 20}]
    card["setup"]["output_dir"] = str(root / "ckpts")
    card["hypothesis"]["expected_effect"] = {"metric": "accuracy"}
    card["evaluation"]["comparator"] = {"ref": "base_model", "value": None,
                                          "path": str(root / "eval/parent.json"), "defer_validation": True}
    p = lineage.cards_dir(root) / "exp-01.yaml"
    schema.dump_card(p, card)
    return root, p


def invoke(root, command, *extra):
    return main(["exp_protocol", command, "--dir", str(root), "exp-01", *extra])


def finish(root, p, *, execution="completed", verdict="supported", decision="adopt", failure=None):
    c = schema.load_card(p)
    c["result"] = {"execution": execution, "output_checkpoint": str(root / "ckpts/final"),
                   "measurements": [{"metric": "accuracy", "value": .8, "n": c["evaluation"]["protocol"]["n"], "path": str(root / "child.json")}],
                   "failure": failure}
    c["conclusion"] = {"verdict": verdict, "decision": decision,
                       "mechanism_verdict": "not_tested", "summary": "toy lifecycle test"}
    schema.dump_card(p, c)


def evidence(root, payload=None):
    (root / "eval/parent.json").write_text(json.dumps(payload if payload is not None else {"n": 150, "accuracy": .5}))


def hook(root, hook_path=None):
    p = hook_path or preflight.skill_dir() / "hooks/stop_open_cards.py"
    result = subprocess.run([sys.executable, "-S", str(p)], input=json.dumps({"cwd": str(root)}),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def state(root):
    cards = lineage.load_cards(lineage.cards_dir(root))
    return lineage.index_rows(cards, lineage.cards_dir(root))[0], lineage.starting_points(cards), collect.collect([root])[0]


def test_full_deferred_lifecycle_and_all_consumers(setup):
    root, p = setup
    assert invoke(root, "check") == 0
    assert invoke(root, "lock") == 0
    report = json.loads(lock.preflight_path(p).read_text())
    assert next(x for x in report["results"] if x["check"] == "comparator_same_protocol")["status"] == "warn"
    finish(root, p)
    assert invoke(root, "close") == 1
    assert not p.with_suffix(".comparator.json").exists()
    row, points, stats = state(root)
    assert row["status"] == "unverified" and points == []
    assert stats["n_closed"] == 1  # preserve the raw conclusion statistic
    assert stats["n_deferred_unverified"] == 1 and stats["n_deferred_verified"] == 0
    assert hook(root)["decision"] == "block"
    evidence(root)
    assert invoke(root, "close") == 0
    proof = json.loads(p.with_suffix(".comparator.json").read_text())
    assert proof["outcome"] == "verified" and proof["observation"]["value"] == .5
    assert schema.load_card(p)["evaluation"]["comparator"]["value"] is None
    row, points, stats = state(root)
    assert row["status"] == "closed" and len(points) == 1
    assert stats["n_deferred_verified"] == 1 and stats["n_deferred_unverified"] == 0
    assert hook(root) == {}


@pytest.mark.parametrize("payload", [
    {"n": 300, "accuracy": .5}, {"accuracy": .5}, {"limit": 150, "accuracy": .5},
    {"n": True, "accuracy": .5}, {"n": 150, "accuracy": float("nan")},
    {"n": 150, "accuracy": 10 ** 400}, {"n": 150}, [],
    {"n": 150, "accuracy": .5, "status": "error"},
    {"n": 150, "num_samples": 149, "accuracy": .5},
    {"n": 150, "accuracy": .5, "results": {"completed_samples": 149}},
])
def test_invalid_known_or_unknown_evidence_never_certifies(setup, payload):
    root, p = setup
    evidence(root, payload)
    assert invoke(root, "lock") == 1  # cannot hide an existing bad result behind deferral
    (root / "eval/parent.json").unlink()
    assert invoke(root, "lock") == 0
    finish(root, p)
    evidence(root, payload)
    assert invoke(root, "close") == 1
    assert not p.with_suffix(".comparator.json").exists()


def inspect_payload(n=8):
    return {"status": "success", "eval": {"dataset": {"samples": 1319}, "config": {"limit": n}},
            "results": {"total_samples": n, "completed_samples": n,
                        "scores": [{"scored_samples": n, "unscored_samples": 0,
                                    "metrics": {"accuracy": {"value": .625}}}]},
            "samples": [{"id": i} for i in range(n)]}


def test_inspect_counts_are_actual_not_dataset_population(setup):
    root, p = setup
    c = schema.load_card(p)
    c["evaluation"]["protocol"]["n"] = 8
    schema.dump_card(p, c)
    assert invoke(root, "lock") == 0
    finish(root, p)
    evidence(root, inspect_payload())
    assert invoke(root, "close") == 0
    assert json.loads(p.with_suffix(".comparator.json").read_text())["observation"]["n"] == 8


@pytest.mark.parametrize("fault", ["unscored", "sample_error", "partial", "non_json"])
def test_incomplete_or_unreadable_inspect_report_fails(setup, fault):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p)
    payload = inspect_payload(150)
    if fault == "unscored":
        payload["results"]["scores"][0]["unscored_samples"] = 1
    elif fault == "sample_error":
        payload["samples"][0]["error"] = {"message": "failed"}
    elif fault == "partial":
        payload["samples"].pop()
    evidence(root, payload)
    if fault == "non_json":
        (root / "eval/parent.json").write_text("not json")
    assert invoke(root, "close") == 1


@pytest.mark.parametrize("execution", ["failed", "killed", "not_run"])
def test_failed_experiment_closes_honestly_without_comparator(setup, execution):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p, execution=execution, verdict="inconclusive", decision="abandon_line", failure="paired evaluation failed")
    assert invoke(root, "close") == 0
    proof = json.loads(p.with_suffix(".comparator.json").read_text())
    assert proof["outcome"] == execution and proof["observation"] is None
    assert hook(root) == {} and state(root)[1] == []
    assert state(root)[2]["n_deferred_failed_closed"] == 1


@pytest.mark.parametrize("verdict,decision,failure", [
    ("supported", "reject", "failed"), ("contradicted", "reject", "failed"),
    ("inconclusive", "adopt", "failed"), ("inconclusive", "reject", None),
])
def test_failure_route_does_not_certify_success(setup, verdict, decision, failure):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p, execution="failed", verdict=verdict, decision=decision, failure=failure)
    assert invoke(root, "close") == 1
    assert not p.with_suffix(".comparator.json").exists()


def test_edits_invalidate_receipt_and_mode_cannot_be_removed(setup):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p); evidence(root)
    assert invoke(root, "close") == 0
    c = schema.load_card(p)
    c["conclusion"]["summary"] = "updated explanation"
    schema.dump_card(p, c)
    assert hook(root)["decision"] == "block" and state(root)[1] == []
    assert invoke(root, "close") == 0
    evidence(root, {"n": 150, "accuracy": .6})
    assert hook(root)["decision"] == "block" and state(root)[1] == []
    assert invoke(root, "close") == 0
    c = schema.load_card(p)
    c["evaluation"]["comparator"]["defer_validation"] = False
    schema.dump_card(p, c)
    assert invoke(root, "close") == 1
    assert invoke(root, "lock", "--relock", "disable the check") == 1
    assert hook(root)["decision"] == "block" and state(root)[1] == []


def test_receipt_tampering_and_relock_invalidate_old_proof(setup):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p); evidence(root)
    assert invoke(root, "close") == 0
    proof_path = p.with_suffix(".comparator.json")
    original = proof_path.read_bytes()
    for modified in [b"not json", b"{}", original.replace(b'"value": 0.5', b'"value": 0.9')]:
        proof_path.write_bytes(modified)
        assert hook(root)["decision"] == "block" and state(root)[1] == []
    proof_path.write_bytes(original)
    assert hook(root) == {}
    assert invoke(root, "lock", "--relock", "repeat verification") == 0
    assert hook(root)["decision"] == "block"
    assert invoke(root, "close") == 0


def test_portable_receipt_and_standalone_installed_hook(setup, tmp_path):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p); evidence(root)
    assert invoke(root, "close") == 0
    install.install(root, "claude")
    installed_hook = root / "skills/exp_protocol/hooks/stop_open_cards.py"
    assert hook(root, installed_hook) == {}  # -S: no YAML/third-party dependency
    relocated = tmp_path / "harvest/cell/task"
    shutil.copytree(root, relocated)
    q = lineage.cards_dir(relocated) / p.name
    evidence(relocated, {"n": 150, "accuracy": .9})
    assert not comparator.completion_state(q, schema.load_card(q), lock.read_lock(q))["valid"]
    (relocated / "eval/parent.json").unlink()
    (root / "eval/parent.json").unlink()
    assert not comparator.completion_state(p, schema.load_card(p), lock.read_lock(p))["valid"]
    moved = comparator.completion_state(q, schema.load_card(q), lock.read_lock(q))
    assert moved["valid"] and moved["evidence_available"] is False
    assert state(relocated)[2]["n_deferred_verified"] == 1


def test_opt_in_is_typed_and_cannot_claim_prelaunch_measurement(setup):
    root, p = setup
    original = schema.load_card(p)
    for key, value in [("defer_validation", "true"), ("value", .5), ("path", "relative.json"), ("ref", "")]:
        c = schema.load_card(p)
        c["evaluation"]["comparator"][key] = value
        assert not schema.validate_plan(c, root).ok
    legacy = original
    legacy["evaluation"]["comparator"].pop("defer_validation")
    schema.dump_card(p, legacy)
    assert invoke(root, "lock") == 1


def test_override_does_not_bypass_postrun_validation(setup):
    root, p = setup
    evidence(root, {"n": 300, "accuracy": .5})
    assert invoke(root, "lock", "--override", "comparator_same_protocol=test override") == 0
    finish(root, p)
    assert invoke(root, "close") == 1
    with pytest.raises(ValueError):
        comparator.write_completion(p, schema.load_card(p), lock.read_lock(p))
    assert not p.with_suffix(".comparator.json").exists()


def test_deferred_hook_keeps_the_existing_twelve_block_cap(setup):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p)
    for _ in range(12):
        assert hook(root)["decision"] == "block"
    assert hook(root) == {}
    assert state(root)[1] == []  # exhausting a liveness guard is not scientific verification


def test_real_retained_inspect_report_is_supported():
    report = (paths.REPO_ROOT / "results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2"
              / "g01s07/task/logs/2026-09-03T10-18-54+00-00_gsm8k_c9nAXCMAt5uogCsgH5DczH.json")
    result = comparator.helper().inspect_output(str(report), 8, "accuracy")
    assert result["status"] == "pass" and result["value"] == .625
    assert comparator.helper().inspect_output(str(report), 1319, "accuracy")["status"] == "fail"


def test_reader_works_with_only_the_six_shipped_paths(setup, tmp_path):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p); evidence(root)
    assert invoke(root, "close") == 0
    shipped = tmp_path / "shipped"
    for rel in EXP_PROTOCOL_SHIP:
        source, target = paths.REPO_ROOT / rel, shipped / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)
    env = {**os.environ, "PYTHONPATH": str(shipped), "AWM_EXP_PROTOCOL_DIR": str(shipped / "skills/exp_protocol")}
    code = ("import json,sys; from pathlib import Path; from awm.exp_protocol import collect,comparator; "
            "print(json.dumps({'api':comparator.__file__,'helper':comparator.helper().__file__,"
            "'row':collect.collect([Path(sys.argv[1])])[0]}))")
    result = subprocess.run([sys.executable, "-c", code, str(root)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert Path(payload["api"]) == shipped / "awm/exp_protocol/comparator.py"
    assert Path(payload["helper"]) == shipped / "skills/exp_protocol/hooks/comparator_receipt.py"
    assert payload["row"]["n_deferred_verified"] == 1


@pytest.mark.parametrize("update", [{"n": 300}, {"metric": "loss"}, {"value": float("nan")}, {"path": ""}])
def test_matching_target_record_is_required_for_completed_comparison(setup, update):
    root, p = setup
    assert invoke(root, "lock") == 0
    finish(root, p); evidence(root)
    c = schema.load_card(p)
    c["result"]["measurements"][0].update(update)
    schema.dump_card(p, c)
    assert invoke(root, "close") == 1
    assert not p.with_suffix(".comparator.json").exists()
