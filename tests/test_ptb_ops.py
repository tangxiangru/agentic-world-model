"""The operator's tools: queue, plan, apply, harvest, cancel, all against fakes."""

from __future__ import annotations

import gzip
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from awm import paths
from awm import ptb_experiments as ptb
from awm import ptb_ops as ops
from awm.cli import main

MANIFEST = paths.REPO_ROOT / "experiments/posttrainbench/gsm8k-opus5-4x4-batch1.yaml"


def _small_manifest(batch_id: str) -> dict:
    data = deepcopy(ptb.load_manifest(MANIFEST))
    data.pop("_path", None)
    gemma_max = next(cell for cell in data["cells"] if cell["id"] == "b04")
    data["batch_id"] = batch_id
    data["cells"] = [gemma_max | {"id": f"p01r{r}", "replicate": r} for r in (1, 2)]
    data["contract"]["replication"] = {"settings": 1, "repeats": 2}
    data["contract"]["base_models"] = {
        "google/gemma-3-4b-pt": data["contract"]["base_models"]["google/gemma-3-4b-pt"]
    }
    data["pilot"] = {"cell": "p01r1", "agent_budget_hours": 1}
    return data


@pytest.fixture
def repo(tmp_path: Path, monkeypatch):
    """A fake repository root with one manifest, a data volume, and no Slurm."""
    root = tmp_path / "repo"
    (root / "experiments" / "posttrainbench").mkdir(parents=True)
    # validate_manifest looks the spec up under paths.REPO_ROOT, which the CLI test repoints here
    spec = "doc/spec/2026-08-30-ptb-gpu-slicing-and-gsm8k-batch1.md"
    (root / spec).parent.mkdir(parents=True)
    (root / spec).write_text("# spec\n")
    manifest_path = root / "experiments" / "posttrainbench" / "ep-r01.yaml"
    manifest_path.write_text(yaml.safe_dump(_small_manifest("ep-r01"), sort_keys=False))
    monkeypatch.setattr(ops.paths, "data_root", lambda *_a, **_k: tmp_path / "vol")
    states: dict[str, str] = {}
    monkeypatch.setattr(ops, "job_state", lambda job_id: states.get(job_id, "UNKNOWN"))
    monkeypatch.setattr(ops, "result_for_job", lambda job_id: None)
    monkeypatch.setattr(ops, "audit_result", lambda result_dir: [])
    monkeypatch.setattr(ops, "_worktree_dirty", lambda repo_root: "")
    monkeypatch.setattr(ops, "_submission_ownership_issue", lambda: None)
    return root, states


def _queue(root: Path, *entries: dict) -> Path:
    path = root / "experiments" / "posttrainbench" / "queue.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "entries": list(entries)}, sort_keys=False))
    return path


def _receipt(tmp_root: Path, batch: str, kind: str, jobs: list[tuple[str, str]], state="submitted") -> Path:
    out = tmp_root / "ptb" / "batches" / batch
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{kind}-2026-09-02T000000.json"
    path.write_text(json.dumps({
        "schema_version": 1, "batch_id": batch, "kind": kind, "state": state,
        "jobs": [{"cell_id": c, "job_id": j, "job_name": f"branch.ptb.{batch}.{c}.{kind}.r1"} for c, j in jobs],
    }, indent=2))
    return path


ENTRY = {"manifest": "experiments/posttrainbench/ep-r01.yaml", "want": "submitted", "why": "round 1"}


# ---- queue -----------------------------------------------------------------

def test_queue_is_validated(repo) -> None:
    root, _ = repo
    path = _queue(root, ENTRY)
    assert ops.load_queue(path, root) == [ENTRY]
    for bad, message in (
        ({**ENTRY, "want": "maybe"}, "want must be"),
        ({**ENTRY, "manifest": "experiments/posttrainbench/missing.yaml"}, "does not exist"),
        ({**ENTRY, "manifest": "/etc/passwd"}, "committed experiments"),
        ({**ENTRY, "why": ""}, "why must say"),
        ({**ENTRY, "pilot": "later"}, "pilot must be"),
    ):
        with pytest.raises(ops.OpsError, match=message):
            ops.load_queue(_queue(root, bad), root)
    with pytest.raises(ops.OpsError, match="listed twice"):
        ops.load_queue(_queue(root, ENTRY, ENTRY), root)
    assert ops.load_queue(_queue(root), root) == []


def test_receipt_kind_survives_the_timestamp_in_the_name() -> None:
    assert ops._receipt_kind("pilot-2026-09-02T000000.123456+0000.json") == "pilot"
    assert ops._receipt_kind("formal-2026-09-02T000000.123456+0000.json") == "formal"
    assert ops._receipt_kind("formal-retry2-2026-09-02T000000.json") == "formal-retry2"
    assert ops._receipt_kind("something-else.json") == "something-else"


# ---- plan ------------------------------------------------------------------

def test_an_entry_without_a_receipt_is_submitted(repo) -> None:
    root, _ = repo
    (actions,) = ops.plan([ENTRY], root)
    assert (actions.kind, actions.batch, actions.pilot, actions.manifest) == (
        "submit", "ep-r01", False, ENTRY["manifest"])


def test_pilot_first_gates_the_formal_submission(repo, tmp_path: Path) -> None:
    root, states = repo
    entry = {**ENTRY, "pilot": "first"}
    (first,) = ops.plan([entry], root)
    assert (first.kind, first.pilot) == ("submit", True)

    _receipt(tmp_path / "vol", "ep-r01", "pilot", [("p01r1", "100")])
    states["100"] = "RUNNING"
    kinds = [a.kind for a in ops.plan([entry], root)]
    assert kinds == ["copy_receipt", "wait"]

    states["100"] = "COMPLETED"
    kinds = [a.kind for a in ops.plan([entry], root)]
    assert kinds == ["copy_receipt", "harvest", "wait"]  # harvest comes before the gate opens

    ops.apply(ops.plan([entry], root), root)  # copies the receipt and harvests (no result dir)
    status = json.loads((root / "results/ptb/ep-r01/p01r1/status.json").read_text())
    assert status["complete"] is False and status["issues"] == ["result directory not found"]
    (blocked,) = ops.plan([entry], root)
    assert blocked.kind == "blocked" and "did not validate" in blocked.detail

    status["complete"] = True
    (root / "results/ptb/ep-r01/p01r1/status.json").write_text(json.dumps(status))
    (formal,) = ops.plan([entry], root)
    assert (formal.kind, formal.pilot) == ("submit", False)


def test_finished_jobs_are_harvested_once(repo, tmp_path: Path) -> None:
    root, states = repo
    _receipt(tmp_path / "vol", "ep-r01", "formal", [("p01r1", "201"), ("p01r2", "202")])
    states.update({"201": "COMPLETED", "202": "RUNNING"})
    actions = ops.plan([ENTRY], root)
    assert [a.kind for a in actions] == ["copy_receipt", "harvest"]
    assert actions[1].cell == "p01r1" and actions[1].state == "COMPLETED"
    ops.apply(actions, root)
    assert (root / "results/ptb/ep-r01/formal-2026-09-02T000000.json").is_file()
    assert (root / "results/ptb/ep-r01/p01r1/status.json").is_file()
    assert ops.plan([ENTRY], root) == []  # harvested, tracked, and the formal receipt exists
    states["202"] = "FAILED"
    (again,) = ops.plan([ENTRY], root)
    assert (again.kind, again.cell, again.state) == ("harvest", "p01r2", "FAILED")


def test_a_cancelled_entry_cancels_pending_jobs_only(repo, tmp_path: Path, monkeypatch) -> None:
    root, states = repo
    monkeypatch.setattr(ops.ptb, "read_ptb_env", dict)
    _receipt(tmp_path / "vol", "ep-r01", "formal", [("p01r1", "301"), ("p01r2", "302"), ("p01r3", "303")])
    states.update({"301": "PENDING", "302": "RUNNING", "303": "COMPLETED"})
    entry = {**ENTRY, "want": "cancelled", "why": "replaced by ep-r02"}
    actions = ops.plan([entry], root)
    assert [(a.kind, a.cell) for a in actions] == [
        ("copy_receipt", None), ("harvest", "p01r3"), ("cancel", "p01r1"), ("wait", "p01r2")]
    calls: list[list[str]] = []
    monkeypatch.setattr(ops.subprocess, "run", lambda cmd, **kw: (calls.append(list(cmd)) or
                        __import__("subprocess").CompletedProcess(cmd, 0, "", "")))
    lines = ops.apply(actions, root)
    assert calls == [["scancel", "301"]]
    receipt = json.loads((root / "results/ptb/ep-r01/formal-2026-09-02T000000.json").read_text())
    assert receipt["cancellations"][0]["job_id"] == "301"
    assert receipt["cancellations"][0]["reason"] == "replaced by ep-r02"
    assert any(line.startswith("cancel ep-r01/p01r1 job=301 (PENDING)") for line in lines)
    # the next plan does not cancel it again, and still leaves the running one alone
    states["301"] = "CANCELLED"
    kinds = [(a.kind, a.cell) for a in ops.plan([entry], root)]
    assert kinds == [("harvest", "p01r1"), ("wait", "p01r2")]


def test_a_receipt_that_did_not_reach_submitted_blocks(repo, tmp_path: Path) -> None:
    root, _ = repo
    _receipt(tmp_path / "vol", "ep-r01", "formal", [], state="submission_failed")
    actions = ops.plan([ENTRY], root)
    assert [a.kind for a in actions] == ["copy_receipt", "blocked"]
    assert "submission_failed" in actions[1].detail


# ---- apply: submit -----------------------------------------------------------

def test_apply_submits_through_the_launcher_and_tracks_the_receipt(repo, tmp_path: Path, monkeypatch) -> None:
    root, _states = repo
    submitted: list[tuple[str, bool]] = []

    def fake_submit(manifest, *, pilot=False, cell_ids=None):
        submitted.append((manifest["batch_id"], pilot))
        return _receipt(tmp_path / "vol", manifest["batch_id"], "pilot" if pilot else "formal",
                        [("p01r1", "401")] if pilot else [("p01r1", "402"), ("p01r2", "403")])

    monkeypatch.setattr(ops, "submit_batch", fake_submit)
    lines = ops.apply(ops.plan([ENTRY], root), root)
    assert submitted == [("ep-r01", False)]
    assert (root / "results/ptb/ep-r01/formal-2026-09-02T000000.json").is_file()
    assert lines[0].startswith("submit ep-r01: 2 job(s) 402,403")
    log = (root / "results/ptb/ops-log.md").read_text()
    assert "submit ep-r01: 2 job(s)" in log
    assert ops.plan([ENTRY], root) == []


def test_a_dirty_worktree_blocks_submits_but_not_harvests(repo, tmp_path: Path, monkeypatch) -> None:
    root, _states = repo
    monkeypatch.setattr(ops, "_worktree_dirty", lambda repo_root: "?? results/ptb/x")
    monkeypatch.setattr(ops, "submit_batch", lambda *a, **k: pytest.fail("must not submit"))
    lines = ops.apply(ops.plan([ENTRY], root), root)
    assert lines[0].startswith("blocked submit: the worktree is not clean")


def test_ownership_failure_blocks_submits(repo, monkeypatch) -> None:
    root, _states = repo
    monkeypatch.setattr(
        ops, "_submission_ownership_issue", lambda: "OWNERSHIP FAIL: 1 placement violation(s)"
    )
    monkeypatch.setattr(ops, "submit_batch", lambda *a, **k: pytest.fail("must not submit"))

    lines = ops.apply(ops.plan([ENTRY], root), root)

    assert lines == ["blocked submit: OWNERSHIP FAIL: 1 placement violation(s)"]


def test_a_launcher_refusal_is_written_down(repo, monkeypatch) -> None:
    root, _ = repo

    def refuse(manifest, *, pilot=False, cell_ids=None):
        raise ptb.ExperimentError("submission gates failed:\n- missing container: x.sif")

    monkeypatch.setattr(ops, "submit_batch", refuse)
    lines = ops.apply(ops.plan([ENTRY], root), root)
    assert lines[0].startswith("blocked submit ep-r01: submission gates failed")
    assert "missing container" in (root / "results/ptb/ep-r01/blocked.md").read_text()


# ---- harvest -------------------------------------------------------------------

def _fake_result(tmp_path: Path) -> Path:
    result = tmp_path / "results" / "agent_x" / "gsm8k_gemma_555"
    task = result / "task"
    (task / "memory" / "cards").mkdir(parents=True)
    (task / "final_model").mkdir()
    (task / "data").mkdir()
    (task / "skills" / "exp_protocol").mkdir(parents=True)
    (task / ".claude" / "skills").mkdir(parents=True)
    (result / "metrics.json").write_text('{"accuracy": 0.8125, "stderr": 0.01}')
    (result / "runtime_provenance.json").write_text('{"experiment": {"cell_id": "p01r1"}}')
    (result / "time_taken.txt").write_text("36000\n")
    (result / "judgement_general.json").write_text('{"general_anomaly": true}')
    (result / "judgement_api.json").write_text('{"disallowed_api_usage": false}')
    (result / "solve_parsed.txt").write_text("trace line\n" * 1000)
    (result / "solve_out.txt").write_text("{}\n" * 1000)
    (result / "final_eval_1.txt").write_text("eval log\n")
    (task / "memory" / "cards" / "exp-01.yaml").write_text("card_id: exp-01\n")
    (task / "memory" / "index.md").write_text("| exp-01 |\n")
    (task / "awm_sandbox.json").write_text('{"sha": "abc123"}')
    (task / "final_model" / "model.safetensors").write_bytes(b"w" * 10)
    (task / "data" / "big.jsonl").write_bytes(b"x" * (ops.PER_FILE_CAP + 1))
    (task / "data" / "small.jsonl").write_text('{"a": 1}\n')
    (task / "ckpt.bin").write_bytes(b"b" * 3)
    (task / "skills" / "exp_protocol" / "SKILL.md").write_text("skill\n")
    (task / ".claude" / "skills" / "exp_protocol").symlink_to("../../skills/exp_protocol")
    return result


def test_harvest_rejects_a_runtime_node_outside_the_frozen_site(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ops, "audit_result", lambda _result_dir: [])
    result = _fake_result(tmp_path)
    (result / "runtime_provenance.json").write_text(
        '{"experiment": {"cell_id": "p01r1"}, "slurm": {"node": "spill-node"}}'
    )

    status = ops.harvest_job(
        result,
        tmp_path / "bundle",
        batch="batch",
        cell="p01r1",
        job_id="555",
        expected_nodes={"owned-node-0", "owned-node-1"},
    )

    assert status["complete"] is False
    assert status["issues"] == [
        "runtime Slurm node spill-node is outside frozen site nodes owned-node-0,owned-node-1"
    ]


def test_harvest_keeps_the_readable_part_and_lists_the_rest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ops, "audit_result", lambda result_dir: [])
    result = _fake_result(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "job.name-555.out").write_text("".join(f"line {i}\n" for i in range(500)))
    out = tmp_path / "bundle" / "p01r1"
    status = ops.harvest_job(result, out, batch="ep-r01", cell="p01r1", job_id="555",
                             job_name="job.name", state="COMPLETED", slurm_log_dir=logs)
    assert status["accuracy"] == 0.8125 and status["complete"] is True
    assert status["judge_flags"] == ["general_anomaly"] and status["awm_sha"] == "abc123"
    assert (out / "metrics.json").is_file() and (out / "judgement_general.json").is_file()
    assert (out / "final_eval_1.txt").is_file()
    with gzip.open(out / "solve_parsed.txt.gz", "rt") as f:
        assert f.readline() == "trace line\n"
    assert not (out / "solve_out.txt").exists()
    assert (out / "task" / "memory" / "cards" / "exp-01.yaml").is_file()
    assert (out / "task" / "data" / "small.jsonl").is_file()
    assert not (out / "task" / "data" / "big.jsonl").exists()
    assert not (out / "task" / "final_model").exists()
    assert not (out / "task" / "ckpt.bin").exists()
    link = out / "task" / ".claude" / "skills" / "exp_protocol"
    assert link.is_symlink() and (link / "SKILL.md").is_file()
    skipped = {s["path"]: s["reason"] for s in status["skipped"]}
    assert skipped["solve_out.txt"] == "listed only"
    assert skipped["final_model"] == "directory skipped by policy"
    assert skipped["ckpt.bin"] == "binary" and skipped["data/big.jsonl"].startswith("over")
    tail = (out / "slurm.out.tail").read_text().splitlines()
    assert len(tail) == ops.LOG_TAIL_LINES and tail[-1] == "line 499"
    assert json.loads((out / "status.json").read_text()) == status


def test_harvest_without_a_result_dir_records_that(tmp_path: Path) -> None:
    out = tmp_path / "bundle" / "p01r1"
    status = ops.harvest_job(None, out, batch="ep-r01", cell="p01r1", job_id="9", state="NODE_FAIL")
    assert status["issues"] == ["result directory not found"] and status["complete"] is False
    assert sorted(p.name for p in out.iterdir()) == ["status.json"]


def test_harvesting_a_later_attempt_keeps_the_earlier_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ops, "audit_result", lambda result_dir: [])
    result = _fake_result(tmp_path)
    out = tmp_path / "results" / "ptb" / "ep-r01" / "p01r1"
    ops.harvest_job(result, out, batch="ep-r01", cell="p01r1", job_id="555", state="FAILED")
    ops.harvest_job(result, out, batch="ep-r01", cell="p01r1", job_id="556", state="COMPLETED")
    assert json.loads((out / "status.json").read_text())["job_id"] == "556"
    assert (out.parent / "p01r1.j555" / "status.json").is_file()


def test_collect_reads_a_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ops, "audit_result", lambda result_dir: [])
    result = _fake_result(tmp_path)
    shutil.copy(paths.REPO_ROOT / "skills" / "exp_protocol" / "example-card.yaml",
                result / "task" / "memory" / "cards" / "exp-01.yaml")
    out = tmp_path / "results" / "ptb" / "ep-r01" / "p01r1"
    ops.harvest_job(result, out, batch="ep-r01", cell="p01r1", job_id="555", state="COMPLETED")
    from awm.exp_protocol import collect
    (row,) = collect.collect([out / "task"])
    assert row["session"] == "p01r1/task" and row["accuracy"] == 0.8125 and row["n_closed"] == 1


# ---- cancel ---------------------------------------------------------------------

def test_cancel_refuses_anything_but_pending(tmp_path: Path, monkeypatch) -> None:
    receipt = _receipt(tmp_path, "ep-r01", "formal", [("p01r1", "701")])
    monkeypatch.setattr(ops, "job_state", lambda job_id: "RUNNING")
    monkeypatch.setattr(ops.subprocess, "run", lambda *a, **k: pytest.fail("must not scancel"))
    with pytest.raises(ops.OpsError, match="only PENDING"):
        ops.cancel_job(receipt, "p01r1", "why")
    with pytest.raises(ops.OpsError, match="no cell"):
        ops.cancel_job(receipt, "nope", "why")


def test_cancel_uses_sudo_for_root_owned_allocations(monkeypatch) -> None:
    monkeypatch.setattr(ops.ptb, "read_ptb_env", lambda: {"POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT": "1"})
    assert ops._scancel_command("5") == ["sudo", "scancel", "5"]
    monkeypatch.setattr(ops.ptb, "read_ptb_env", dict)
    assert ops._scancel_command("5") == ["scancel", "5"]


# ---- CLI --------------------------------------------------------------------------

def test_reconcile_cli_prints_the_plan_without_apply(repo, monkeypatch, capsys) -> None:
    root, _ = repo
    _queue(root, ENTRY)
    monkeypatch.setattr(ops.paths, "REPO_ROOT", root)
    assert main(["ptb", "reconcile"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("would submit ep-r01")
    assert "awm.ptb_ops" in __import__("sys").modules  # imported only now, by the handler
    _queue(root)
    assert main(["ptb", "reconcile"]) == 0
    assert "nothing to do" in capsys.readouterr().out
