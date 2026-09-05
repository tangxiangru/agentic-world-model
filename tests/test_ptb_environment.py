"""Receipt/source/node-bound synthetic environment operations; never Slurm or models."""

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from awm import paths, ptb_ops, ptb_results, slurm_queue
from awm import ptb_environment as environment
from awm import ptb_experiments as ptb

NODES = {"slurm2-a3nodesetondem-0", "slurm2-a3nodesetondem-1"}
NODE = "slurm2-a3nodesetondem-0"


def manifest():
    data = ptb.load_manifest(paths.REPO_ROOT / "experiments/posttrainbench/exp-protocol-opus48-gsm8k-none-x4-v1.yaml")
    data["batch_id"] = "synthetic-environment"
    data["cells"] = data["cells"][:1]
    data["contract"]["replication"]["repeats"] = 1
    data["operation"] = {"kind": environment.KIND, "target": "humaneval", "walltime": "00:15:00",
                         **{f"{k}_sha256": hashlib.sha256(k.encode()).hexdigest() for k in environment.SOURCE_PATHS}}
    data["placement"] = {"requested_nodes": [NODE]}
    return data


def receipt(tmp_path):
    data = manifest()
    return {**data, "schema_version": 1, "kind": environment.KIND, "state": "held",
            "source": {"top_commit": "a" * 40, "ptb_commit": "b" * 40},
            "site": {"POST_TRAIN_BENCH_SLURM_NODELIST": ",".join(sorted(NODES)),
                     "POST_TRAIN_BENCH_SLURM_RESERVATION": "owned-reservation"},
            "environment_output_root": str(tmp_path / "probe"), "environment_uid": 1234,
            "jobs": [{"job_id": "101", "cell_id": data["cells"][0]["id"], "job_name": "owned.probe"}]}


def synthetic_runtime_source(name):
    return {"bwrap": "/synthetic/bwrap", "interpreter": "/usr/bin/python3.10",
            "files": [["/synthetic/python", "/usr/bin/python3.10", hashlib.sha256(name.encode()).hexdigest()]],
            "bwrap_sha256": "d" * 64, "materialization": None}


@pytest.fixture(autouse=True)
def use_explicit_synthetic_runtime_identities(monkeypatch):
    monkeypatch.setattr(environment, "SOURCE_RUNTIME_SHA256", {
        name: hashlib.sha256(json.dumps(synthetic_runtime_source(name), sort_keys=True).encode()).hexdigest()
        for name in ("opus_5.sif", "vllm_debug.sif")})


def report_fixture(tmp_path):
    """Synthetic serialized native evidence; never execute these invented programs."""
    rec = receipt(tmp_path)
    job = rec["jobs"][0]
    root = environment.report_directory(rec, job)
    root.mkdir(parents=True)
    evidence, profile, probe = environment._native_modules()
    programs = probe.programs()
    metadata = {"prompt": "def arrange(a,b):\n", "entry_point": "arrange",
                "test": "def check(fn):\n    assert fn(9,3)==[3,9]\n"}
    selected = [SimpleNamespace(id=case, input=case, target="synthetic target not executed",
                                metadata=metadata) for case in programs]
    contract = profile.selection_contract(selected)
    images = {}
    for name, digest in {"opus_5.sif": rec["contract"]["container"]["sha256"],
                         "vllm_debug.sif": rec["contract"]["evaluation_container"]["sha256"]}.items():
        base = f"official_eval/{name}/normal/" + ("home/task" if name == "opus_5.sif" else "result")
        source_runtime = synthetic_runtime_source(name)
        source_sha = environment.SOURCE_RUNTIME_SHA256[name]
        materialization = {"source_image": {"sha256": digest}, "source_runtime": source_runtime,
                           "source_runtime_sha256": source_sha}
        transport_sha = hashlib.sha256((json.dumps(materialization, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
        transported = {**source_runtime, "materialization": {"sha256": transport_sha,
                                                             "source_runtime_sha256": source_sha}}
        runtime_sha = hashlib.sha256(json.dumps(transported, sort_keys=True).encode()).hexdigest()
        invocation = {"attempt_id": "environment-native-synthetic", "model": "mockllm/model", "max_tokens": 37,
                      "sandbox_runtime_sha256": runtime_sha,
                      "sandbox_helper_sha256": rec["operation"]["helper_sha256"],
                      "sandbox_limits": profile.EXECUTION_LIMITS}
        rows, samples = [], []
        for item in selected:
            value = "C" if item.id in {"invented-good", "invented-private"} else "I"
            outcome, category = (("timeout", "wall_timeout") if item.id == "invented-timeout" else
                                 (("success", None) if value == "C" else ("program_failure", "program_exit")))
            execution = {"schema": "ptb-python-sandbox-execution-v1", "started": True,
                         "monitor_reaped": True, "descendants_reaped": True, "cleanup_errors": [],
                         "runtime_sha256": invocation["sandbox_runtime_sha256"],
                         "backend_sha256": invocation["sandbox_helper_sha256"], "limits": profile.EXECUTION_LIMITS,
                         "outcome": outcome, "error_category": category, "program_returncode": 0 if value == "C" else 1,
                         "code_sha256": hashlib.sha256((metadata["prompt"] + programs[item.id] + "\n" +
                             metadata["test"] + "\ncheck(" + metadata["entry_point"] + ")").encode()).hexdigest()}
            samples.append({"id": item.id, "epoch": 1, "input": item.input, "target": item.target,
                            "metadata": metadata, "scores": {"verify": {"value": value, "answer": programs[item.id]}},
                            "store": {"ptb_python_execution": [execution]}})
            rows.append({"id": item.id, "score": value, "error": False, "executions": 1, "started": True,
                         "cleanup_complete": True, "outcome": outcome, "error_category": category})
        log = {"status": "success", "eval": {"task": "humaneval", "model": "mockllm/model",
                "config": {"epochs": 1}, "model_generate_config": {"max_tokens": 37},
                "metadata": {"ptb_invocation": invocation, "ptb_selection_sha256": evidence.digest(contract)}},
               "samples": samples, "results": {"total_samples": 5, "completed_samples": 5,
                   "scores": [{"name": "verify", "scorer": "verify", "scored_samples": 5, "unscored_samples": 0,
                               "metrics": {"accuracy": {"value": .4}, "stderr": {"value": (.4*.6/4)**.5}}}]}}
        checked = evidence.validate_log(log, contract, invocation)
        raw = json.dumps(log).encode()
        metrics = {"accuracy": checked["accuracy"], "stderr": checked["stderr"], "official_evidence": {
            "schema_version": 1, "kind": "ptb-humaneval-inspect-v1", "raw_log": ".official-inspect-fixture/inspect.json",
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw), "contract": contract,
            "invocation": invocation, "validated": checked}}
        runtime = {"helper_sha256": invocation["sandbox_helper_sha256"], "runtime_sha256": runtime_sha,
                   "runtime": transported, "source_runtime_sha256": source_sha,
                   "limits": profile.EXECUTION_LIMITS, "materialization": materialization}
        layout = {"role": "scientist" if name == "opus_5.sif" else "official-evaluator",
                  "home": "/home/ben" if name == "opus_5.sif" else "/home/synthetic",
                  "cwd": "/home/ben/task" if name == "opus_5.sif" else "/synthetic/src/eval/tasks/humaneval"}
        image = {"sha256": digest, "cuda": {"count": 1, "names": ["NVIDIA H100 synthetic"], "runtime_uid": 1234},
                 "rows": rows, "native_revalidated": True, "layout": layout,
                 "layout_observed": {k: layout[k] for k in ("home", "cwd")},
                 "normal_execution": {"passed": True, "cleanup_complete": True, "observed_native_sandbox": True},
                 "outer_timeout": {key: True for key in ("timed_out", "cleanup_complete", "passed",
                     "observed_native_sandbox", "admitted_before_timeout", "admitted_sandbox_live_at_timeout",
                     "admission_supervisor_live_at_timeout", "alarm_sent")}, "raw_files": []}
        image["outer_timeout"].update(elapsed_since_admission=8.0, termination_grace_seconds=5)
        observed = {**{k: image[k] for k in ("cuda", "rows", "layout_observed", "native_revalidated")},
                    "dataset": environment.DATASET, "status": "passed", "real_model_called": False,
                    "benchmark_programs_executed": False}
        files = {"probe-result.json": observed, "request.json": {"contract": contract, "invocation": invocation},
                 "python-runtime.json": runtime, "metrics.json": metrics}
        for filename, data in files.items():
            path = root / base / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        path = root / base / ".official-inspect-fixture/inspect.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        for path in sorted((root / base).rglob("*")):
            if path.is_file():
                content = path.read_bytes()
                image["raw_files"].append({"path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
        outer_code = metadata["prompt"] + probe.programs(outer_timeout=True)["invented-outer-timeout"] + "\n" + metadata["test"] + "\ncheck(arrange)"
        event = {"schema": "ptb-python-admission-v1", "started": True, "supervisor_pid": 123,
                 "attested_namespace_pid": 1, "code_sha256": hashlib.sha256(outer_code.encode()).hexdigest()}
        image["outer_timeout"]["admission_event"] = event
        outer_path = root / base.replace("/normal/", "/outer-timeout/") / "outer-admitted.json"
        outer_path.parent.mkdir(parents=True)
        outer_path.write_text(json.dumps({"marker": "PTB_OUTER_ADMITTED_68d273", "observer": "native-supervisor-admission", "event": event}))
        raw = outer_path.read_bytes()
        image["raw_files"].append({"path": outer_path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        images[name] = image
    report = {"schema_version": environment.REPORT_SCHEMA, "job_id": job["job_id"],
              "batch_id": rec["batch_id"], "cell_id": job["cell_id"], "node": NODE, "uid": 1234,
              "source": rec["source"], "probe_sources": {k: rec["operation"][f"{k}_sha256"] for k in environment.SOURCE_PATHS},
              "dataset": environment.DATASET, "images": images, "status": "passed", "errors": [],
              "scientific_result": False, "operation": environment.KIND, "target": "humaneval"}
    (root / "acceptance.json").write_text(json.dumps(report))
    return rec, job, root, report


@pytest.mark.parametrize("change", ["target", "walltime", "hash", "unknown", "pilot", "placement", "foreign"])
def test_manifest_rejects_invalid_operation_or_placement(change):
    data = manifest()
    if change in ("target", "walltime"):
        data["operation"][change] = "invalid"
    elif change == "hash":
        data["operation"]["probe_sha256"] = "unfrozen"
    elif change == "unknown":
        data["operation"]["command"] = "arbitrary"
    elif change == "pilot":
        data["pilot"] = {"cell": data["cells"][0]["id"], "agent_budget_hours": 1}
    elif change == "placement":
        data["placement"] = {"requested_nodes": []}
    else:
        data["placement"] = {"requested_nodes": ["foreign-node"]}
    with pytest.raises((ptb.ExperimentError, environment.EnvironmentError)):
        ptb.validate_manifest(data)
        environment.effective_nodes(data, NODES)


def test_launch_is_short_held_probe_with_explicit_node_and_frozen_spec(tmp_path, monkeypatch):
    data = manifest()
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    launch, = ptb.build_launches(data)
    assert launch.command[-4:] == ("--environment-acceptance", "humaneval", "--walltime", "00:15:00")
    assert "--hold" in launch.command
    assert launch.command[launch.command.index("--nodelist") + 1] == NODE
    assert launch.checkout is None
    assert launch.environment["POST_TRAIN_BENCH_RUN_PURPOSE"] == environment.KIND
    assert launch.environment["POST_TRAIN_BENCH_REQUIRE_COMPLETE"] == "0"
    assert json.loads(launch.environment["POST_TRAIN_BENCH_ENVIRONMENT_ACCEPTANCE_SPEC"])["operation"] == data["operation"]
    assert launch.environment["POST_TRAIN_BENCH_FROZEN_REQUESTED_NODES"] == NODE
    with pytest.raises(ptb.ExperimentError):
        ptb.build_launches(data, purpose="context-smoke")


@pytest.mark.parametrize("change", ["none", "changed", "untracked"])
def test_local_gate_checks_actual_probe_source_bytes(tmp_path, monkeypatch, change):
    data = manifest()
    for name, relative in environment.SOURCE_PATHS.items():
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(name.encode())
    if change == "changed":
        (tmp_path / environment.SOURCE_PATHS["probe"]).write_text("different code")
    monkeypatch.setattr(ptb, "PTB_ROOT", tmp_path)
    monkeypatch.setattr(ptb, "_is_git_tracked", lambda _root, rel: not (
        change == "untracked" and rel == environment.SOURCE_PATHS["probe"]))
    monkeypatch.setattr(ptb, "read_ptb_env", lambda: {"HF_HOME": str(tmp_path),
        "POST_TRAIN_BENCH_SLURM_NODELIST": ",".join(sorted(NODES))})
    monkeypatch.setattr(ptb, "_expanded_nodes", lambda _: NODES)
    issues = ptb.local_issues(data, require_context=False)
    probe_issues = [issue for issue in issues if "environment source" in issue]
    assert bool(probe_issues) is (change != "none")


def test_registry_and_result_consumers_keep_requested_subset(tmp_path, monkeypatch):
    rec = receipt(tmp_path)
    rec["subqueue"] = "gangda_exp-protocol-evolve"
    rec["jobs"][0]["job_name"] = rec["ownership"]["branch"] + ".probe"
    directory = tmp_path / "receipts" / rec["batch_id"]
    directory.mkdir(parents=True)
    file = directory / "environment-acceptance-test.json"
    file.write_text(json.dumps(rec))
    registry_path = tmp_path / "registry.json"
    slurm_queue.register_receipt(file, registry_path=registry_path)
    registry = json.loads(registry_path.read_text())
    assert registry["sources"][0]["jobs"][0]["requested_nodes"] == [NODE]
    monkeypatch.setattr(ptb_results, "_receipts_root", lambda: tmp_path / "receipts")
    monkeypatch.setattr(ptb_results, "_expand_nodelist", lambda _: NODES)
    assert ptb_results._expected_nodes_by_job(rec["batch_id"])["101"] == {NODE}
    monkeypatch.setattr(ptb_ops.subprocess, "run", lambda command, **kwargs:
        subprocess.CompletedProcess(command, 0, "\n".join(sorted(NODES)), ""))
    assert ptb_ops._receipt_expected_nodes(file) == {NODE}
    rec.pop("placement")
    file.write_text(json.dumps(rec))
    assert ptb_results._expected_nodes_by_job(rec["batch_id"])["101"] == NODES
    assert ptb_ops._receipt_expected_nodes(file) == NODES


@pytest.mark.parametrize("registration_fails", [False, True])
def test_submission_always_holds_and_freezes_receipt_before_registry(tmp_path, monkeypatch, registration_fails):
    data = manifest()
    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(ptb, "local_issues", lambda *a, **kw: [])
    monkeypatch.setattr(ptb, "site_issues", list)
    monkeypatch.setattr(ptb, "dry_run", lambda *a, **kw: [])
    monkeypatch.setattr(ptb, "source_snapshot", lambda: {"top_branch": data["ownership"]["branch"],
        "top_commit": "a" * 40, "ptb_commit": "b" * 40, "top_status": "", "ptb_status": ""})
    monkeypatch.setattr(ptb, "read_ptb_env", lambda: {"POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY": str(tmp_path / "registry"),
        "POST_TRAIN_BENCH_SLURM_NODELIST": ",".join(sorted(NODES))})
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        assert "--hold" in command and "--environment-acceptance" in command
        return subprocess.CompletedProcess(command, 0, "Submitted Slurm job 101\n", "")
    monkeypatch.setattr(ptb.subprocess, "run", fake_run)
    def register(path, **kwargs):
        rec = json.loads(path.read_text())
        assert rec["state"] == "held" and rec["jobs"][0]["job_id"] == "101"
        assert rec["placement"] == data["placement"] and rec["kind"] == environment.KIND
        if registration_fails:
            raise slurm_queue.QueueError("synthetic registry error")
    monkeypatch.setattr(slurm_queue, "register_receipt", register)
    if registration_fails:
        with pytest.raises(ptb.ExperimentError, match="remain held"):
            ptb.submit(data)
    else:
        out = ptb.submit(data)
        rec = json.loads(out.read_text())
        assert rec["state"] == "held" and rec["environment_uid"] == os.getuid()
        assert "released_at" not in rec
    assert len(calls) == 1


@pytest.mark.parametrize("drift", [False, True])
def test_release_compares_job_subset_and_full_site_reservation(tmp_path, monkeypatch, drift):
    rec = receipt(tmp_path)
    file = tmp_path / "receipt.json"
    file.write_text(json.dumps(rec))
    monkeypatch.setattr(ptb, "read_ptb_env", lambda: {"POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY": str(tmp_path / "registry")})
    monkeypatch.setattr(slurm_queue, "collect_snapshot", lambda _: {"ownership_ok": True})
    monkeypatch.setattr(ptb, "_job_state", lambda _: "PENDING")
    released = []
    def fake_run(command, **kwargs):
        if command[:3] == ["scontrol", "show", "hostnames"]:
            return subprocess.CompletedProcess(command, 0, command[3].replace(",", "\n"), "")
        if command[:3] == ["scontrol", "show", "reservation"]:
            return subprocess.CompletedProcess(command, 0, "Nodes=" + ",".join(sorted(NODES)), "")
        if command[:3] == ["scontrol", "show", "job"]:
            requested = ",".join(sorted(NODES)) if drift else NODE
            return subprocess.CompletedProcess(command, 0, f"Reason=JobHeldUser ReqNodeList={requested}", "")
        if command[:2] == ["scontrol", "release"]:
            released.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)
    monkeypatch.setattr(ptb.subprocess, "run", fake_run)
    if drift:
        with pytest.raises(ptb.ExperimentError, match="ReqNodeList differs"):
            ptb.release_held(file)
        assert not released and json.loads(file.read_text())["state"] == "held"
    else:
        assert ptb.release_held(file)["state"] == "submitted"
        assert len(released) == 1


def test_operator_does_not_duplicate_environment_receipt(tmp_path, monkeypatch):
    data = manifest()
    rec = receipt(tmp_path)
    file = tmp_path / "environment-acceptance-20260905.json"
    file.write_text(json.dumps(rec))
    monkeypatch.setattr(ptb, "load_manifest", lambda _: data)
    monkeypatch.setattr(ptb_ops, "tracked_receipts", lambda *_: [file])
    monkeypatch.setattr(ptb_ops, "data_receipts", lambda *_: [])
    monkeypatch.setattr(ptb_ops, "job_state", lambda _: "PENDING")
    entries = [{"manifest": "fake.yaml", "want": "submitted", "why": "synthetic environment"}]
    actions = ptb_ops.plan(entries, tmp_path)
    assert [a.kind for a in actions] == ["release"]


@pytest.mark.parametrize("change", ["raw", "source", "node", "uid", "timeout", "rows", "image", "identity", "path"])
def test_acceptance_rejects_unbound_or_incomplete_evidence(tmp_path, change):
    rec, job, root, report = report_fixture(tmp_path)
    assert environment.validate_acceptance(rec, job, site_nodes=NODES, expected_uid=1234)["node"] == NODE
    image = report["images"]["opus_5.sif"]
    if change == "raw":
        (root / image["raw_files"][0]["path"]).write_text("changed")
    elif change == "source":
        report["source"] = {"top_commit": "c" * 40, "ptb_commit": "b" * 40}
    elif change == "node":
        report["node"] = "slurm2-a3nodesetondem-1"
    elif change == "uid":
        report["uid"] = 0
    elif change == "timeout":
        image["outer_timeout"]["cleanup_complete"] = False
    elif change == "rows":
        image["rows"] = image["rows"][:-1]
    elif change == "image":
        image["sha256"] = "c" * 64
    elif change == "identity":
        report["job_id"] = "other"
    else:
        image["raw_files"][0]["path"] = "../escape"
    (root / "acceptance.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, OSError)):
        environment.validate_acceptance(rec, job, site_nodes=NODES, expected_uid=1234)


def test_scoped_readiness_and_harvest_remain_non_scientific(tmp_path, monkeypatch):
    rec, job, _root, _ = report_fixture(tmp_path)
    proposed = deepcopy(rec)
    proposed.pop("operation")
    proposed["contract"]["task"] = "humaneval"
    args = {"site_nodes": NODES, "ptb_commit": "b" * 40, "completed_job_ids": {"101"}}
    assert len(environment.validate_readiness(proposed, rec, **args)) == 1
    with pytest.raises(environment.EnvironmentError, match="unaccepted"):
        environment.validate_readiness(proposed, rec, **{**args, "completed_job_ids": set()})
    proposed.pop("placement")
    with pytest.raises(environment.EnvironmentError, match="unaccepted"):
        environment.validate_readiness(proposed, rec, **args)
    monkeypatch.setattr(ptb, "_expanded_nodes", lambda _: NODES)
    monkeypatch.setattr(ptb_ops, "audit_result", lambda *_a, **_k: pytest.fail("must not invoke scientific validator"))
    status = ptb_ops.harvest_environment(rec, job, tmp_path / "bundle", state="COMPLETED")
    assert status["acceptance_state"] == "passed"
    assert status["scientific_result"] is False and status["complete"] is False and status["eligible"] is False
    assert (tmp_path / "bundle/environment/official_eval/opus_5.sif/normal/home/task/metrics.json").is_file()
    assert ptb_results.discover_attempts(rec) == {job["cell_id"]: []}
    assert "humaneval" not in ptb.APPROVED_TASKS


def test_failed_probe_still_harvests_logs(tmp_path, monkeypatch):
    rec, job, root, report = report_fixture(tmp_path)
    report["status"] = "failed"
    report["errors"] = ["synthetic timeout"]
    (root / "acceptance.json").write_text(json.dumps(report))
    (root / "failed.log").write_text("partial output survives")
    monkeypatch.setattr(ptb, "_expanded_nodes", lambda _: NODES)
    status = ptb_ops.harvest_environment(rec, job, tmp_path / "bundle", state="TIMEOUT")
    assert status["acceptance_state"] == "failed" and not status["eligible"]
    assert (tmp_path / "bundle/environment/failed.log").read_text() == "partial output survives"


@pytest.mark.parametrize("change", ["runtime_uid", "operation", "target", "layout", "layout_observed",
    "admitted_before_timeout", "admitted_sandbox_live_at_timeout", "admission_supervisor_live_at_timeout",
    "alarm_sent", "started", "cleanup_complete", "timeout_as_failure", "unrelated_raw",
    "missing_request", "missing_runtime", "missing_snapshot", "probe_mismatch", "runtime_mismatch",
    "wrong_program", "raw_cleanup", "raw_score", "metrics_binding", "missing_admission", "wrong_admission"])
def test_native_acceptance_rejects_false_positive_evidence(tmp_path, change):
    rec, job, root, report = report_fixture(tmp_path)
    image = report["images"]["opus_5.sif"]
    base = "official_eval/opus_5.sif/normal/home/task/"
    def alter(relative, mutate):
        path = root / base / relative
        data = json.loads(path.read_text())
        mutate(data)
        raw = json.dumps(data).encode()
        path.write_bytes(raw)
        item = next(item for item in image["raw_files"] if item["path"] == base + relative)
        item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return raw
    if change == "runtime_uid":
        image["cuda"]["runtime_uid"] = 0
    elif change in {"operation", "target"}:
        report[change] = "wrong"
    elif change == "layout":
        image["layout"]["role"] = "official-evaluator"
    elif change == "layout_observed":
        image["layout_observed"]["cwd"] = "/wrong"
    elif change in image["outer_timeout"]:
        image["outer_timeout"].pop(change)
    elif change in {"started", "cleanup_complete"}:
        image["rows"][0][change] = False
    elif change == "timeout_as_failure":
        row = next(row for row in image["rows"] if row["id"] == "invented-timeout")
        row.update(outcome="program_failure", error_category="program_exit")
    elif change == "unrelated_raw":
        path = root / "official_eval/opus_5.sif/arbitrary.json"
        path.write_text("{}")
        image["raw_files"] = [{"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(b"{}").hexdigest(), "bytes": 2}]
    elif change.startswith("missing_"):
        ending = {"missing_request": "request.json", "missing_runtime": "python-runtime.json",
                  "missing_snapshot": "inspect.json", "missing_admission": "outer-admitted.json"}[change]
        image["raw_files"] = [item for item in image["raw_files"] if not item["path"].endswith("/" + ending)]
    elif change == "wrong_admission":
        image["outer_timeout"]["admission_event"] = {"schema": "wrong"}
    elif change == "probe_mismatch":
        alter("probe-result.json", lambda data: data.update(native_revalidated=False))
    elif change == "runtime_mismatch":
        alter("python-runtime.json", lambda data: data.update(helper_sha256="0" * 64))
    elif change == "metrics_binding":
        alter("metrics.json", lambda data: data["official_evidence"].update(raw_log="../../other.json"))
    else:
        def mutate(data):
            sample = data["samples"][0]
            if change == "wrong_program":
                sample["scores"]["verify"]["answer"] = "    return []\n"
                metadata = sample["metadata"]
                code = metadata["prompt"] + "    return []\n" + "\n" + metadata["test"] + "\ncheck(arrange)"
                sample["store"]["ptb_python_execution"][0]["code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
            elif change == "raw_cleanup":
                sample["store"]["ptb_python_execution"][0]["descendants_reaped"] = False
            else:
                sample["scores"]["verify"]["value"] = "I"
        raw = alter(".official-inspect-fixture/inspect.json", mutate)
        alter("metrics.json", lambda data: data["official_evidence"].update(
            raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw)))
    (root / "acceptance.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, OSError)):
        environment.validate_acceptance(rec, job, site_nodes=NODES, expected_uid=1234)


def test_slurm_submission_rejects_known_worker_local_source_directories():
    from pathlib import Path
    config = {"POST_TRAIN_BENCH_JOB_SCHEDULER": "slurm"}
    assert environment.worker_source_issues(Path("/tmp/local-repo"), Path("/tmp/local-repo/ptb"), config)
    assert environment.worker_source_issues(Path("/rmeng_data/shared"), Path("/var/tmp/local-ptb"), config)
    assert not environment.worker_source_issues(Path("/rmeng_data/shared"), Path("/rmeng_data/shared/ptb"), config)
    assert not environment.worker_source_issues(Path("/tmp/unit"), Path("/tmp/unit/ptb"), {})


@pytest.mark.parametrize("elapsed,grace", [(30.0, 5), (float("nan"), 5), (8.0, 30)])
def test_inner_deadline_cannot_masquerade_as_outer_cleanup(tmp_path, elapsed, grace):
    rec, job, root, report = report_fixture(tmp_path)
    report["images"]["opus_5.sif"]["outer_timeout"].update(
        elapsed_since_admission=elapsed, termination_grace_seconds=grace)
    (root / "acceptance.json").write_text(json.dumps(report))
    with pytest.raises(environment.EnvironmentError, match="outer timeout"):
        environment.validate_acceptance(rec, job, site_nodes=NODES, expected_uid=1234)
