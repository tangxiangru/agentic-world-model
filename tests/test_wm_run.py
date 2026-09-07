import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.outcome_prediction import wm_agent as agent
from tools.outcome_prediction import wm_evaluate as evaluate
from tools.outcome_prediction import wm_run as runner


def approved(number):
    source_id = f"r0-91/exp-{number:02d}"
    step = {
        "step_id": source_id,
        "role": "training",
        "parents": [],
        "plan": {"setup": {"epochs": 2, "lr": 1e-5}},
        "code": [],
    }
    payload = {
        "schema": "full_recipe_final_only_v1",
        "task": {"benchmark": "gsm8k", "base_model": "synthetic/base", "evaluation_n": 1319},
        "recipe": {"steps": [step], "final_step_id": source_id},
    }
    review = {
        "payload_sha256": evaluate.digest(step),
        "reviewer": "synthetic fixture only",
        "evidence_sha256": "a" * 64,
        "outcome_free": True,
        "executable_recipe_preserved": True,
    }
    return {
        "example_id": source_id,
        "cell_id": "r0-91",
        "model_input": payload,
        "content_review": {
            "status": "approved",
            "payload_sha256": evaluate.digest(payload),
            "step_reviews": {source_id: review},
        },
        "step_sources": {source_id: source_id},
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    model_file = tmp_path / "synthetic-model.bin"
    model_file.write_bytes(b"synthetic fixture, deliberately not a pickle")
    manifest = {
        "train_cell_ids": ["r0-90"],
        "fitted_train_cell_ids": ["r0-90"],
        "forbidden_cell_ids": ["r0-91"],
        "test_data_used": False,
        "training_examples": [{"cell_id": "r0-90"}],
        "empty_train_cell_ids": [],
        "spec": copy.deepcopy(runner.final_model.SPEC),
    }
    monkeypatch.setattr(
        runner.final_model.FinalRecipeWorldModel,
        "load",
        lambda path, expected_sha256: SimpleNamespace(
            training_manifest=copy.deepcopy(manifest), models={"gsm8k": object()}
        ),
    )
    history_root = tmp_path / "history"
    history_root.mkdir()
    (history_root / "raw.txt").write_text("Other TRAIN-run historical trajectory only.\n")
    history = {
        "schema": "wm-disjoint-raw-history-v1",
        "history_run_ids": ["r0-90"],
        "intended_candidate_run_ids": ["r0-91"],
        "files": {"raw.txt": runner.sha((history_root / "raw.txt").read_bytes())},
    }
    executable = tmp_path / "no-inference-executable"
    executable.write_text("synthetic executable metadata fixture")
    runtime = runner.runtime_spec(Path(__file__).parents[1], executable, executable)
    candidates = {"r0-91/exp-01": approved(1), "r0-91/exp-02": approved(2)}
    fp = runner.packet_fingerprints("case-one", candidates, history)
    audit = {
        "schema": "wm-case-packet-audit-v1",
        "case_id": "case-one",
        "status": "approved",
        **{
            k: fp[k]
            for k in ("candidate_payloads_sha256", "evidence_sha256", "history_manifest_sha256")
        },
        "candidate_ancestor_outcomes_excluded": True,
        "test_outcomes_excluded": True,
        "same_evidence_both_arms": True,
        "reviewer": "synthetic packet fixture",
        "review_evidence_sha256": "b" * 64,
    }
    protocol = {
        "schema_version": 1,
        "target": "immediate_official_accuracy",
        "label_source": "archived_checkpoint",
        "failure_policy": "report_missing_no_retry_no_imputation",
        "aggregation": "equal_scientist_run_within_benchmark",
        "wm_model_sha256": runner.sha(model_file.read_bytes()),
        "agent": {
            "model": "fixed-model",
            "effort": "high",
            "max_turns": 5,
            "max_budget_usd": 0.4,
            "timeout_seconds": 30,
            "common_system_sha256": runner.sha(agent.COMMON_SYSTEM.encode()),
            "wm_instruction_sha256": runner.sha(agent.WM_INSTRUCTIONS.encode()),
            "cli_version": "2.1.261",
        },
        "cases": [
            {
                "case_id": "case-one",
                "cell_id": "r0-91",
                "benchmark": "gsm8k",
                "candidate_ids": list(candidates),
                "evidence_sha256": fp["evidence_sha256"],
                "candidate_payloads_sha256": fp["candidate_payloads_sha256"],
                "packet_audit_sha256": evaluate.digest(audit),
            }
        ],
    }
    return {
        "schema": runner.SOURCE_SCHEMA,
        "protocol": protocol,
        "split": {"train_cell_ids": ["r0-90"], "test_cell_ids": ["r0-91"]},
        "runtime": runtime,
        "model": {"path": str(model_file), "sha256": protocol["wm_model_sha256"]},
        "history": {"root": str(history_root), "manifest": history},
        "cases": {"case-one": {"candidate_payloads": candidates, "packet_audit": audit}},
    }


def packed(source, tmp_path):
    return runner.package(source, tmp_path / "bundle")


def stream(arm):
    decision = {
        "choice": "r0-91/exp-01",
        "ranking": ["r0-91/exp-01", "r0-91/exp-02"],
        "confidence": 0.5,
        "rationale": "Synthetic decision; no inference.",
    }
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session",
            "claude_code_version": "2.1.261",
            "model": "fixed-model",
            "permissionMode": "dontAsk",
            "skills": [],
            "plugins": [],
            "agents": [],
            "slash_commands": [],
            "tools": sorted(evaluate.allowed_tools(arm)),
            "mcp_servers": [{"name": "evidence", "status": "connected"}],
        },
        {
            "type": "assistant",
            "session_id": "session",
            "parent_tool_use_id": None,
            "message": {
                "role": "assistant",
                "model": "fixed-model",
                "content": [{"type": "text", "text": json.dumps(decision)}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "session_id": "session",
            "is_error": False,
            "terminal_reason": "completed",
            "stop_reason": "end_turn",
            "num_turns": 1,
            "result": json.dumps(decision),
            "total_cost_usd": 0.1,
            "permission_denials": [],
            "modelUsage": {"fixed-model": {"costUSD": 0.1}},
        },
    ]
    return b"\n".join(json.dumps(e).encode() for e in events) + b"\n"


@pytest.fixture
def launches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner,
        "auth_metadata",
        lambda executable: {
            "cli_version": "2.1.261",
            "oauth_logged_in": True,
            "model_connectivity_tested": False,
        },
    )

    def fake(spec, directory):
        arm = directory.name
        calls.append((copy.deepcopy(spec), directory))
        (directory / "transcript.jsonl").write_bytes(stream(arm))
        (directory / "stderr.log").write_bytes(b"")
        return {
            "returncode": 0,
            "elapsed_seconds": 0.25,
            "timed_out": False,
            "runner_failure": None,
        }

    monkeypatch.setattr(runner, "_launch", fake)
    return calls


def run_bundle(bundle, tmp_path, **options):
    return runner.execute(
        bundle["bundle_path"],
        expected_sha256=bundle["bundle_sha256"],
        output_dir=tmp_path / "captures",
        authorized=True,
        **options,
    )


def test_packager_has_exact_shared_candidate_and_history_bytes(source, tmp_path):
    bundle = packed(source, tmp_path)
    value = runner.validate_bundle(bundle["bundle_path"], expected_sha256=bundle["bundle_sha256"])
    config_a = runner._server_config(
        value, Path(bundle["bundle_path"]).parent, "case-one", "rpm", tmp_path / "a"
    )
    config_b = runner._server_config(
        value, Path(bundle["bundle_path"]).parent, "case-one", "rpm_wm", tmp_path / "b"
    )
    assert set(config_a) == {"evidence_root", "files", "audit_log"}
    assert set(config_b) - set(config_a) == {"model_path", "model_sha256", "candidate_payloads"}
    assert config_a["files"] == config_b["files"]
    assert config_a["evidence_root"] == config_b["evidence_root"]
    assert len(config_a["files"]) == 3
    assert not Path(bundle["bundle_path"]).stat().st_mode & 0o222
    with pytest.raises(FileExistsError):
        packed(source, tmp_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["cases"]["case-one"]["candidate_payloads"]["r0-91/exp-01"].update(
            label={"accuracy": 0.9}
        ),
        lambda s: s["cases"]["case-one"]["candidate_payloads"]["r0-91/exp-01"][
            "content_review"
        ].update(status="needs_review"),
        lambda s: s["cases"]["case-one"]["packet_audit"].update(
            candidate_ancestor_outcomes_excluded=False
        ),
        lambda s: s["protocol"]["cases"][0].update(evidence_sha256="0" * 64),
        lambda s: s["protocol"].update(wm_model_sha256="0" * 64),
        lambda s: s["history"]["manifest"].update(history_run_ids=["r0-91"]),
        lambda s: s["cases"]["case-one"]["candidate_payloads"]["r0-91/exp-01"][
            "step_sources"
        ].update({"r0-91/exp-01": "r0-90/exp-01"}),
    ],
)
def test_unapproved_or_mismatched_inputs_fail_before_inference(source, tmp_path, launches, mutate):
    mutate(source)
    with pytest.raises((ValueError, TypeError)):
        packed(source, tmp_path)
    assert not launches


def test_frozen_model_identity_cannot_change_after_packaging(source, tmp_path, launches):
    bundle = packed(source, tmp_path)
    path = Path(bundle["bundle_path"])
    changed = json.loads(path.read_text())
    changed["protocol"]["agent"]["model"] = "other-model"
    path.chmod(0o600)
    path.write_bytes(runner.encoded(changed))
    with pytest.raises(ValueError):
        run_bundle(bundle, tmp_path)
    assert not launches


def test_candidate_identity_swap_rejected_even_with_recomputed_packet_hashes(source, tmp_path):
    entry = source["cases"]["case-one"]
    candidates = entry["candidate_payloads"]
    candidates["r0-91/exp-01"], candidates["r0-91/exp-02"] = (
        candidates["r0-91/exp-02"],
        candidates["r0-91/exp-01"],
    )
    fp = runner.packet_fingerprints("case-one", candidates, source["history"]["manifest"])
    for key in ("candidate_payloads_sha256", "evidence_sha256"):
        entry["packet_audit"][key] = fp[key]
        source["protocol"]["cases"][0][key] = fp[key]
    source["protocol"]["cases"][0]["packet_audit_sha256"] = evaluate.digest(entry["packet_audit"])
    with pytest.raises(ValueError, match="archive target"):
        packed(source, tmp_path)


def test_success_normalizes_and_resumes_without_any_hidden_relaunch(source, tmp_path, launches):
    bundle = packed(source, tmp_path)
    runs = run_bundle(bundle, tmp_path)
    assert len(launches) == 2 and all(r["status"] == "success" for r in runs)
    assert launches[0][0]["stdin"] == launches[1][0]["stdin"]
    assert [r["cost_usd"] for r in runs] == [0.1, 0.1]
    again = run_bundle(bundle, tmp_path)
    assert again == runs and len(launches) == 2
    for result in runs:
        assert (
            evaluate.validate_run(
                result, source["protocol"], source["protocol"]["cases"][0], result["arm"]
            )
            is None
        )


def test_no_execution_without_explicit_authorization(source, tmp_path, launches):
    bundle = packed(source, tmp_path)
    with pytest.raises(PermissionError):
        runner.execute(
            bundle["bundle_path"],
            expected_sha256=bundle["bundle_sha256"],
            output_dir=tmp_path / "captures",
        )
    assert not launches


def test_interrupted_slot_blocks_all_new_work_without_retry(source, tmp_path, launches):
    bundle = packed(source, tmp_path)
    run_bundle(bundle, tmp_path)
    result = tmp_path / "captures" / runner.sha(b"case-one") / "rpm" / "result.json"
    result.rename(result.with_name("interrupted-for-fixture.json"))
    with pytest.raises(ValueError, match="Interrupted claimed"):
        run_bundle(bundle, tmp_path)
    assert len(launches) == 2


def test_timeout_is_preserved_never_retried(source, tmp_path, launches, monkeypatch):
    def timeout(spec, directory):
        launches.append((spec, directory))
        (directory / "transcript.jsonl").write_bytes(b"")
        (directory / "stderr.log").write_bytes(b"partial")
        return {"returncode": -15, "elapsed_seconds": 30, "timed_out": True, "runner_failure": None}

    monkeypatch.setattr(runner, "_launch", timeout)
    bundle = packed(source, tmp_path)
    runs = run_bundle(bundle, tmp_path)
    assert all(r["status"] == "timeout" and r["cost_usd"] is None for r in runs)
    run_bundle(bundle, tmp_path)
    assert len(launches) == 2


@pytest.mark.parametrize(
    "filename",
    [
        "transcript.jsonl",
        "audit.jsonl",
        "stderr.log",
        "server.json",
        "started.json",
        "capture.json",
    ],
)
def test_changed_capture_bytes_rejected_on_resume(source, tmp_path, launches, filename):
    bundle = packed(source, tmp_path)
    run_bundle(bundle, tmp_path)
    path = launches[0][1] / filename
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run_bundle(bundle, tmp_path)
    assert len(launches) == 2


def test_evidence_mutation_and_symlink_are_rejected(source, tmp_path, launches):
    bundle = packed(source, tmp_path)
    path = Path(bundle["bundle_path"]).parent / "evidence/history/raw.txt"
    path.chmod(0o600)
    path.write_text("changed")
    with pytest.raises(ValueError):
        run_bundle(bundle, tmp_path)
    assert not launches
    original = tmp_path / "different.txt"
    original.write_text("different")
    path.unlink()
    path.symlink_to(original)
    with pytest.raises(ValueError):
        run_bundle(bundle, tmp_path)
    assert not launches


def test_model_manifest_train_only_boundary_is_checked(source, tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.final_model.FinalRecipeWorldModel,
        "load",
        lambda *a, **k: SimpleNamespace(
            training_manifest={
                "train_cell_ids": ["r0-90"],
                "fitted_train_cell_ids": ["r0-90"],
                "forbidden_cell_ids": ["r0-91"],
                "test_data_used": True,
                "training_examples": [],
                "empty_train_cell_ids": [],
            }
        ),
    )
    with pytest.raises(ValueError, match="TRAIN-only"):
        packed(source, tmp_path)


@pytest.mark.parametrize("field", ["python_sha256", "claude_sha256"])
def test_runtime_binary_hashes_are_checked_before_inference(source, tmp_path, launches, field):
    source["runtime"][field] = "0" * 64
    with pytest.raises(ValueError):
        packed(source, tmp_path)
    assert not launches


def test_packet_audit_is_mandatory_and_not_created_by_packager(source, tmp_path):
    source["cases"]["case-one"].pop("packet_audit")
    with pytest.raises(ValueError, match="case bundle"):
        packed(source, tmp_path)


def test_no_second_inference_after_frozen_evidence_changes(source, tmp_path, launches, monkeypatch):
    bundle = packed(source, tmp_path)
    original_launch = runner._launch

    def changed(spec, directory):
        capture = original_launch(spec, directory)
        path = Path(bundle["bundle_path"]).parent / "evidence/history/raw.txt"
        path.chmod(0o600)
        path.write_text("changed between arms")
        return capture

    monkeypatch.setattr(runner, "_launch", changed)
    with pytest.raises(ValueError):
        run_bundle(bundle, tmp_path)
    assert len(launches) == 1


@pytest.mark.parametrize("filename", ["transcript.jsonl", "audit.jsonl", "stderr.log"])
def test_oversized_capture_fails_closed_and_pins_full_bytes(
    source, tmp_path, launches, monkeypatch, filename
):
    monkeypatch.setitem(runner.CAPTURE_LIMITS, filename, 1)
    original_launch = runner._launch

    def large(spec, directory):
        capture = original_launch(spec, directory)
        if filename != "transcript.jsonl":
            (directory / filename).write_bytes(b"oversized private bytes")
        return capture

    monkeypatch.setattr(runner, "_launch", large)
    bundle = packed(source, tmp_path)
    runs = run_bundle(bundle, tmp_path)
    assert all(
        r["status"] == "invalid" and r["normalization"]["failure_reason"] == "capture_size_limit"
        for r in runs
    )
    for run, (_, directory) in zip(runs, launches, strict=True):
        assert run["transcript_sha256"] == runner.sha((directory / "transcript.jsonl").read_bytes())
        assert "private bytes" not in json.dumps(run)
    run_bundle(bundle, tmp_path)
    assert len(launches) == 2


def test_missing_capture_fingerprint_cannot_hide_server_config_tampering(
    source, tmp_path, launches
):
    bundle = packed(source, tmp_path)
    run_bundle(bundle, tmp_path)
    path = launches[0][1] / "result.json"
    result = json.loads(path.read_text())
    result["capture_sha256"].pop("server.json")
    path.chmod(0o600)
    path.write_bytes(runner.encoded(result))
    with pytest.raises(ValueError, match="capture fingerprints"):
        run_bundle(bundle, tmp_path)
    assert len(launches) == 2


def test_missing_oauth_metadata_launches_nothing(source, tmp_path, launches, monkeypatch):
    monkeypatch.setattr(
        runner,
        "auth_metadata",
        lambda executable: {
            "cli_version": "2.1.261",
            "oauth_logged_in": False,
            "model_connectivity_tested": False,
        },
    )
    bundle = packed(source, tmp_path)
    with pytest.raises(ValueError, match="OAuth"):
        run_bundle(bundle, tmp_path)
    assert not launches


def test_environment_excludes_credentials_endpoint_and_hook_injection(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-secret-do-not-copy")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://untrusted.invalid")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/untrusted/config")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    env = runner.execution_environment()
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CONFIG_DIR", "PYTHONPATH"} & set(
        env
    )
    assert env["CLAUDE_CODE_MAX_RETRIES"] == "0"
    assert env["DISABLE_AUTO_COMPACT"] == "1"
    assert "synthetic-secret" not in json.dumps(env)


def test_auth_probe_never_exposes_secret_or_account_values(monkeypatch):
    outputs = [
        SimpleNamespace(stdout="2.1.261 (Claude Code)\n", returncode=0),
        SimpleNamespace(
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "email": "private@example.invalid",
                    "token": "synthetic-secret",
                }
            ).encode(),
            returncode=0,
        ),
    ]
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: outputs.pop(0))
    assert runner.auth_metadata("/fixture/claude") == {
        "cli_version": "2.1.261",
        "oauth_logged_in": True,
        "model_connectivity_tested": False,
    }


def test_low_level_launcher_times_out_and_terminates_exactly_one_process(tmp_path, monkeypatch):
    (tmp_path / "workspace").mkdir()
    calls = []

    class FakeProcess:
        pid = 12345
        returncode = None

        def communicate(self, *args, **kwargs):
            calls.append("communicate")
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("synthetic", 1)
            self.returncode = -15

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: FakeProcess())
    signals = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, signal: signals.append((pid, signal)))
    clock = iter([0, 0, 2])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock, 2))
    capture = runner._launch(
        {"argv": ["synthetic"], "stdin": "prompt", "timeout_seconds": 1}, tmp_path
    )
    assert capture["timed_out"] and len(signals) == 1 and len(calls) == 2
