"""Real synthetic child processes only: no models, scheduler, or benchmark calls."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from exp_protocol_cards import plan_card

from awm import paths
from awm.cli import main
from awm.exp_protocol import execution, lock, schema


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(paths.REPO_ROOT / "skills/exp_protocol"))

    def build(body="print('hello')", fresh=False, env=None, data=False, deferred=False):
        root = tmp_path / "session"
        root.mkdir()
        script = root / "command.py"
        script.write_text(body + "\n")
        card = plan_card()
        card["setup"]["method"] = {"family": "other"}
        card["setup"]["data"] = []
        if data:
            source = root / "data.jsonl"
            source.write_text('{"completion":"ok"}\n')
            card["setup"]["data"] = [{"path": str(source), "source": "local", "n_examples": 1}]
        card["setup"]["command"] = {
            "argv": [sys.executable, str(script)],
            "cwd": str(root),
            "script": str(script),
            "env": env or {},
        }
        card["setup"]["output_dir"] = str(root / "output")
        if fresh:
            card["setup"]["execution"] = {"output_evidence": "fresh-directory"}
        if deferred:
            card["hypothesis"]["expected_effect"] = {"metric": "accuracy"}
            card["evaluation"]["comparator"] = {
                "ref": "base_model",
                "value": None,
                "path": str(root / "parent.json"),
                "defer_validation": True,
            }
        path = root / "memory/cards/exp-01.yaml"
        schema.dump_card(path, card)
        assert main(["exp_protocol", "lock", "--dir", str(root), "exp-01"]) == 0
        return root, path

    return build


def records(root):
    return list((root / "memory/attempts/exp-01").glob("*/finish.json"))


def test_exact_argv_cwd_environment_and_no_automatic_closure(make, monkeypatch):
    monkeypatch.setenv("EXECUTION_TEST_VALUE", "parent")
    root, path = make(
        "import json,os,sys; print(json.dumps([os.getcwd(),sys.argv[1:],"
        "os.environ['EXECUTION_TEST_VALUE'],os.environ['AWM_EXP_ATTEMPT_ID']]))",
        env={"EXECUTION_TEST_VALUE": "child"},
    )
    card = schema.load_card(path)
    card["setup"]["command"]["argv"] += ["literal $NO_EXPANSION", "$(not-a-command)"]
    schema.dump_card(path, card)
    assert (
        main(
            ["exp_protocol", "lock", "--dir", str(root), "exp-01", "--relock", "declare arguments"]
        )
        == 0
    )
    before = path.read_bytes(), lock.lock_path(path).read_bytes()
    handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    result = execution.run_card(path, root)
    attempt = Path(result["attempt_dir"])
    stdout = json.loads((attempt / "stdout.txt").read_text())
    assert stdout == [
        str(root),
        ["literal $NO_EXPANSION", "$(not-a-command)"],
        "child",
        attempt.name,
    ]
    assert os.environ["EXECUTION_TEST_VALUE"] == "parent"
    assert result["observed_returncode"] == result["wrapper_returncode"] == 0
    assert result["artifacts"]["status"] == "unverified"
    assert result["scientific_validation"] == "not_performed"
    assert before == (path.read_bytes(), lock.lock_path(path).read_bytes())
    assert handlers == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    assert len(records(root)) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "no_lock",
        "script_changed",
        "script_missing",
        "data_changed",
        "data_missing",
        "empty_data_hash",
        "wrong_identity",
        "plan_changed",
    ],
)
def test_no_launch_with_incomplete_or_changed_live_evidence(make, defect):
    root, path = make("from pathlib import Path; Path('ran').touch()", data=True)
    if defect == "no_lock":
        lock.lock_path(path).unlink()
    elif defect.startswith("script"):
        script = root / "command.py"
        script.unlink() if defect.endswith("missing") else script.write_text("print('changed')")
    elif defect in ("data_changed", "data_missing"):
        source = root / "data.jsonl"
        source.unlink() if defect.endswith("missing") else source.write_text(
            '{"completion":"changed"}\n'
        )
    elif defect in ("empty_data_hash", "wrong_identity"):
        info = lock.read_lock(path)
        if defect == "empty_data_hash":
            info["data"][0]["sha256"] = ""
        else:
            info["card_id"] = "exp-02"
        lock.lock_path(path).write_text(json.dumps(info))
    else:
        card = schema.load_card(path)
        card["hypothesis"]["claim"] = "changed after locking"
        schema.dump_card(path, card)
    with pytest.raises((execution.ExecutionError, FileNotFoundError)):
        execution.run_card(path, root)
    assert not (root / "ran").exists()


@pytest.mark.parametrize("code", [0, 7])
def test_fresh_output_snapshot_and_child_exit_are_distinct(make, code):
    root, path = make(
        "from pathlib import Path; import sys; "
        f"Path('output/metrics.json').write_text('synthetic'); sys.exit({code})",
        fresh=True,
    )
    result = execution.run_card(path, root)
    assert result["observed_returncode"] == result["wrapper_returncode"] == code
    assert result["artifacts"]["scope"] == "fresh_directory_snapshot"
    assert result["artifacts"]["semantic_validation"] == "not_performed"
    entry = result["artifacts"]["files"][0]
    assert entry["path"] == "metrics.json"
    assert entry["sha256"] == hashlib.sha256(b"synthetic").hexdigest()
    assert result["descendant_completion"] == "not_independently_verified"


def test_existing_output_is_never_deleted_or_certified_as_fresh(make):
    root, path = make("raise AssertionError('must not run')", fresh=True)
    out = root / "output"
    out.mkdir()
    (out / "old").write_text("preserve")
    with pytest.raises(FileExistsError):
        execution.run_card(path, root)
    result = json.loads(records(root)[0].read_text())
    assert result["observed_returncode"] is None and result["child_pid"] is None
    assert (out / "old").read_text() == "preserve"


@pytest.mark.parametrize(
    "body",
    [
        "pass",
        "from pathlib import Path; Path('output/link').symlink_to('../command.py')",
        "from pathlib import Path; p=Path('output'); p.rmdir(); p.symlink_to('.')",
    ],
)
def test_empty_or_symlink_outputs_are_not_verified_even_after_zero_exit(make, body):
    root, path = make(body, fresh=True)
    with pytest.raises(execution.ExecutionError) as caught:
        execution.run_card(path, root)
    assert caught.value.execution_record["observed_returncode"] == 0
    assert caught.value.execution_record["artifacts"]["status"] == "unverified"


def test_inventory_limit_is_an_explicit_incomplete_result(make, monkeypatch):
    root, path = make(
        "from pathlib import Path; Path('output/a').touch(); Path('output/b').touch()", fresh=True
    )
    monkeypatch.setattr(execution, "MAX_OUTPUT_FILES", 1)
    with pytest.raises(execution.ExecutionError, match="limit"):
        execution.run_card(path, root)
    assert json.loads(records(root)[0].read_text())["observed_returncode"] == 0


def test_late_source_mutation_preserves_real_exit_but_fails_integrity(make):
    root, path = make("from pathlib import Path; Path(__file__).write_text('changed')")
    result = execution.run_card(path, root)
    assert result["observed_returncode"] == 0 and result["wrapper_returncode"] == 1
    assert any("changed after the lock" in item for item in result["integrity_after"])


def test_command_may_record_post_result_sections_without_changing_the_locked_plan(make):
    root, path = make(
        "from pathlib import Path; import yaml; "
        "p=Path('memory/cards/exp-01.yaml'); d=yaml.safe_load(p.read_text()); "
        "d['result']={'execution':'completed','measurements':[]}; "
        "p.write_text(yaml.safe_dump(d,sort_keys=False))"
    )
    before_plan = lock.read_lock(path)["plan_sha256"]
    result = execution.run_card(path, root)
    assert result["observed_returncode"] == 0
    assert schema.plan_hash(schema.load_card(path)) == before_plan
    assert result["integrity_after"] == []
    assert result["wrapper_returncode"] == 0


def test_attempt_ids_and_records_are_no_clobber(make, monkeypatch):
    root, path = make()
    monkeypatch.setattr(execution, "uuid4", lambda: SimpleNamespace(hex="fixed-id"))
    first = execution.run_card(path, root)
    saved = records(root)[0].read_bytes()
    with pytest.raises(FileExistsError):
        execution.run_card(path, root)
    assert records(root)[0].read_bytes() == saved
    assert Path(first["attempt_dir"]).stat().st_mode & 0o777 == 0o700


def test_spawn_exception_survives_audit_failure(make, monkeypatch):
    root, path = make()
    card = schema.load_card(path)
    card["setup"]["command"]["argv"][0] = "/nonexistent/execution-test-program"
    schema.dump_card(path, card)
    assert (
        main(
            [
                "exp_protocol",
                "lock",
                "--dir",
                str(root),
                "exp-01",
                "--relock",
                "test missing executable",
            ]
        )
        == 0
    )
    original = execution._write_once

    def fail_finish(target, payload):
        if target.name == "finish.json":
            raise OSError("audit disk failure")
        return original(target, payload)

    monkeypatch.setattr(execution, "_write_once", fail_finish)
    with pytest.raises(FileNotFoundError) as caught:
        execution.run_card(path, root)
    assert caught.value.execution_record["observed_returncode"] is None
    assert caught.value.execution_record["status"] == "observer_failed"


def test_deferred_failure_runs_before_comparator_exists_and_closes_honestly(make):
    root, path = make("raise SystemExit(3)", deferred=True)
    assert main(["exp_protocol", "run", "--dir", str(root), "exp-01"]) == 3
    assert not (root / "parent.json").exists()
    card = schema.load_card(path)
    card["result"] = {"execution": "failed", "failure": "synthetic exit3", "measurements": []}
    card["conclusion"] = {
        "verdict": "inconclusive",
        "decision": "reject",
        "mechanism_verdict": "not_tested",
        "summary": "producer failed",
    }
    schema.dump_card(path, card)
    assert main(["exp_protocol", "close", "--dir", str(root), "exp-01"]) == 0
    assert json.loads(path.with_suffix(".comparator.json").read_text())["outcome"] == "failed"


def test_concurrent_same_card_is_rejected_and_sigterm_retains_real_child_exit(make):
    root, path = make(
        "import time; from pathlib import Path; Path('ready').touch(); time.sleep(30)"
    )
    env = {**os.environ, "PYTHONPATH": str(paths.REPO_ROOT)}
    program = "from awm.cli import main; import sys; raise SystemExit(main(sys.argv[1:]))"
    proc = subprocess.Popen(
        [sys.executable, "-c", program, "exp_protocol", "run", "--dir", str(root), "exp-01"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 8
        while not (root / "ready").exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.02)
        assert (root / "ready").exists()
        with pytest.raises(execution.ExecutionError, match="active guarded"):
            execution.run_card(path, root)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=8) == 143
        result = json.loads(records(root)[0].read_text())
        assert result["status"] == "interrupted" and result["interrupt_signal"] == signal.SIGTERM
        assert result["child_exit_observed_at"]
        assert result["observed_returncode"] == -signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(result["child_pid"], 0)
        # The persistent sidecar is not a stale busy marker: its OS lock is released.
        with execution._claim(path):
            pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=8)
        proc.stderr.close()


def test_observer_sigkill_does_not_make_its_surviving_child_a_safe_retry(make):
    root, path = make(
        "import time; from pathlib import Path; Path('ready').touch(); time.sleep(30)"
    )
    env = {**os.environ, "PYTHONPATH": str(paths.REPO_ROOT)}
    program = "from awm.cli import main; import sys; raise SystemExit(main(sys.argv[1:]))"
    proc = subprocess.Popen(
        [sys.executable, "-c", program, "exp_protocol", "run", "--dir", str(root), "exp-01"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    identity = None
    try:
        deadline = time.monotonic() + 8
        while not (root / "ready").exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.02)
        assert (root / "ready").exists()
        process_file = next((root / "memory/attempts/exp-01").glob("*/process.json"))
        identity = json.loads(process_file.read_text())["identity"]
        proc.kill()
        assert proc.wait(timeout=8) == -signal.SIGKILL
        assert not records(root)
        assert (
            execution._process_identity(identity["pid"])["start_ticks"] == identity["start_ticks"]
        )
        with pytest.raises(
            execution.ExecutionError, match="unresolved prior attempt.*same child birth"
        ):
            execution.run_card(path, root)
        assert len(list((root / "memory/attempts/exp-01").iterdir())) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=8)
        # This exact synthetic process was created by this test, not discovered by name.
        current = execution._process_identity(identity["pid"]) if identity else None
        if current and all(
            current[key] == identity[key] for key in ("pid", "start_ticks", "boot_id")
        ):
            os.killpg(identity["pid"], signal.SIGTERM)


@pytest.mark.parametrize(
    "policy", [[], {}, {"output_evidence": "pretend"}, {"output_evidence": True}]
)
def test_optional_execution_policy_is_validated_before_lock(make, policy):
    root, path = make()
    card = schema.load_card(path)
    card["setup"]["execution"] = policy
    assert not schema.validate_plan(card, root).ok


def test_fresh_preflight_failure_is_not_replaced_by_an_old_passing_lock(make, monkeypatch):
    root, path = make()
    from awm.exp_protocol import preflight

    monkeypatch.setitem(
        preflight.CHECKS,
        "new_failure",
        (
            "synthetic",
            lambda ctx: preflight.CheckResult(
                "new_failure", "fail", "changed external requirement"
            ),
        ),
    )
    with pytest.raises(execution.ExecutionError, match="fresh preflight failed"):
        execution.run_card(path, root)
    assert not records(root)


def test_unknown_prior_record_is_not_completion_even_if_pid_was_recycled(make):
    root, path = make()
    attempt = root / "memory/attempts/exp-01/old"
    attempt.mkdir(parents=True)
    identity = execution._process_identity(os.getpid())
    identity["start_ticks"] -= 1
    (attempt / "process.json").write_text(json.dumps({"identity": identity}))
    with pytest.raises(execution.ExecutionError, match="absent/different"):
        execution.run_card(path, root)
    assert not (attempt / "finish.json").exists()


def test_relative_data_does_not_silently_bind_to_the_run_callers_directory(make, monkeypatch):
    root, path = make(data=True)
    card = schema.load_card(path)
    card["setup"]["data"][0]["path"] = "data.jsonl"
    schema.dump_card(path, card)
    monkeypatch.chdir(root)
    assert (
        main(
            [
                "exp_protocol",
                "lock",
                "--dir",
                str(root),
                "exp-01",
                "--relock",
                "relative-path fixture",
            ]
        )
        == 0
    )
    with pytest.raises(execution.ExecutionError, match="absolute data paths"):
        execution.run_card(path, root)


def test_card_identity_cannot_change_inside_command(make):
    root, path = make(
        "from pathlib import Path; import yaml; p=Path('memory/cards/exp-01.yaml'); "
        "d=yaml.safe_load(p.read_text()); d['card_id']='exp-09'; p.write_text(yaml.safe_dump(d))"
    )
    result = execution.run_card(path, root)
    assert result["observed_returncode"] == 0 and result["wrapper_returncode"] == 1
    assert "card identity changed during command" in result["integrity_after"]
