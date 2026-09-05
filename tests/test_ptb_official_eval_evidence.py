"""Prospective archive helper tests: synthetic files only, no scheduler/model calls."""

import fcntl
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/ptb_official_eval_evidence.py"
spec = importlib.util.spec_from_file_location("official_evidence", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def payload(status="success"):
    return {
        "status": status,
        "eval": {"model": "vllm/SYNTHETIC/final_model", "dataset": {"name": "SYNTHETIC", "samples": 1319},
                 "model_args": {"gpu_memory_utilization": .3},
                 "model_generate_config": {"max_connections": 2, "max_tokens": 4000}},
        "results": {"total_samples": 2, "completed_samples": 2,
                    "scores": [{"name": "match", "scored_samples": 2, "unscored_samples": 0}]},
        "samples": [
            {"id": i, "epoch": 1, "input": f"SYNTHETIC {i}", "target": "toy",
             "events": [{"event": "model", "input": [{"role": "user", "content": f"SYNTHETIC {i}"}]}],
             "scores": {"match": {"value": value}},
             "output": {"completion": f"toy {i}", "choices": [{"stop_reason": "stop"}],
                        "usage": {"input_tokens": 3, "output_tokens": 2}}}
            for i, value in enumerate(("C", "I"))
        ],
    }


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "job scratch" / "official-eval" / "attempt-0001"
    source.mkdir(parents=True)
    result = tmp_path / "result space"
    result.mkdir()
    provenance = {"experiment": {"official_log_retention": ev.MODE, "task": "gsm8k",
                                 "batch_id": "SYNTHETIC", "cell_id": "toy01", "run_purpose": "synthetic-fixture"},
                  "source": {"top_commit": "a" * 40, "ptb_commit": "b" * 40},
                  "slurm": {"job_id": "901"}, "evaluation_container": {"sha256": "c" * 64}}
    (result / "runtime_provenance.json").write_text(json.dumps(provenance))
    (result / "metrics.json").write_text('{"accuracy":0.5,"SYNTHETIC":true}\n')
    return source, result


def preserve(setup, **kwargs):
    source, result = setup
    options = dict(job_id="901", attempt=1, phase="post_attempt", exit_code=0)
    options.update(kwargs)
    return ev.preserve_attempt(source, result, **options)


def receipt_dir(setup, attempt=1):
    return setup[1] / "official_eval" / f"attempt-{attempt:04d}"


def write_log(setup, value=None, name="toy.json"):
    path = setup[0] / name
    path.write_text(json.dumps(payload() if value is None else value))
    return path


def test_roundtrip_and_idempotency_do_not_change_scientific_metrics(setup):
    log = write_log(setup)
    original = log.read_bytes()
    metrics = (setup[1] / "metrics.json").read_bytes()
    receipt = preserve(setup)
    record = receipt["logs"][0]
    assert receipt["archive_status"] == "preserved"
    assert "complete" not in receipt and "accuracy" not in receipt
    assert record["source_sha256"] == ev.hashlib.sha256(original).hexdigest()
    assert gzip.decompress((receipt_dir(setup) / record["archive"]["file"]).read_bytes()) == original
    compact = record["compact"]
    assert compact["status"] == "available"
    rows = [json.loads(line) for line in gzip.decompress((receipt_dir(setup) / compact["file"]).read_bytes()).splitlines()]
    assert len(rows) == 2 and [row["scores"]["match"] for row in rows] == ["C", "I"]
    assert rows[0]["input_target_sha256"] == ev.value_hash({"input": "SYNTHETIC 0", "target": "toy"})
    assert compact["metadata"]["counts"]["completed_samples"] == 2
    assert compact["metadata"]["dataset"]["samples"] == 1319  # never confuse population with actual n
    assert compact["metadata"]["generation_config"]["seed"] is None
    assert preserve(setup) == receipt
    assert preserve(setup, phase="cleanup", exit_code=None) == receipt
    setup[0].rename(setup[0].with_name("original-source-preserved"))
    assert preserve(setup, phase="cleanup", exit_code=None) == receipt
    assert (setup[1] / "metrics.json").read_bytes() == metrics


@pytest.mark.parametrize("status", ["error", "cancelled", "started"])
def test_partial_attempt_keeps_counts_and_unknown_output(setup, status):
    doc = payload(status)
    doc["samples"] = [{"id": 0, "epoch": 1}]
    doc["results"]["completed_samples"] = 1
    write_log(setup, doc)
    r = preserve(setup, exit_code=124)
    c = r["logs"][0]["compact"]
    assert c["status"] == "available" and c["metadata"]["source_status"] == status
    assert c["metadata"]["observed_sample_rows"] == 1
    rows = gzip.decompress((receipt_dir(setup) / c["file"]).read_bytes()).splitlines()
    row = json.loads(rows[0])
    assert row["input_target_sha256"] is None and row["completion_sha256"] is None
    assert row["request_role_content_sha256"] is None
    assert r["evaluator_exit_code"] == 124


@pytest.mark.parametrize("raw", [b'{"partial":', b'[]', b'{"status":"error"}', b'not json'])
def test_unparseable_or_missing_samples_keeps_raw_bytes(setup, raw):
    (setup[0] / "bad.json").write_bytes(raw)
    r = preserve(setup, exit_code=1)
    item = r["logs"][0]
    assert item["compact"]["status"] == "unavailable"
    assert gzip.decompress((receipt_dir(setup) / item["archive"]["file"]).read_bytes()) == raw


def test_no_log_and_cleanup_do_not_claim_latest_metrics_belong_to_attempt(setup):
    r = preserve(setup, phase="cleanup", exit_code=None)
    assert r["archive_status"] == "no_log" and r["logs"] == []
    assert r["metrics_observed_after_attempt"] is None
    assert r["evaluator_exit_code"] is None


def test_multiple_logs_and_retries_are_independent(setup):
    write_log(setup)
    write_log(setup, payload("error"), "second.json")
    first = preserve(setup, exit_code=1)
    second_source = setup[0].with_name("attempt-0002")
    second_source.mkdir()
    (second_source / "toy.json").write_text(json.dumps(payload()))
    second = ev.preserve_attempt(second_source, setup[1], job_id="901", attempt=2, phase="post_attempt", exit_code=0)
    assert len(first["logs"]) == 2 and len(second["logs"]) == 1
    assert first["attempt"] == 1 and second["attempt"] == 2
    assert (receipt_dir(setup) / "receipt.json").is_file()


@pytest.mark.parametrize("mutation", ["no_opt_in", "wrong_task", "bad_source", "bad_image", "not_mapping"])
def test_invalid_provenance_refuses_before_destination_creation(setup, mutation):
    p = setup[1] / "runtime_provenance.json"
    doc = json.loads(p.read_text())
    if mutation == "no_opt_in": del doc["experiment"]["official_log_retention"]
    elif mutation == "wrong_task": doc["experiment"]["task"] = "aime2025"
    elif mutation == "bad_source": doc["source"]["ptb_commit"] = None
    elif mutation == "bad_image": doc["evaluation_container"]["sha256"] = None
    else: doc = []
    p.write_text(json.dumps(doc))
    with pytest.raises(ev.RetentionError): preserve(setup)
    assert not (setup[1] / "official_eval").exists()


@pytest.mark.parametrize("options", [{"job_id": "902"}, {"attempt": 2}, {"attempt": 0},
                                    {"phase": "cleanup", "exit_code": 0}, {"exit_code": None}])
def test_wrong_identity_or_exit_observation_rejected(setup, options):
    with pytest.raises(ev.RetentionError): preserve(setup, **options)


@pytest.mark.parametrize("where", ["source_file", "source_dir", "source_parent", "destination", "metrics"])
def test_symlinks_never_followed(setup, tmp_path, where):
    source, result = setup
    victim = tmp_path / "victim"
    victim.write_text("preserve me")
    if where == "source_file": (source / "toy.json").symlink_to(victim)
    elif where == "source_dir":
        saved = source.with_name("saved")
        source.rename(saved)
        source.symlink_to(saved, target_is_directory=True)
    elif where == "source_parent":
        saved = source.parent.with_name("saved-parent")
        source.parent.rename(saved)
        source.parent.symlink_to(saved, target_is_directory=True)
    elif where == "destination": (result / "official_eval").symlink_to(tmp_path, target_is_directory=True)
    else:
        (result / "metrics.json").unlink()
        (result / "metrics.json").symlink_to(victim)
    with pytest.raises((ev.RetentionError, OSError)): preserve(setup)
    assert victim.read_text() == "preserve me"
    assert not (receipt_dir(setup) / "receipt.json").exists()


def test_fifo_json_refused_without_opening(setup):
    os.mkfifo(setup[0] / "pipe.json")
    with pytest.raises(ev.RetentionError, match="non-file"): preserve(setup)


def test_collision_is_not_overwritten(setup):
    write_log(setup)
    out = receipt_dir(setup)
    out.mkdir(parents=True)
    target = out / "toy.json.gz"
    target.write_bytes(b"previous unrelated bytes")
    with pytest.raises(ev.RetentionError, match="differs"): preserve(setup)
    assert target.read_bytes() == b"previous unrelated bytes"
    assert not (out / "receipt.json").exists()


@pytest.mark.parametrize("change", ["source_bytes", "new_log", "archive", "receipt_identity", "exit_code"])
def test_edits_after_finalization_are_detected(setup, change):
    log = write_log(setup)
    r = preserve(setup)
    out = receipt_dir(setup)
    if change == "source_bytes": log.write_text("changed")
    elif change == "new_log": write_log(setup, name="late.json")
    elif change == "archive": (out / r["logs"][0]["archive"]["file"]).write_bytes(b"changed")
    elif change == "receipt_identity":
        r["identity"]["cell_id"] = "different"
        (out / "receipt.json").write_text(json.dumps(r))
    with pytest.raises(ev.RetentionError): preserve(setup, **({"exit_code": 1} if change == "exit_code" else {}))


def test_compaction_cap_preserves_large_raw_log(setup):
    doc = payload()
    doc["padding"] = "x" * (3 * 1024 * 1024)
    log = write_log(setup, doc)
    r = preserve(setup, compact_max_bytes=1024)
    item = r["logs"][0]
    assert item["source_bytes"] == log.stat().st_size
    assert item["compact"]["status"] == "unavailable"
    assert gzip.decompress((receipt_dir(setup) / item["archive"]["file"]).read_bytes()) == log.read_bytes()


@pytest.mark.parametrize("failure", ["timeout", "signal"])
def test_compaction_resource_failure_does_not_lose_raw(setup, monkeypatch, failure):
    write_log(setup)
    def run(command, **kwargs):
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, -9, "", "")
    monkeypatch.setattr(ev.subprocess, "run", run)
    r = preserve(setup)
    assert r["archive_status"] == "preserved"
    assert r["logs"][0]["compact"]["status"] == "unavailable"
    assert not list(receipt_dir(setup).glob("*.samples.jsonl.gz"))


def test_failed_receipt_write_is_resumable_without_claiming_completion(setup, monkeypatch):
    log = write_log(setup)
    original = ev.write_atomic
    def fail(*_): raise OSError("SYNTHETIC disk error")
    monkeypatch.setattr(ev, "write_atomic", fail)
    with pytest.raises(OSError): preserve(setup)
    assert log.is_file() and not (receipt_dir(setup) / "receipt.json").exists()
    monkeypatch.setattr(ev, "write_atomic", original)
    assert preserve(setup)["archive_status"] == "preserved"
    assert not list(receipt_dir(setup).glob(".retention-*.tmp"))


def test_attempt_lock_prevents_concurrent_writers(setup):
    out = receipt_dir(setup)
    out.mkdir(parents=True)
    with (out / ".archive.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ev.RetentionError, match="another archiver"): preserve(setup)


def test_cli_runs_on_cpu_with_space_paths(setup):
    write_log(setup)
    proc = subprocess.run([sys.executable, str(SCRIPT), "--source-dir", str(setup[0]),
                           "--result-dir", str(setup[1]), "--job-id", "901", "--attempt", "1",
                           "--phase", "post_attempt", "--exit-code", "0"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["logs"][0]["compact"]["status"] == "available"


def test_source_change_during_copy_never_gets_a_receipt(setup, monkeypatch):
    log = write_log(setup)
    original = ev.open_regular
    class ChangingReader:
        def __enter__(self):
            self.stream = original(log)
            self.changed = False
            return self
        def __exit__(self, *_): self.stream.close()
        def fileno(self): return self.stream.fileno()
        def read(self, size):
            data = self.stream.read(size)
            if not self.changed:
                self.changed = True
                with log.open("ab") as extra: extra.write(b"late writer")
            return data
    monkeypatch.setattr(ev, "open_regular", lambda path: ChangingReader() if Path(path) == log else original(path))
    with pytest.raises(ev.RetentionError, match="changed while archiving"): preserve(setup)
    assert log.read_bytes().endswith(b"late writer")
    assert not (receipt_dir(setup) / "receipt.json").exists()
    assert not list(receipt_dir(setup).glob(".retention-*.tmp"))


def test_receipt_path_traversal_is_rejected(setup):
    write_log(setup)
    r = preserve(setup)
    r["logs"][0]["archive"]["file"] = "../../metrics.json"
    (receipt_dir(setup) / "receipt.json").write_text(json.dumps(r))
    with pytest.raises(ev.RetentionError, match="unsafe artifact"): preserve(setup)


def test_malformed_receipt_is_not_accepted(setup):
    write_log(setup)
    preserve(setup)
    (receipt_dir(setup) / "receipt.json").write_text("[]")
    with pytest.raises(ev.RetentionError, match="malformed existing receipt"): preserve(setup)


def test_real_memory_limit_keeps_raw_when_compaction_cannot_run(setup):
    write_log(setup)
    r = preserve(setup, compact_memory_mb=1)
    assert r["archive_status"] == "preserved"
    assert r["logs"][0]["compact"]["status"] == "unavailable"


def test_archiver_cpu_fixture_in_pinned_evaluation_image(setup):
    executable = Path("/rmeng_data/robtang/tools/apt-root/usr/bin/apptainer")
    image = Path("/rmeng_data/robtang/ptb-containers/vllm_debug.sif")
    if not executable.is_file() or not image.is_file():
        pytest.skip("optional site image unavailable")
    log = write_log(setup)
    # Pure file fixture: no --nv, evaluator, real model or network invocation.
    root = setup[1].parent
    source = Path("/fixture") / setup[0].relative_to(root)
    result = Path("/fixture") / setup[1].relative_to(root)
    proc = subprocess.run([
        str(executable), "exec", "--cleanenv", "--no-home", "--containall",
        "--bind", f"{SCRIPT}:/opt/ptb-evidence.py:ro", "--bind", f"{root}:/fixture",
        str(image), "python", "/opt/ptb-evidence.py", "--source-dir", str(source),
        "--result-dir", str(result), "--job-id", "901", "--attempt", "1",
        "--phase", "post_attempt", "--exit-code", "0",
    ], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout)["logs"][0]
    assert record["compact"]["status"] == "available"
    assert gzip.decompress((receipt_dir(setup) / record["archive"]["file"]).read_bytes()) == log.read_bytes()


def test_real_inspect_json_sink_then_archive_with_local_mock_model(setup):
    executable = Path("/rmeng_data/robtang/tools/apt-root/usr/bin/apptainer")
    image = Path("/rmeng_data/robtang/ptb-containers/vllm_debug.sif")
    if not executable.is_file() or not image.is_file():
        pytest.skip("optional site image unavailable")
    # The image's MockLLM provider returns a constant Python string. No network,
    # model inference, PTB task/evaluator or benchmark dataset is used.
    root = setup[1].parent
    source = Path("/fixture") / setup[0].relative_to(root)
    result = Path("/fixture") / setup[1].relative_to(root)
    code = r'''
import importlib.util,json,os,sys
from pathlib import Path
source,result=map(Path,sys.argv[1:])
os.environ["INSPECT_LOG_DIR"]=str(source)
runtime=Path("/fixture/inspect-runtime")
for setting,leaf in (("XDG_DATA_HOME","data"),("XDG_CACHE_HOME","cache"),("XDG_CONFIG_HOME","config")):
    directory=runtime/leaf
    directory.mkdir(parents=True,exist_ok=True)
    os.environ[setting]=str(directory)
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import match
task=Task(dataset=[Sample(id=f"toy-{i}",input=f"SYNTHETIC fixture {i}",
                         target="Default output from mockllm/model") for i in (1,2)],
          solver=generate(),scorer=match(),name="synthetic_retention_fixture")
logs=eval(task,model="mockllm/model",log_format="json",display="plain",log_samples=True,
          log_realtime=False,score_display=False,max_connections=2,max_tokens=40)
assert len(logs)==1 and logs[0].status=="success"
assert Path(logs[0].location).parent.resolve()==source.resolve()
assert len(list(source.glob("*.json")))==1
module_spec=importlib.util.spec_from_file_location("evidence","/opt/ptb-evidence.py")
module=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
receipt=module.preserve_attempt(source,result,job_id="901",attempt=1,phase="post_attempt",exit_code=0)
item=receipt["logs"][0]
assert item["compact"]["status"]=="available"
assert item["compact"]["metadata"]["observed_sample_rows"]==2
assert item["compact"]["metadata"]["model"]=="mockllm/model"
print("SYNTHETIC_RETENTION_RESULT="+json.dumps(receipt))
'''
    before = (setup[1] / "metrics.json").read_bytes()
    proc = subprocess.run([
        str(executable), "exec", "--cleanenv", "--no-home", "--containall",
        "--bind", f"{SCRIPT}:/opt/ptb-evidence.py:ro", "--bind", f"{root}:/fixture",
        "--pwd", "/fixture", str(image), "python", "-c", code, str(source), str(result),
    ], capture_output=True, text=True, timeout=40)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    marker = "SYNTHETIC_RETENTION_RESULT="
    line = next(line for line in proc.stdout.splitlines() if line.startswith(marker))
    item = json.loads(line[len(marker):])["logs"][0]
    raw = (setup[0] / item["source_name"]).read_bytes()
    assert gzip.decompress((receipt_dir(setup) / item["archive"]["file"]).read_bytes()) == raw
    assert (setup[1] / "metrics.json").read_bytes() == before
