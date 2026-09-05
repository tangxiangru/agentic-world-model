#!/usr/bin/env python3
"""Prospective, opt-in official Inspect evidence archiver (stdlib, no model calls).

Not wired into PTB or the operator yet. The caller must stop the evaluator
before invoking this helper. Scientific completion and scores are never set.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MODE = "inspect-json-v1"
SCHEMA = "ptb-official-evidence-v1"
COMPACT_INPUT_LIMIT = 128 * 1024 * 1024
COMPACT_OUTPUT_LIMIT = 2 * 1024 * 1024
CHUNK = 1024 * 1024


class RetentionError(ValueError):
    pass


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def value_hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RetentionError(f"not a regular file: {path}")
    return os.fdopen(fd, "rb")


def file_hash(path):
    with open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        size = 0
        for block in iter(lambda: stream.read(CHUNK), b""):
            size += len(block)
            digest.update(block)
        if fingerprint(before) != fingerprint(os.fstat(stream.fileno())) or size != before.st_size:
            raise RetentionError(f"file changed while reading: {path}")
    return {"sha256": digest.hexdigest(), "bytes": size}


def publish(temp, target):
    """Atomic no-clobber publication; resumable artifacts must match byte-for-byte."""
    try:
        os.link(temp, target)
    except FileExistsError:
        if file_hash(temp) != file_hash(target):
            raise RetentionError(f"existing artifact differs: {target}")


def temporary(directory):
    fd, name = tempfile.mkstemp(prefix=".retention-", suffix=".tmp", dir=directory)
    os.close(fd)
    return Path(name)


def write_atomic(target, raw):
    temp = temporary(target.parent)
    try:
        with temp.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        publish(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def archive_raw(source, target):
    temp = temporary(target.parent)
    try:
        with open_regular(source) as stream, temp.open("wb") as output:
            before = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            size = 0
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=1) as zipped:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    size += len(block)
                    digest.update(block)
                    zipped.write(block)
            if fingerprint(before) != fingerprint(os.fstat(stream.fileno())) or size != before.st_size:
                raise RetentionError(f"log changed while archiving: {source}")
            output.flush()
            os.fsync(output.fileno())
        publish(temp, target)
        return {"source_sha256": digest.hexdigest(), "source_bytes": size,
                "archive": {"file": target.name, **file_hash(target)}}
    finally:
        temp.unlink(missing_ok=True)


def selected(mapping, names):
    return {name: mapping.get(name) for name in names} if isinstance(mapping, dict) else None


def compact_rows(doc):
    samples = doc.get("samples")
    if not isinstance(samples, list):
        raise RetentionError("no sample list; raw log retained")
    for sample in samples:
        if not isinstance(sample, dict):
            raise RetentionError("non-object sample; raw log retained")
        calls = [event for event in (sample.get("events") or [])
                 if isinstance(event, dict) and event.get("event") == "model"]
        requests = []
        for event in calls:
            if not isinstance(event.get("input"), list) or any(
                not isinstance(message, dict) or "role" not in message or "content" not in message
                for message in event["input"]
            ):
                raise RetentionError("malformed model request; raw log retained")
            requests.append([selected(message, ("role", "content")) for message in event["input"]])
        output = sample.get("output") or {}
        completion = output.get("completion")
        score_records = sample.get("scores") or {}
        if not isinstance(score_records, dict) or any(not isinstance(record, dict) for record in score_records.values()):
            raise RetentionError("malformed sample scores; raw log retained")
        scores = {name: record.get("value") for name, record in score_records.items()}
        yield {
            "id": sample.get("id"), "epoch": sample.get("epoch"), "scores": scores,
            "input_target_sha256": value_hash(selected(sample, ("input", "target")))
            if sample.get("input") is not None and sample.get("target") is not None else None,
            "request_role_content_sha256": value_hash(requests) if calls else None,
            "model_call_count": len(calls),
            "completion_sha256": hashlib.sha256(completion.encode()).hexdigest()
            if isinstance(completion, str) else None,
            "completion_chars": len(completion) if isinstance(completion, str) else None,
            "finish_reasons": [choice.get("stop_reason", choice.get("finish_reason"))
                               for choice in (output.get("choices") or [])],
            "usage": selected(output.get("usage"), ("input_tokens", "output_tokens", "total_tokens")),
        }


def compact_worker(archive, output, max_bytes, memory_mb, cpu_seconds):
    """Runs in a disposable subprocess; only raw archival is required for preservation."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    with gzip.open(archive, "rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise RetentionError("compaction input byte limit exceeded")
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise RetentionError("log is not an object")
    count = 0
    with open(output, "wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=1) as zipped:
            for row in compact_rows(doc):
                zipped.write(encoded(row) + b"\n")
                count += 1
                if stream.tell() > COMPACT_OUTPUT_LIMIT:
                    raise RetentionError("compact artifact exceeds byte limit")
    if os.path.getsize(output) > COMPACT_OUTPUT_LIMIT:
        raise RetentionError("compact artifact exceeds byte limit")
    evaluation = doc.get("eval") or {}
    results = doc.get("results") or {}
    summary = {
        "source_status": doc.get("status"), "observed_sample_rows": count,
        "model": evaluation.get("model"),
        "dataset": selected(evaluation.get("dataset"), ("name", "location", "samples", "shuffled", "seed")),
        "model_args": selected(evaluation.get("model_args"),
                               ("gpu_memory_utilization", "chat_template", "dtype", "seed")),
        "generation_config": selected(evaluation.get("model_generate_config"),
                                      ("max_connections", "max_tokens", "temperature", "top_p", "top_k", "seed")),
        "packages": selected(evaluation.get("packages"), ("inspect_ai", "vllm", "torch", "transformers")),
        "counts": selected(results, ("total_samples", "completed_samples")),
        "scorers": [selected(score, ("name", "scorer", "scored_samples", "unscored_samples"))
                    for score in (results.get("scores") or [])],
    }
    if len(encoded(summary)) > 64 * 1024:
        raise RetentionError("metadata exceeds byte limit")
    return summary


def make_compact(archive, target, *, max_bytes, memory_mb, cpu_seconds):
    temp = temporary(target.parent)
    try:
        command = [sys.executable, str(Path(__file__).resolve()), "_compact", str(archive),
                   str(temp), str(max_bytes), str(memory_mb), str(cpu_seconds)]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=cpu_seconds + 10)
        except subprocess.TimeoutExpired:
            return {"status": "unavailable", "reason": "compaction wall-time limit exceeded"}
        if process.returncode:
            return {"status": "unavailable", "reason": (process.stderr.strip()[:300]
                    or f"compactor exit {process.returncode}; possible resource limit")}
        metadata = json.loads(process.stdout)
        publish(temp, target)
        return {"status": "available", "file": target.name, **file_hash(target), "metadata": metadata}
    finally:
        temp.unlink(missing_ok=True)


def identity_for(result_dir, job_id):
    provenance = result_dir / "runtime_provenance.json"
    with open_regular(provenance) as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(1024 * 1024 + 1)
        if fingerprint(before) != fingerprint(os.fstat(stream.fileno())):
            raise RetentionError("provenance changed while reading")
    if len(raw) > 1024 * 1024:
        raise RetentionError("provenance exceeds byte limit")
    doc = json.loads(raw)
    if not isinstance(doc, dict) or any(not isinstance(doc.get(key), dict) for key in
                                        ("experiment", "source", "slurm", "evaluation_container")):
        raise RetentionError("malformed runtime provenance")
    experiment = doc.get("experiment") or {}
    source = doc.get("source") or {}
    if experiment.get("official_log_retention") != MODE or experiment.get("task") != "gsm8k":
        raise RetentionError("result has no supported frozen GSM8K retention opt-in")
    if str((doc.get("slurm") or {}).get("job_id")) != str(job_id) or not re.fullmatch(r"\d+", str(job_id)):
        raise RetentionError("requested job differs from runtime provenance")
    for field in ("batch_id", "cell_id", "run_purpose"):
        if not isinstance(experiment.get(field), str) or not experiment[field]:
            raise RetentionError(f"missing experiment identity: {field}")
    image = (doc.get("evaluation_container") or {}).get("sha256", "")
    if not isinstance(image, str) or not re.fullmatch(r"[0-9a-f]{64}", image):
        raise RetentionError("missing frozen evaluation-image hash")
    for key in ("top_commit", "ptb_commit"):
        if not isinstance(source.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", source[key]):
            raise RetentionError(f"missing frozen {key}")
    return {**selected(experiment, ("batch_id", "cell_id", "run_purpose", "task")),
            "job_id": str(job_id), "mode": MODE, "top_commit": source["top_commit"],
            "ptb_commit": source["ptb_commit"], "evaluation_image_sha256": image,
            "provenance_sha256": hashlib.sha256(raw).hexdigest()}


def artifact_path(directory, name):
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise RetentionError("unsafe artifact name in receipt")
    path = directory / name
    if path.is_symlink():
        raise RetentionError(f"symlink artifact refused: {path}")
    return path


def source_logs(directory):
    if directory.is_symlink():
        raise RetentionError("symlink source directory refused")
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise RetentionError("source is not a directory")
    logs = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise RetentionError(f"symlink source refused: {path}")
        if path.suffix == ".json":
            if not path.is_file():
                raise RetentionError(f"non-file JSON source: {path}")
            logs.append(path)
        elif not path.name.startswith("."):
            raise RetentionError(f"unexpected source entry: {path}")
    return logs


def verify_existing(directory, receipt, identity, source, attempt, phase, exit_code):
    if not isinstance(receipt, dict) or not isinstance(receipt.get("logs"), list):
        raise RetentionError("malformed existing receipt")
    if (receipt.get("schema_version") != SCHEMA or receipt.get("identity") != identity
            or receipt.get("attempt") != attempt or receipt.get("source_dir") != str(source)):
        raise RetentionError("existing receipt has a different identity")
    if phase != "cleanup" and (receipt.get("phase"), receipt.get("evaluator_exit_code")) != (phase, exit_code):
        raise RetentionError("attempt already finalized with a different exit observation")
    for record in receipt["logs"]:
        if (not isinstance(record, dict) or not isinstance(record.get("archive"), dict)
                or not isinstance(record.get("compact"), dict) or "file" not in record["archive"]):
            raise RetentionError("malformed archived-log record")
        for artifact in (record["archive"], record["compact"]):
            if "file" in artifact:
                actual = file_hash(artifact_path(directory, artifact["file"]))
                if actual != selected(artifact, ("sha256", "bytes")):
                    raise RetentionError("archived artifact hash mismatch")
    if source.exists():
        logs = source_logs(source)
        expected = {row["source_name"]: row for row in receipt["logs"]}
        if {path.name for path in logs} != set(expected):
            raise RetentionError("source log set changed after finalization")
        for path in logs:
            wanted = expected[path.name]
            if file_hash(path) != {"sha256": wanted["source_sha256"], "bytes": wanted["source_bytes"]}:
                raise RetentionError("source bytes changed after finalization")
    return receipt


def preserve_attempt(source_dir, result_dir, *, job_id, attempt, phase, exit_code=None,
                     compact_max_bytes=COMPACT_INPUT_LIMIT, compact_memory_mb=512, compact_cpu_seconds=10):
    if type(attempt) is not int or attempt < 1:
        raise RetentionError("attempt must be a positive integer")
    if phase not in ("post_attempt", "cleanup") or (phase == "cleanup" and exit_code is not None):
        raise RetentionError("cleanup cannot invent an evaluator exit code")
    if phase == "post_attempt" and type(exit_code) is not int:
        raise RetentionError("post-attempt archival needs the observed evaluator exit code")
    if any(type(value) is not int or value <= 0 for value in
           (compact_max_bytes, compact_memory_mb, compact_cpu_seconds)):
        raise RetentionError("compaction resource limits must be positive")
    source = Path(source_dir).absolute()
    expected_name = f"attempt-{attempt:04d}"
    if source.name != expected_name or source.parent.name != "official-eval" or ".." in source.parts:
        raise RetentionError("source must be the exact numbered official-eval attempt directory")
    if source.resolve() != source:
        raise RetentionError("symlink in source path refused")
    result = Path(result_dir).resolve(strict=True)
    identity = identity_for(result, job_id)  # fail before creating anything for legacy results
    for directory in (result / "official_eval", result / "official_eval" / expected_name):
        if directory.is_symlink():
            raise RetentionError("symlink destination directory refused")
        directory.mkdir(exist_ok=True)
    output = result / "official_eval" / expected_name
    fd = os.open(output / ".archive.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RetentionError("another archiver owns this attempt") from exc
        receipt_path = output / "receipt.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            with open_regular(receipt_path) as stream:
                receipt = json.load(stream)
            return verify_existing(output, receipt, identity, source, attempt, phase, exit_code)
        logs = source_logs(source)
        records = []
        for log in logs:
            raw = archive_raw(log, output / (log.name + ".gz"))
            compact = ({"status": "unavailable", "reason": "compaction input byte limit exceeded"}
                       if raw["source_bytes"] > compact_max_bytes else
                       make_compact(output / raw["archive"]["file"], output / (log.stem + ".samples.jsonl.gz"),
                                    max_bytes=compact_max_bytes, memory_mb=compact_memory_mb,
                                    cpu_seconds=compact_cpu_seconds))
            records.append({"source_name": log.name, **raw, "compact": compact})
        metrics = result / "metrics.json"
        metrics_snapshot = file_hash(metrics) if phase == "post_attempt" and (metrics.exists() or metrics.is_symlink()) else None
        receipt = {"schema_version": SCHEMA, "identity": identity, "attempt": attempt,
                   "phase": phase, "evaluator_exit_code": exit_code, "source_dir": str(source),
                   "archived_at": datetime.now(timezone.utc).isoformat(),
                   "archive_status": "preserved" if records else "no_log",
                   "source_directory_existed": source.exists(), "logs": records,
                   "metrics_observed_after_attempt": metrics_snapshot,
                   "compaction_limits": {"input_bytes": compact_max_bytes, "output_bytes": COMPACT_OUTPUT_LIMIT,
                                         "memory_mb": compact_memory_mb, "cpu_seconds": compact_cpu_seconds}}
        # The receipt is the commit point; unindexed artifacts remain recoverable, never authoritative.
        write_atomic(receipt_path, encoded(receipt) + b"\n")
        return receipt
    finally:
        os.close(fd)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_compact":
        try:
            result = compact_worker(sys.argv[2], sys.argv[3], *map(int, sys.argv[4:7]))
            print(encoded(result).decode())
            return 0
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=("post_attempt", "cleanup"))
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    try:
        receipt = preserve_attempt(args.source_dir, args.result_dir, job_id=args.job_id,
                                   attempt=args.attempt, phase=args.phase, exit_code=args.exit_code)
        print(encoded(receipt).decode())
        return 0
    except (RetentionError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"retention failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
