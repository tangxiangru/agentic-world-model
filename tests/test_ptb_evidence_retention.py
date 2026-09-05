"""Official source bytes survive harvest even when larger than its text cap."""

import gzip
import hashlib
import json
import shutil

import pytest

from awm import ptb_evidence_retention as retention
from awm import ptb_ops


def fixture_result(tmp_path):
    root = tmp_path / "result"
    attempt = root / "official_eval" / ("a" * 32)
    attempt.mkdir(parents=True)
    (attempt / "request.json").write_text('{"synthetic":true}')
    (attempt / "python-runtime.json").write_text('{"synthetic_runtime":true}')
    failed = root / "official_eval" / ("b" * 32)
    failed.mkdir()
    (failed / "failed.log").write_text("synthetic failed attempt remains evidence\n")
    snapshot = root / ".official-inspect-frozen" / "inspect.json"
    snapshot.parent.mkdir()
    raw = json.dumps({"synthetic": "x" * (ptb_ops.PER_FILE_CAP + 100)}).encode()
    snapshot.write_bytes(raw)
    (root / "runtime_provenance.json").write_text('{"synthetic":true}')
    (root / "metrics.json").write_text(json.dumps({
        "accuracy": 0.5, "stderr": 0.1, "official_evidence": {
            "raw_log": snapshot.relative_to(root).as_posix(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw),
        },
    }))
    return root, snapshot


def harvest(root, bundle, monkeypatch, **kwargs):
    monkeypatch.setattr(ptb_ops, "audit_result", lambda *_a, **_k: [])
    return ptb_ops.harvest_job(root, bundle, batch="synthetic", cell="c1", job_id="123",
                               expected_task="humaneval", **kwargs)


def verify(bundle):
    # status is the caller's separately retained operator record, not index data.
    status = json.loads((bundle / "status.json").read_text())
    return retention.verify_official_evidence(
        bundle, expected_index_fingerprint=status["official_evidence_retention"]["index_fingerprint"]
    )


def test_harvest_keeps_large_raw_and_failed_attempt_without_original_volume(tmp_path, monkeypatch):
    root, snapshot = fixture_result(tmp_path)
    raw = snapshot.read_bytes()
    bundle = tmp_path / "bundle"
    status = harvest(root, bundle, monkeypatch)
    assert status["official_evidence_retention"]["state"] == "preserved"
    assert status["official_evidence_retention"]["files"] == 4
    shutil.rmtree(root)
    index = verify(bundle)
    saved = next(r for r in index["files"] if r["source"].endswith("inspect.json"))
    assert gzip.decompress((bundle / "official-evidence" / saved["archive"]).read_bytes()) == raw
    assert any(r["source"].endswith("failed.log") for r in index["files"])


@pytest.mark.parametrize("change", ["delete", "tamper", "metadata", "symlink"])
def test_archive_changes_are_rejected(tmp_path, monkeypatch, change):
    root, _ = fixture_result(tmp_path)
    bundle = tmp_path / "bundle"
    harvest(root, bundle, monkeypatch)
    index = verify(bundle)
    path = bundle / "official-evidence" / index["files"][0]["archive"]
    if change == "metadata":
        (bundle / "metrics.json").write_text('{"accuracy":1.0}')
    elif change == "delete":
        path.unlink()
    elif change == "symlink":
        other = tmp_path / "same-bytes"
        path.rename(other)
        path.symlink_to(other)
    else:
        path.write_bytes(b"changed")
    with pytest.raises((retention.RetentionError, OSError)):
        verify(bundle)


@pytest.mark.parametrize("change", ["delete_raw", "tamper_raw", "symlink_dir", "weights"])
def test_incomplete_or_unsafe_source_is_not_certified(tmp_path, monkeypatch, change):
    root, snapshot = fixture_result(tmp_path)
    if change == "delete_raw":
        snapshot.unlink()
    elif change == "tamper_raw":
        snapshot.write_text("changed")
    elif change == "symlink_dir":
        target = tmp_path / "external"
        target.mkdir()
        (target / "secret.json").write_text("must not be read")
        (root / "official_eval" / "external").symlink_to(target, target_is_directory=True)
    else:
        (root / "official_eval" / "model.safetensors").write_bytes(b"not a log")
    bundle = tmp_path / "bundle"
    status = harvest(root, bundle, monkeypatch)
    assert status["official_evidence_retention"]["state"] == "partial"
    with pytest.raises(retention.RetentionError):
        verify(bundle)
    assert not list((bundle / "official-evidence").rglob("secret.json.gz"))
    assert not list((bundle / "official-evidence").rglob("*.safetensors.gz"))


def test_retention_does_not_erase_judge_or_placement_facts(tmp_path, monkeypatch):
    root, _ = fixture_result(tmp_path)
    (root / "judgement_general.json").write_text('{"general_anomaly":true}')
    status = harvest(root, tmp_path / "bundle", monkeypatch, expected_nodes={"owned-node"})
    assert status["quarantined"] and not status["eligible"]
    assert status["judge_flags"] == ["general_anomaly"]
    assert status["official_evidence_retention"]["state"] == "preserved"


def test_legacy_result_without_official_evidence_is_unchanged(tmp_path, monkeypatch):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "metrics.json").write_text('{"accuracy":0.5}')
    status = harvest(root, tmp_path / "bundle", monkeypatch)
    assert status["complete"] and status["accuracy"] == 0.5
    assert status["official_evidence_retention"]["state"] == "absent"


@pytest.mark.parametrize("change", ["empty", "source", "binding"])
def test_index_cannot_certify_its_own_edits(tmp_path, monkeypatch, change):
    root, _ = fixture_result(tmp_path)
    bundle = tmp_path / "bundle"
    harvest(root, bundle, monkeypatch)
    path = bundle / "official-evidence/index.json"
    index = json.loads(path.read_text())
    if change == "empty":
        index.update(files=[], bound_files={}, errors=[], state="preserved")
    elif change == "source":
        index["files"][0]["source"] = "official_eval/forged.json"
    else:
        index["bound_files"] = {}
    path.write_text(json.dumps(index))
    with pytest.raises(retention.RetentionError, match="index changed"):
        verify(bundle)


def test_walk_permission_error_is_visible_not_preserved(tmp_path, monkeypatch):
    root, _ = fixture_result(tmp_path)
    walk = retention.os.walk

    def denied(top, **kwargs):
        if top.name == "official_eval":
            kwargs["onerror"](PermissionError("synthetic unreadable failed attempt"))
            return iter(())
        return walk(top, **kwargs)

    monkeypatch.setattr(retention.os, "walk", denied)
    status = harvest(root, tmp_path / "bundle", monkeypatch)
    assert status["official_evidence_retention"]["state"] == "partial"
    assert any("cannot enumerate" in e["error"] for e in status["official_evidence_retention"]["errors"])


def test_added_attempt_during_copy_prevents_full_retention_claim(tmp_path, monkeypatch):
    root, _ = fixture_result(tmp_path)
    original = retention._archive

    def racing(source, relative, output):
        result = original(source, relative, output)
        (root / "official_eval/new-attempt.json").write_text('{"late":true}')
        return result

    monkeypatch.setattr(retention, "_archive", racing)
    status = harvest(root, tmp_path / "bundle", monkeypatch)
    assert status["official_evidence_retention"]["state"] == "partial"
    assert any("inventory changed" in e["error"] for e in status["official_evidence_retention"]["errors"])


def test_existing_archive_is_never_deleted_on_publish_conflict(tmp_path):
    root, snapshot = fixture_result(tmp_path)
    output = tmp_path / "archive"
    output.mkdir()
    relative = snapshot.relative_to(root)
    first = retention._archive(root, relative, output)
    path = output / first["archive"]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        retention._archive(root, relative, output)
    assert path.read_bytes() == original
    assert not list(output.rglob("*.tmp"))


def test_repeated_capture_refuses_without_mutating_evidence(tmp_path, monkeypatch):
    root, _ = fixture_result(tmp_path)
    bundle = tmp_path / "bundle"
    harvest(root, bundle, monkeypatch)
    before = {p: p.read_bytes() for p in (bundle / "official-evidence").rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        retention.preserve_official_evidence(root, bundle)
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_archive_output_cannot_follow_directory_symlinks(tmp_path):
    root, snapshot = fixture_result(tmp_path)
    output = tmp_path / "archive"
    output.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (output / "raw").symlink_to(external, target_is_directory=True)
    with pytest.raises(OSError):
        retention._archive(root, snapshot.relative_to(root), output)
    assert list(external.iterdir()) == []


@pytest.mark.parametrize("failure", ["inventory", "index", "gzip"])
def test_retention_failure_still_writes_scientific_status(tmp_path, monkeypatch, failure):
    root, _ = fixture_result(tmp_path)
    (root / "judgement_general.json").write_text('{"general_anomaly":true}')
    if failure == "inventory":
        def fail_inventory(_root):
            raise PermissionError("inventory unavailable")
        monkeypatch.setattr(retention, "_sources", fail_inventory)
    elif failure == "index":
        original_open = retention.os.open

        def fail_index(path, *args, **kwargs):
            if path == "index.json":
                raise PermissionError("index unavailable")
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr(retention.os, "open", fail_index)
    else:
        def fail_gzip(*args, **kwargs):
            raise OSError("gzip output unavailable")
        monkeypatch.setattr(retention.gzip, "GzipFile", fail_gzip)
    status = harvest(root, tmp_path / "bundle", monkeypatch, expected_nodes={"owned"})
    assert status["complete"] and status["quarantined"]
    assert status["judge_flags"] == ["general_anomaly"]
    assert status["official_evidence_retention"]["state"] in {"failed", "partial"}
    assert (tmp_path / "bundle/status.json").is_file()


def test_modified_source_after_archive_is_partial(tmp_path, monkeypatch):
    root, _ = fixture_result(tmp_path)
    original = retention._archive

    def racing(source, relative, output):
        result = original(source, relative, output)
        if relative.name == "failed.log":
            (source / relative).write_text("changed after initial copy")
        return result

    monkeypatch.setattr(retention, "_archive", racing)
    status = harvest(root, tmp_path / "bundle", monkeypatch)
    assert status["official_evidence_retention"]["state"] == "partial"
    assert any("changed after archiving" in e["error"] for e in status["official_evidence_retention"]["errors"])


@pytest.mark.parametrize("target", ["official-evidence/index.json", "metrics.json"])
def test_metadata_is_parsed_from_one_verified_read(tmp_path, monkeypatch, target):
    root, _ = fixture_result(tmp_path)
    bundle = tmp_path / "bundle"
    harvest(root, bundle, monkeypatch)
    original = retention._open_beneath
    opens = []

    def open_once(base, relative):
        if base / relative == bundle / target:
            opens.append(relative)
            if len(opens) > 1:
                raise AssertionError("metadata reopened after verifying different bytes")
        return original(base, relative)

    monkeypatch.setattr(retention, "_open_beneath", open_once)
    verify(bundle)
    assert len(opens) == 1
