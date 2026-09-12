"""Metadata fetches are read-only remotely and never overwrite unverified files."""

import hashlib
import json
from types import SimpleNamespace

from tools.outcome_prediction.prefix_fetch_metadata import fetch_one

CHECKPOINT = {"id": "r0-01-exp-01", "gs_path": "gs://bucket/checkpoints/r0-01-exp-01/"}


def test_download_receipt_binds_uri_and_exact_bytes(tmp_path, monkeypatch):
    raw = b'{"temperature":0}\n'
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")

    monkeypatch.setattr("tools.outcome_prediction.prefix_fetch_metadata.subprocess.run", run)
    receipt = fetch_one(CHECKPOINT, tmp_path, ["generation_config.json"])
    item = receipt["files"]["generation_config.json"]
    assert item["sha256"] == hashlib.sha256(raw).hexdigest()
    assert calls == [["gcloud", "storage", "cat", CHECKPOINT["gs_path"] + "generation_config.json"]]
    assert (tmp_path / CHECKPOINT["id"] / "generation_config.json").read_bytes() == raw
    fetch_one(CHECKPOINT, tmp_path, ["generation_config.json"], receipt)
    assert len(calls) == 1


def test_unverified_existing_file_is_preserved(tmp_path, monkeypatch):
    target = tmp_path / CHECKPOINT["id"] / "generation_config.json"
    target.parent.mkdir()
    target.write_text('{"temperature":0.7}')
    monkeypatch.setattr(
        "tools.outcome_prediction.prefix_fetch_metadata.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    receipt = fetch_one(CHECKPOINT, tmp_path, ["generation_config.json"])
    assert (
        receipt["files"]["generation_config.json"]["status"]
        == "unverified_local_file_not_overwritten"
    )
    assert json.loads(target.read_text()) == {"temperature": 0.7}


def test_auth_failure_creates_no_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.outcome_prediction.prefix_fetch_metadata.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"Reauthentication failed"
        ),
    )
    receipt = fetch_one(CHECKPOINT, tmp_path, ["generation_config.json"])
    assert receipt["files"]["generation_config.json"]["status"] == "read_failed"
    assert not (tmp_path / CHECKPOINT["id"] / "generation_config.json").exists()


def test_invalid_json_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.outcome_prediction.prefix_fetch_metadata.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=b"[]", stderr=b""),
    )
    receipt = fetch_one(CHECKPOINT, tmp_path, ["generation_config.json"])
    assert receipt["files"]["generation_config.json"]["status"] == "invalid_json"
    assert not (tmp_path / CHECKPOINT["id"] / "generation_config.json").exists()
