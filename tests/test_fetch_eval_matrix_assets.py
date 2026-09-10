"""Offline contract tests: pinned small assets only, no inference or credentials."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.outcome_prediction import fetch_eval_matrix_assets as fetch


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    (root / "phases").mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps({"benchmark_files": {"gsm8k": [
        {"path": "data/old-mirror/rescore10/eval/tasks/gsm8k/test_data.json",
         "sha256": digest(b"[]")}]}}))
    rows = [{"checkpoint_id": "r0-01-exp-01"}, {"checkpoint_id": "r0-01-exp-01"},
            {"checkpoint_id": "aime-r0-01-exp-02"}]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    (root / "phases/1_operational_pilot.jsonl").write_text(content)
    (root / "experiment_matrix.jsonl").write_text(content)
    return root


def mock_hub(tmp_path, *, listed=None):
    calls = []
    evaluator = "rescore10/eval/tasks/gsm8k/test_data.json"
    listed = listed if listed is not None else [SimpleNamespace(path=evaluator, size=2)]

    class API:
        def list_repo_tree(self, repo, **kwargs):
            calls.append(("list", repo, kwargs))
            return listed

    def download(**kwargs):
        calls.append(("download", kwargs))
        raw = b"[]" if kwargs["filename"] == evaluator else b"{}"
        target = tmp_path / (digest(json.dumps(kwargs, sort_keys=True).encode()) + ".cache")
        target.write_bytes(raw)
        return str(target)

    return API(), download, calls


def test_plan_deduplicates_checkpoints_and_pins_allowlist(bundle):
    plan = fetch.build_plan(bundle)
    assert plan["phase"] == "pilot"
    assert len(plan["checkpoint_ids"]) == 2
    assert len(plan["metadata_files"]) == 4
    assert plan["eval_revision"] == fetch.EVAL_REV
    assert plan["metadata_revision"] == fetch.META_REV
    assert all(name.endswith(("/config.json", "/generation_config.json"))
               for name in plan["metadata_files"])
    assert fetch.build_plan(bundle, "all")["checkpoint_ids"] == plan["checkpoint_ids"]


def test_dry_run_is_offline_and_creates_nothing(bundle, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "fetch_assets", lambda *a, **k: pytest.fail("network attempted"))
    out = tmp_path / "not-created"
    assert fetch.main(["--bundle", str(bundle), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert json.loads(capsys.readouterr().out)["eval_revision"] == fetch.EVAL_REV


def test_fetch_receipts_resume_and_only_download_allowed_files(bundle, tmp_path):
    api, downloader, calls = mock_hub(tmp_path)
    plan, out = fetch.build_plan(bundle), tmp_path / "assets"
    records = fetch.fetch_assets(plan, out, api=api, downloader=downloader)
    assert len(records) == 6
    assert calls[0][2] == {"repo_type": "dataset", "revision": fetch.EVAL_REV,
                           "path_in_repo": "rescore10/eval", "recursive": True}
    downloads = [call[1] for call in calls if call[0] == "download"]
    assert len(downloads) == 6
    for record in records:
        path = out / record["revision"] / record["filename"]
        assert path.is_file()
        assert digest(path.read_bytes()) == record["sha256"]
        assert json.loads(Path(str(path) + ".receipt.json").read_text()) == record
    fetch.fetch_assets(plan, out, api=api, downloader=downloader)
    assert len([call for call in calls if call[0] == "download"]) == 6


def test_protocol_hash_mismatch_is_not_materialized(bundle, tmp_path):
    api, downloader, _ = mock_hub(tmp_path)
    plan = fetch.build_plan(bundle)
    filename = next(iter(plan["expected_sha256"]))
    plan["expected_sha256"][filename] = "0" * 64
    with pytest.raises(fetch.AssetError, match="SHA256 mismatch"):
        fetch.fetch_assets(plan, tmp_path / "assets", api=api, downloader=downloader)
    assert not (tmp_path / "assets" / fetch.EVAL_REV / filename).exists()


def test_existing_unverified_file_is_never_overwritten(tmp_path):
    target = tmp_path / fetch.META_REV / "checkpoints_meta/id/config.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original")
    with pytest.raises(fetch.AssetError, match="unverified"):
        fetch.fetch_one(tmp_path, fetch.META_REV, "checkpoints_meta/id/config.json", None,
                        lambda **k: pytest.fail("must preserve existing file"))
    assert target.read_bytes() == b"original"


def test_modified_receipt_is_not_trusted(bundle, tmp_path):
    api, downloader, _ = mock_hub(tmp_path)
    plan, out = fetch.build_plan(bundle), tmp_path / "assets"
    record = fetch.fetch_assets(plan, out, api=api, downloader=downloader)[0]
    receipt = out / record["revision"] / (record["filename"] + ".receipt.json")
    record["revision"] = "bad"
    receipt.write_text(json.dumps(record))
    with pytest.raises(fetch.AssetError, match="receipt/hash mismatch"):
        fetch.fetch_assets(plan, out, api=api, downloader=downloader)


@pytest.mark.parametrize("value", ["../escape", "/absolute", "x/y", "x\\y", "", "."])
def test_unsafe_checkpoint_ids_rejected(bundle, value):
    (bundle / "phases/1_operational_pilot.jsonl").write_text(json.dumps({"checkpoint_id": value}))
    with pytest.raises(fetch.AssetError, match="checkpoint"):
        fetch.build_plan(bundle)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a\\b", "a//b"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(fetch.AssetError):
        fetch.relative_path(path)


def test_symlink_cache_escape_rejected(tmp_path):
    root, elsewhere = tmp_path / "cache", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (root / fetch.META_REV).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(fetch.AssetError, match="symlink"):
        fetch.destination(root, fetch.META_REV + "/checkpoints_meta/id/config.json")


def test_unexpected_evaluator_file_rejected_before_download(bundle, tmp_path):
    api, downloader, calls = mock_hub(tmp_path, listed=[
        SimpleNamespace(path="rescore10/eval/model.safetensors", size=1)])
    with pytest.raises(fetch.AssetError, match="Unexpected"):
        fetch.fetch_assets(fetch.build_plan(bundle), tmp_path / "assets", api=api, downloader=downloader)
    assert not any(call[0] == "download" for call in calls)


def test_cache_inside_bundle_rejected(bundle, tmp_path):
    api, downloader, calls = mock_hub(tmp_path)
    with pytest.raises(fetch.AssetError, match="outside"):
        fetch.fetch_assets(fetch.build_plan(bundle), bundle / "cache", api=api, downloader=downloader)
    assert calls == []


def test_auth_error_does_not_echo_secret(bundle, tmp_path, monkeypatch, capsys):
    class Denied(Exception):
        response = SimpleNamespace(status_code=401)

    def fail(*args, **kwargs):
        raise Denied("private-token-never-print")

    monkeypatch.setattr(fetch, "fetch_assets", fail)
    assert fetch.main(["--bundle", str(bundle), "--out", str(tmp_path / "cache")]) == 2
    captured = capsys.readouterr()
    assert "Authentication/read access failed" in captured.err
    assert "private-token-never-print" not in captured.err + captured.out
