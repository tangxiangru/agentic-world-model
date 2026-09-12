"""Verify the source fetcher's trust boundary without network requests."""

import hashlib
import json

import pytest

from tools.outcome_prediction.hf_benchmark_fetch import (
    AssetError,
    build_plan,
    fetch_assets,
    fetch_one,
    raw_downloader,
    selection_reason,
    verify_bytes,
)

REVISION = "a" * 40


def entry(path="rescore10/results/example.json", raw=b'{"ok":true}', **kwargs):
    return {"path": path, "size": len(raw), "blob_id": hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw).hexdigest(), "lfs_sha256": None,
        "selection_reason": "ten_run_result", **kwargs}


def test_hash_failure_never_writes_unverified_asset(tmp_path):
    expected = entry()
    wrong_bytes = b'{"ok":null}'
    assert len(wrong_bytes) == expected["size"]
    with pytest.raises(AssetError, match="hash_mismatch"):
        fetch_one(expected, tmp_path / "out", [], lambda _: wrong_bytes)
    assert not (tmp_path / "out" / expected["path"]).exists()


def test_corrupt_cache_falls_back_to_download_and_preserves_mirror(tmp_path):
    expected = entry()
    root, mirror = tmp_path / "out", tmp_path / "mirror"
    for directory in (root, mirror):
        (directory / expected["path"]).parent.mkdir(parents=True)
        (directory / expected["path"]).write_bytes(b'{"ok":null}')
    calls = []

    def download(asset):
        calls.append(asset["path"])
        return b'{"ok":true}'

    record = fetch_one(expected, root, [mirror], download)
    assert calls == [expected["path"]]
    assert record["method"] == "verified_hf_download"
    assert len(record["rejected_local_copies"]) == 2
    assert (root / expected["path"]).read_bytes() == b'{"ok":true}'
    assert (mirror / expected["path"]).read_bytes() == b'{"ok":null}'


def test_lfs_verifies_content_not_pointer_blob():
    raw = b"large file contents"
    expected = entry(raw=raw, lfs_sha256=hashlib.sha256(raw).hexdigest(), blob_id="b" * 40)
    assert verify_bytes(raw, expected)["verified_against"] == "lfs_sha256"
    with pytest.raises(AssetError, match="hash_mismatch"):
        verify_bytes(raw.upper(), expected)


def test_unavailable_selected_files_are_explicit_in_receipt(tmp_path):
    first = entry()
    second = entry("cells/example/solve_out_sanitized.txt")
    inventory = {"sha": REVISION, "files": [first, second, entry("weights/model.safetensors")]}
    plan = build_plan(inventory, tmp_path / "out", tmp_path / "receipt")
    receipt = fetch_assets(inventory, plan, downloader=lambda _: b'{"ok":true}', workers=1)
    assert receipt["repository_file_count"] == 3
    assert receipt["selected_file_count"] == 2
    assert receipt["verified_file_count"] == 1
    assert receipt["unavailable"] == [{"path": second["path"],
                                     "reason": "no_verified_local_copy_local_only",
                                     "selection_reason": "recipe_reconstruction_local_only"}]
    assert set(receipt["source_mapping"]) == {first["path"]}
    saved = json.loads((tmp_path / "receipt/fetch_receipt.json").read_text())
    assert {r["status"] for r in saved["records"]} == {"verified", "missing"}


@pytest.mark.parametrize("path", ["../escape", "/absolute", "cells/../escape", "cells\\escape"])
def test_rejects_unsafe_inventory_paths(path):
    with pytest.raises(AssetError):
        selection_reason(path)


def test_selection_includes_nested_results_recipes_and_excludes_logs():
    for path in ["eval_matrix_1k/phase6/results/x.json", "rescore10/results/x.json",
                 "cells/x/wm/cards/exp-1/snapshot/train.py",
                 "dojo_ab_gsm8k/rpm/seed01/artifacts/step002/solution.py"]:
        assert selection_reason(path)
    for path in ["eval_matrix_1k/logs/x.gz", "rescore10/trajectories/x.json.gz",
                 "dojo_ab_gsm8k/rpm/seed01/rollout.jsonl", "weights/model.safetensors"]:
        assert selection_reason(path) is None


def test_hf_rate_limit_waits_until_advertised_reset(monkeypatch):
    clock = [0.0]
    calls = []

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"RateLimit": '"resolvers";r=0;t=3'}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_content(self, _chunk_size):
            yield b'{"ok":true}'

    class Session:
        def get(self, *_args, **_kwargs):
            calls.append(clock[0])
            return Response(429 if len(calls) == 1 else 200)

    monkeypatch.setattr("requests.Session", Session)
    monkeypatch.setattr("tools.outcome_prediction.hf_benchmark_fetch.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("tools.outcome_prediction.hf_benchmark_fetch.time.sleep",
                        lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert raw_downloader(REVISION)(entry()) == b'{"ok":true}'
    assert calls == [0.0, 5.0]
