import hashlib
import json

import pytest

from tools.outcome_prediction.wm_history import package_history, validate_history_for_candidates
from tools.outcome_prediction.wm_model import digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def dataset(tmp_path):
    root, raw = tmp_path / "dataset", tmp_path / "raw"
    raw.mkdir()
    trace = raw / "cells/train-run/solve_out_sanitized.txt"
    trace.parent.mkdir(parents=True)
    trace.write_text("Historical observations and lr=1e-5 top_p=0.95\n")
    split = {"train_cell_ids": ["train-run"], "test_cell_ids": ["test-run"]}
    source = {"benchmark": "gsm8k", "source_revision": "pinned", "raw_root": str(raw)}
    write_json(root / "split.json", split)
    write_json(root / "protocol.json", {"split_sha256": digest(split), "sources": [source]})
    write_json(
        root / "train/raw_trajectory_index.json",
        [
            {
                "cell_id": "train-run",
                "benchmark": "gsm8k",
                "source_revision": "pinned",
                "path": str(trace),
                "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                "bytes": trace.stat().st_size,
            }
        ],
    )
    rows = [
        {
            "example_id": "train-run/exp-01",
            "cell_id": "train-run",
            "card_id": "exp-01",
            "benchmark": "gsm8k",
            "first_stage": "plan",
            "model_input": {
                "plan": {"setup": {"lr": 1e-5}},
                "code": [
                    {
                        "role": "training",
                        "script_path": "train.py",
                        "status": "reconstructed",
                        "content": "lr=1e-5\n",
                    }
                ],
                "known_previous_checkpoints": [{"DO_NOT_COPY_STRUCTURED_CONTEXT": True}],
            },
            "label": {"accuracy": 0.0, "evaluation_n": 1319},
            "audit": {"eligible": True},
        },
        {
            # Non-train row must not be dereferenced beyond its run identity.
            "cell_id": "test-run",
            "model_input": None,
            "label": "HELD_OUT_SECRET",
        },
    ]
    (root / "private").mkdir()
    (root / "private/inventory.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return root, raw, tmp_path / "history"


def test_train_only_raw_history_not_wm_feature_input(dataset):
    root, raw, out = dataset
    manifest = package_history(root, out)
    assert manifest["history_run_ids"] == ["train-run"]
    assert manifest["intended_candidate_run_ids"] == ["test-run"]
    validate_history_for_candidates(manifest, candidate_run_ids=["test-run"])
    for name, sha in manifest["files"].items():
        content = (out / "evidence" / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == sha
        assert b"HELD_OUT_SECRET" not in content
        assert b"DO_NOT_COPY_STRUCTURED_CONTEXT" not in content
    assert (out / "evidence/runs/train-run/raw_trajectory.txt").read_bytes() == (
        raw / "cells/train-run/solve_out_sanitized.txt"
    ).read_bytes()
    cards = json.loads((out / "evidence/historical_cards.json").read_text())
    assert cards[0]["official_accuracy"] == 0.0
    assert "not current WM training membership" in cards[0]["scope"]
    assert "manifest.json" not in manifest["files"]
    assert "provenance.json" not in manifest["files"]


@pytest.mark.parametrize(
    "candidate,ancestors",
    [
        (["train-run"], []),
        (["test-run"], ["train-run"]),
    ],
)
def test_candidate_ancestor_overlap_never_allowed(dataset, candidate, ancestors):
    root, _, out = dataset
    manifest = package_history(root, out)
    with pytest.raises(ValueError, match="expose outcomes"):
        validate_history_for_candidates(
            manifest, candidate_run_ids=candidate, ancestor_run_ids=ancestors
        )


def test_unknown_candidate_run_rejected(dataset):
    root, _, out = dataset
    manifest = package_history(root, out)
    with pytest.raises(ValueError, match="outside"):
        validate_history_for_candidates(manifest, candidate_run_ids=["new-run"])


def test_raw_tamper_and_source_symlink_rejected(dataset, tmp_path):
    root, raw, out = dataset
    trace = raw / "cells/train-run/solve_out_sanitized.txt"
    trace.write_text("tampered")
    with pytest.raises(ValueError, match="SHA256"):
        package_history(root, out)
    assert not (out / "manifest.json").exists()
    original = tmp_path / "original.txt"
    trace.rename(original)
    trace.symlink_to(original)
    with pytest.raises(ValueError, match="symlinks"):
        package_history(root, tmp_path / "second")


def test_no_overwrite(dataset):
    root, _, out = dataset
    manifest = package_history(root, out)
    with pytest.raises(FileExistsError):
        package_history(root, out)
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_credentials_removed_without_numeric_redaction(dataset):
    root, raw, out = dataset
    trace = raw / "cells/train-run/solve_out_sanitized.txt"
    synthetic = "hf_" + "X" * 35
    trace.write_text(f"{synthetic} lr=1e-5 top_p=0.95 n=8604\n")
    index_path = root / "train/raw_trajectory_index.json"
    index = json.loads(index_path.read_text())
    index[0].update(
        sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), bytes=trace.stat().st_size
    )
    write_json(index_path, index)
    package_history(root, out)
    text = (out / "evidence/runs/train-run/raw_trajectory.txt").read_text()
    assert synthetic not in text
    assert "lr=1e-5 top_p=0.95 n=8604" in text
    audit = json.loads((out / "provenance.json").read_text())
    assert synthetic not in json.dumps(audit)
