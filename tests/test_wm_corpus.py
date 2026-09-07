import hashlib
import json

import pytest

from tools.outcome_prediction.wm_corpus import CorpusError, package_corpus, redact_credentials
from tools.outcome_prediction.wm_evidence import EvidenceStore
from tools.outcome_prediction.wm_model import digest, public_payload


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


@pytest.fixture
def prepared(tmp_path):
    dataset, source = tmp_path / "dataset", tmp_path / "source"
    source.mkdir()
    write_json(source / "manifest.json", [{"cell_id": "r0-01"}, {"cell_id": "r0-02"}])
    source_sha = sha((source / "manifest.json").read_bytes())
    rows, index = [], []
    for cell in ["r0-01", "r0-02"]:
        cell_dir = source / "cells" / cell
        cell_dir.mkdir(parents=True)
        trace = cell_dir / "solve_out_sanitized.txt"
        trace.write_bytes(
            ("TRAIN_RAW lr=1e-5 top_p=0.95\r\n" if cell == "r0-01" else "HELD_OUT_SECRET").encode()
        )
        if cell == "r0-01":
            index.append(
                {
                    "cell_id": cell,
                    "benchmark": "gsm8k",
                    "source_revision": "rev",
                    "path": str(trace),
                    "sha256": sha(trace.read_bytes()),
                    "bytes": trace.stat().st_size,
                }
            )
        for number in [1, 2] if cell == "r0-01" else [1]:
            card = f"exp-{number:02d}"
            folder = cell_dir / "wm/cards" / card
            plan = {
                "problem": {"evidence": [{"observation": "old experiment"}]},
                "setup": {"data": [{"source": "scientific/data", "count": 1000}], "lr": 1e-5},
            }
            first = {"at": "2026-01-01T00:00:00Z", "card": {"card_id": card, **plan}}
            final = {"card_id": card, **plan, "result": {"historical_only": True}}
            write_json(folder / "record-01.json", first)
            write_json(folder / "record-02.json", {"at": "2026-01-02T00:00:00Z", "card": final})
            write_json(folder / "card.json", final)
            write_json(cell_dir / "wm_metrics" / f"{card}.json", {"accuracy": 0.4})
            snapshot = folder / "snapshot"
            snapshot.mkdir()
            (snapshot / "train.py").write_text("historical_archive = True\nlr=1e-5\n")
            (snapshot / "unlisted-private.pkl").write_bytes(b"\xffDO_NOT_COPY")
            write_json(
                snapshot / "MANIFEST.json",
                {
                    "files": [
                        {
                            "path": "train.py",
                            "sha256": sha((snapshot / "train.py").read_bytes()),
                            "bytes": (snapshot / "train.py").stat().st_size,
                        },
                        {"path": "nested/missing.py", "sha256": "0" * 64, "bytes": 10},
                    ]
                },
            )
            rows.append(
                {
                    "example_id": f"{cell}/{card}",
                    "cell_id": cell,
                    "card_id": card,
                    "benchmark": "gsm8k",
                    "first_submitted_at": first["at"],
                    "first_stage": "plan",
                    "model_input": {
                        "task": {"benchmark": "gsm8k"},
                        "plan": plan,
                        "code": [
                            {
                                "role": "training",
                                "status": "reconstructed",
                                "script_path": "/original/train.py",
                                "content": "prospective = True\nlr=1e-5\n",
                            }
                        ],
                    },
                    "label": {"accuracy": 0.4, "evaluation_n": 1319},
                    "audit": {"eligible": number == 1},
                    "provenance": {
                        "source_revision": "rev",
                        "first_record_path": str(folder / "record-01.json"),
                        "first_record_sha256": sha((folder / "record-01.json").read_bytes()),
                        "card_sha256": sha((folder / "card.json").read_bytes()),
                        "official_metric_sha256": sha(
                            (cell_dir / "wm_metrics" / f"{card}.json").read_bytes()
                        ),
                    },
                }
            )
    split = {
        "train_cell_ids": ["r0-01"],
        "test_cell_ids": ["r0-02"],
        "train_example_ids": [r["example_id"] for r in rows if r["cell_id"] == "r0-01"],
    }
    train = rows[0]
    examples = [
        {
            "example_id": train["example_id"],
            "cell_id": train["cell_id"],
            "benchmark": "gsm8k",
            "payload_sha256": digest(public_payload(train)),
            "accuracy": train["label"]["accuracy"],
        }
    ]
    model_manifest = {
        "training_examples": examples,
        "declared_train_cell_ids": ["r0-01"],
        "forbidden_cell_ids": ["r0-02"],
    }
    source_code = b"# frozen source\n"
    (dataset / "provenance/source").mkdir(parents=True)
    (dataset / "provenance/source/wm_prepare.py").write_bytes(source_code)
    protocol = {
        "split_sha256": digest(split),
        "training_payload_sha256": digest(examples),
        "source_code_sha256": {"wm_prepare.py": sha(source_code)},
        "sources": [
            {
                "benchmark": "gsm8k",
                "source_revision": "rev",
                "raw_root": str(source),
                "manifest_name": "manifest.json",
            }
        ],
    }
    for name, value in [
        ("protocol.json", protocol),
        ("split.json", split),
        ("model/training_manifest.json", model_manifest),
        ("train/raw_trajectory_index.json", index),
        (
            "source_audits.json",
            [
                {
                    "task": {"benchmark": "gsm8k"},
                    "source_revision": "rev",
                    "manifest_name": "manifest.json",
                    "manifest_sha256": source_sha,
                }
            ],
        ),
    ]:
        write_json(dataset / name, value)
    (dataset / "private").mkdir()
    (dataset / "private/inventory.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    write_json(dataset / "private/test_labels.jsonl", {"secret": "HELD_OUT_SECRET"})
    write_json(dataset / "test/inputs.jsonl", {"secret": "HELD_OUT_SECRET"})
    (dataset / "model/wm.joblib").write_bytes(b"DO_NOT_LOAD_OR_COPY")
    return dataset, source, tmp_path / "corpus"


def mutate_json(path, callback):
    obj = json.loads(path.read_text())
    callback(obj)
    write_json(path, obj)


def test_package_complete_shared_train_only_corpus(prepared, tmp_path):
    dataset, source, output = prepared
    manifest = package_corpus(dataset, output)
    assert manifest["training_run_ids"] == ["r0-01"]
    assert manifest["training_example_count"] == 1 and manifest["historical_card_count"] == 2
    assert manifest["same_evidence_for_both_arms"]
    assert manifest["protocol_status"] == "legacy_ancestor_scored_evidence_not_approved"
    assert all(
        "r0-02" not in name and "private" not in name and ".pkl" not in name
        for name in manifest["files"]
    )
    assert "manifest.json" not in manifest["files"] and "provenance.json" not in manifest["files"]
    for name, expected in manifest["files"].items():
        raw = (output / "evidence" / name).read_bytes()
        assert sha(raw) == expected
        assert b"HELD_OUT_SECRET" not in raw and b"DO_NOT_LOAD_OR_COPY" not in raw
    raw_name = "runs/r0-01/raw_trajectory.txt"
    assert (output / "evidence" / raw_name).read_bytes() == (
        source / "cells/r0-01/solve_out_sanitized.txt"
    ).read_bytes()
    base = output / "evidence/runs/r0-01/cards/exp-01"
    assert "prospective = True" in (base / "recovered_preproposal_code/000.txt").read_text()
    assert "historical_archive" in (base / "historical_final_snapshot/train.py").read_text()
    audit = json.loads((output / "provenance.json").read_text())
    assert (
        sum(a["kind"] == "snapshot_member_missing" for a in audit["missing_historical_artifacts"])
        == 2
    )
    assert not audit["credential_redactions"]
    assert all(
        "not_launch" in a["role"] for a in audit["artifacts"] if "historical_final" in a["path"]
    )
    examples = [
        json.loads(line)
        for line in (output / "evidence/training_examples.jsonl").read_text().splitlines()
    ]
    assert examples[0]["official_accuracy"] == 0.4
    with EvidenceStore(
        output / "evidence", manifest["files"], audit_log=tmp_path / "audit"
    ) as store:
        assert "TRAIN_RAW" in store.handle_call("read_evidence", {"path": raw_name})["text"]


def test_existing_output_never_overwritten(prepared):
    dataset, _, output = prepared
    package_corpus(dataset, output)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        package_corpus(dataset, output)
    assert (output / "manifest.json").read_bytes() == before


def test_dangling_output_symlink_rejected(prepared):
    dataset, _, output = prepared
    output.symlink_to(output.parent / "absent")
    with pytest.raises(FileExistsError):
        package_corpus(dataset, output)


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_hash",
        "raw_path",
        "test_index",
        "missing_index",
        "source_manifest",
        "archived_source",
        "split",
        "model_fit",
    ],
)
def test_integrity_and_split_fail_closed(prepared, mutation):
    dataset, source, output = prepared
    index = dataset / "train/raw_trajectory_index.json"
    if mutation == "raw_hash":
        (source / "cells/r0-01/solve_out_sanitized.txt").write_text("tampered")
    elif mutation == "raw_path":
        mutate_json(
            index, lambda x: x[0].update(path=str(source / "cells/r0-02/solve_out_sanitized.txt"))
        )
    elif mutation == "test_index":
        mutate_json(index, lambda x: x[0].update(cell_id="r0-02"))
    elif mutation == "missing_index":
        write_json(index, [])
    elif mutation == "source_manifest":
        (source / "manifest.json").write_text("[]")
    elif mutation == "archived_source":
        (dataset / "provenance/source/wm_prepare.py").write_text("changed")
    elif mutation == "split":
        mutate_json(dataset / "split.json", lambda x: x.update(train_cell_ids=["r0-02"]))
    elif mutation == "model_fit":
        mutate_json(
            dataset / "model/training_manifest.json",
            lambda x: x["training_examples"][0].update(example_id="r0-02/exp-01"),
        )
    with pytest.raises(CorpusError):
        package_corpus(dataset, output)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize(
    "kind", ["root", "raw_leaf", "cell_directory", "snapshot_leaf", "snapshot_directory"]
)
def test_source_symlinks_never_followed(prepared, kind):
    dataset, source, output = prepared
    target = {
        "root": source,
        "raw_leaf": source / "cells/r0-01/solve_out_sanitized.txt",
        "cell_directory": source / "cells/r0-01",
        "snapshot_leaf": source / "cells/r0-01/wm/cards/exp-01/snapshot/train.py",
        "snapshot_directory": source / "cells/r0-01/wm/cards/exp-01/snapshot",
    }[kind]
    moved = target.with_name(target.name + "-moved")
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(CorpusError):
        package_corpus(dataset, output)
    assert not (output / "manifest.json").exists()


def test_snapshot_corruption_fails_but_missing_members_are_explicit(prepared):
    dataset, source, output = prepared
    snapshot = source / "cells/r0-01/wm/cards/exp-01/snapshot"
    (snapshot / "train.py").write_text("corrupt")
    with pytest.raises(CorpusError, match="SHA256"):
        package_corpus(dataset, output)
    assert not (output / "manifest.json").exists()


def test_snapshot_traversal_rejected(prepared):
    dataset, source, output = prepared
    path = source / "cells/r0-01/wm/cards/exp-01/snapshot/MANIFEST.json"
    mutate_json(path, lambda x: x["files"][0].update(path="../../../../r0-02/secret"))
    with pytest.raises(ValueError, match="logical"):
        package_corpus(dataset, output)


def test_unlisted_new_cached_run_is_not_enumerated(prepared):
    dataset, source, output = prepared
    extra = source / "cells/r0-99"
    extra.mkdir()
    (extra / "secret").write_text("NEW_REVISION_NOT_ALLOWED")
    manifest = package_corpus(dataset, output)
    assert not any("r0-99" in name for name in manifest["files"])


def test_output_cannot_mutate_input_roots(prepared):
    dataset, source, _ = prepared
    for output in [dataset / "forbidden", source / "forbidden"]:
        with pytest.raises(CorpusError, match="outside"):
            package_corpus(dataset, output)
        assert not output.exists()


@pytest.mark.parametrize(
    "token",
    [
        "hf_" + "a" * 30,
        "sk-proj-" + "b" * 35,
        "sk-ant-api03-" + "c" * 32,
        "ghp_" + "d" * 30,
        "github_pat_" + "e" * 30,
        "ghs_123_aaaa.bbbb.cccc",
        "AKIA" + "A1" * 8,
    ],
)
def test_vendor_credentials_redacted_whole_without_secret_audit(token):
    text = f"before ({token}), after lr=1e-5."
    redacted, counts = redact_credentials(text)
    assert redacted == "before ([REDACTED_CREDENTIAL]), after lr=1e-5."
    assert sum(counts.values()) == 1 and token not in json.dumps(counts)
    assert redact_credentials(redacted) == (redacted, {})


@pytest.mark.parametrize(
    "text",
    [
        'export HF_TOKEN="a_literal_secret_without_digits"',
        "OPENAI_API_KEY='a9VerySecretValue'",
        "AWS_SECRET_ACCESS_KEY=abcdef1234567890+/",
        '"password": "literal-password"',
        '"apiKey": "secret-value"',
        "Authorization: Bearer abcdef123456.xyz",
        '"Authorization": "Basic abcd1234=="',
    ],
)
def test_explicit_secret_values_preserve_surrounding_syntax(text):
    redacted, counts = redact_credentials(text)
    assert "[REDACTED_CREDENTIAL]" in redacted and sum(counts.values()) == 1
    assert redact_credentials(redacted) == (redacted, {})


def test_raw_nested_json_remains_valid_after_credential_replacement():
    original = json.dumps(
        {"content": 'export HF_TOKEN="' + "hf_" + "a" * 30 + '"\ntrain(lr=0.0001)\n'}
    )
    redacted, counts = redact_credentials(original)
    parsed = json.loads(redacted)
    assert parsed["content"] == 'export HF_TOKEN="[REDACTED_CREDENTIAL]"\ntrain(lr=0.0001)\n'
    assert sum(counts.values()) == 1


def test_scientific_numbers_hashes_paths_and_placeholders_preserved_exactly():
    raw = (
        "lr=1e-5 top_p=0.95 accuracy=0.537 seed=42\r\n" + "a" * 64 + "\n"
        'source="scientific/data" endpoint="https://api.example.com/v1"\n'
        'key=123 token=4096 API_KEY=os.getenv("OPENAI_API_KEY")\n'
        'HF_TOKEN=$HF_TOKEN\nHF_TOKEN="${HF_TOKEN}"\nHF_TOKEN="<TOKEN>"\n'
        'api_key=config.api_key\napi_key="YOUR_API_KEY"\npassword="***"\n'
    )
    assert redact_credentials(raw) == (raw, {})


def test_private_key_block_and_unterminated_key():
    raw = "before\n-----BEGIN PRIVATE KEY-----\nabcdef\n-----END PRIVATE KEY-----\nafter"
    result, counts = redact_credentials(raw)
    assert result == "before\n[REDACTED_CREDENTIAL]\nafter" and counts == {"private_key": 1}
    with pytest.raises(CorpusError, match="Unterminated"):
        redact_credentials("-----BEGIN PRIVATE KEY-----\nsecret")


def test_raw_redaction_counted_and_hashes_audited_without_secret(prepared):
    dataset, source, output = prepared
    token = "hf_" + "a" * 30
    trace = source / "cells/r0-01/solve_out_sanitized.txt"
    trace.write_text(f"HF_TOKEN={token}\ntrain(lr=1e-5,temperature=0.7)\n")
    mutate_json(
        dataset / "train/raw_trajectory_index.json",
        lambda x: x[0].update(sha256=sha(trace.read_bytes()), bytes=trace.stat().st_size),
    )
    manifest = package_corpus(dataset, output)
    name = "runs/r0-01/raw_trajectory.txt"
    text = (output / "evidence" / name).read_text()
    assert token not in text and "train(lr=1e-5,temperature=0.7)" in text
    audit_raw = (output / "provenance.json").read_text()
    assert token not in audit_raw
    audit = json.loads(audit_raw)
    item = next(x for x in audit["artifacts"] if x["path"] == name)
    assert item["source_sha256"] == sha(trace.read_bytes())
    assert item["output_sha256"] == manifest["files"][name] != item["source_sha256"]
    assert sum(audit["credential_redactions"].values()) == 1


def test_pinned_missing_card_fails_not_silently_omitted(prepared):
    dataset, source, output = prepared
    (source / "cells/r0-01/wm/cards/exp-01/card.json").unlink()
    with pytest.raises(CorpusError, match="Pinned historical"):
        package_corpus(dataset, output)
