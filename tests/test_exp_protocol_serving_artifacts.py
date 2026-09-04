"""Byte/layout/publication tests; native cases use tiny owned CPU assets only."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import pytest

from awm.exp_protocol import serving_artifacts as serving


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def safe_file(path, tensors=None, *, payload_bytes=None):
    tensors = tensors or {"weight": ("F32", [1], b"\0" * 4)}
    header, payload = {"__metadata__": {"format": "pt"}}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.write(payload)
        if payload_bytes is not None:
            stream.truncate(8 + len(raw) + payload_bytes)


def artifact(path, *, generation=None, gemma=False):
    path.mkdir()
    config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "n_positions": 16,
        "vocab_size": 4,
    }
    if gemma:
        config = {
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"],
            "text_config": {"model_type": "gemma3_text"},
            "vision_config": {"model_type": "siglip_vision_model"},
        }
        write_json(
            path / "preprocessor_config.json", {"image_processor_type": "Gemma3ImageProcessor"}
        )
        write_json(path / "processor_config.json", {"processor_class": "Gemma3Processor"})
    write_json(path / "config.json", config)
    write_json(
        path / "generation_config.json", generation or {"do_sample": True, "temperature": 0.7}
    )
    write_json(
        path / "tokenizer.json",
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "[PAD]": 1, "hello": 2, "[EOS]": 3},
                "unk_token": "[UNK]",
            },
        },
    )
    write_json(
        path / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "eos_token": "[EOS]",
        },
    )
    safe_file(path / "model.safetensors")
    return path


def generation_hash(path):
    return hashlib.sha256((path / "generation_config.json").read_bytes()).hexdigest()


def publish(source, destination, identity, scope, **kwargs):
    return serving.publish_serving_artifact(
        source,
        destination,
        identity,
        generation_hash(source),
        session_dir=scope,
        task_id="synthetic-cpu-task",
        reference_id="exp-07-selected",
        load_metadata=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "generation",
    [
        {"do_sample": True, "temperature": 0.7, "top_k": 64},
        {"do_sample": False, "temperature": 0.0, "top_k": 0, "eos_token_id": [1, 106]},
    ],
)
def test_exact_selected_bytes_survive_new_publication(tmp_path, generation):
    source = artifact(tmp_path / "source", generation=generation)
    # Freeze before the simulated user's measurement, not after a file changes.
    expected = serving.snapshot_serving_artifact(source)
    frozen_generation = (source / "generation_config.json").read_bytes()
    (source / "optimizer.pt").write_bytes(b"opaque training state not copied")
    (source / "run.log").write_text("not serving data")
    result = publish(source, tmp_path / "final", expected, tmp_path)
    assert result["status"] == "published" and result["backup"] is None
    assert (tmp_path / "final/generation_config.json").read_bytes() == frozen_generation
    assert not (tmp_path / "final/optimizer.pt").exists()
    assert not (tmp_path / "final/run.log").exists()
    assert serving.snapshot_serving_artifact(tmp_path / "final")["content"] == expected["content"]
    assert result["published_verification"]["model_load"] == "not_performed"
    assert result["scientific_validation"] == "not_performed"


def test_normalized_but_unselected_checkpoint_is_refused(tmp_path):
    selected = artifact(tmp_path / "selected", generation={"do_sample": False, "temperature": 0.0})
    selected_hash = generation_hash(selected)
    normalized = artifact(
        tmp_path / "normalized", generation={"do_sample": False, "temperature": 1.0}
    )
    expected = serving.snapshot_serving_artifact(normalized)
    with pytest.raises(serving.ServingArtifactError, match="explicitly selected"):
        serving.verify_serving_artifact(normalized, expected, selected_hash, load_metadata=False)
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ],
)
def test_missing_required_metadata_fails(tmp_path, name):
    source = artifact(tmp_path / "source", gemma=True)
    (source / name).unlink()
    with pytest.raises(serving.ServingArtifactError, match="required|requires"):
        serving.snapshot_serving_artifact(source)


def sharded(path):
    (path / "model.safetensors").unlink()
    safe_file(path / "model-00001-of-00002.safetensors", {"first": ("F32", [1], b"\0" * 4)})
    safe_file(path / "model-00002-of-00002.safetensors", {"second": ("F32", [2], b"\0" * 8)})
    write_json(
        path / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 12},
            "weight_map": {
                "first": "model-00001-of-00002.safetensors",
                "second": "model-00002-of-00002.safetensors",
            },
        },
    )


def test_sharded_index_checks_actual_tensor_inventory(tmp_path):
    source = artifact(tmp_path / "source")
    sharded(source)
    expected = serving.snapshot_serving_artifact(source)
    assert expected["content"]["weights"]["layout"] == "indexed"
    result = publish(source, tmp_path / "final", expected, tmp_path)
    assert len(result["copied_files"]) == 7


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "empty",
        "mixed",
        "orphan",
        "traversal",
        "wrong_tensor",
        "wrong_size",
        "non_mapping",
    ],
)
def test_invalid_or_ambiguous_shards_are_not_certified(tmp_path, mutation):
    source = artifact(tmp_path / "source")
    sharded(source)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if mutation == "missing":
        (source / "model-00002-of-00002.safetensors").unlink()
    elif mutation == "empty":
        (source / "model-00002-of-00002.safetensors").write_bytes(b"")
    elif mutation == "mixed":
        (source / "pytorch_model.bin").write_bytes(b"not loaded")
    elif mutation == "orphan":
        safe_file(source / "unindexed.safetensors")
    elif mutation == "traversal":
        index["weight_map"]["first"] = "../outside.safetensors"
    elif mutation == "wrong_tensor":
        index["weight_map"]["nonexistent"] = index["weight_map"].pop("first")
    elif mutation == "wrong_size":
        index["metadata"]["total_size"] = 999
    else:
        index["weight_map"] = ["model-00001-of-00002.safetensors"]
    write_json(index_path, index)
    with pytest.raises(serving.ServingArtifactError):
        serving.snapshot_serving_artifact(source)


@pytest.mark.parametrize(
    "kind", ["file_symlink", "root_symlink", "hardlink", "fifo", "unknown_asset"]
)
def test_unexpected_links_special_files_and_assets_are_explicitly_unsupported(tmp_path, kind):
    source = artifact(tmp_path / "source")
    if kind == "file_symlink":
        (source / "generation_config.json").unlink()
        external = tmp_path / "external.json"
        external.write_text("{}")
        (source / "generation_config.json").symlink_to(external)
    elif kind == "root_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    elif kind == "hardlink":
        os.link(source / "model.safetensors", tmp_path / "external-weights")
    elif kind == "fifo":
        os.mkfifo(source / "pipe")
    else:
        (source / "custom_inference_state.json").write_text("{}")
    with pytest.raises(serving.UnsupportedServingArtifact) as caught:
        serving.snapshot_serving_artifact(source)
    assert caught.value.report["status"] == "unsupported"


def test_pickle_is_only_opaque_identity_by_explicit_opt_in(tmp_path):
    source = artifact(tmp_path / "source")
    (source / "model.safetensors").unlink()
    (source / "pytorch_model.bin").write_bytes(b"arbitrary bytes; must never be unpickled")
    with pytest.raises(serving.UnsupportedServingArtifact, match="opaque"):
        serving.snapshot_serving_artifact(source)
    expected = serving.snapshot_serving_artifact(source, allow_opaque_weights=True)
    verified = serving.verify_serving_artifact(
        source, expected, generation_hash(source), load_metadata=False, allow_opaque_weights=True
    )
    assert verified["weight_structure"] == "opaque_bytes_only_no_pickle_load"
    assert verified["model_load"] == "not_performed"


def test_vocabulary_tokens_are_not_misread_as_configuration_directives(tmp_path):
    source = artifact(tmp_path / "source")
    write_json(source / "vocab.json", {"custom_generate": 0, "tokenizer_file": 1, "hello": 2})
    write_json(source / "added_tokens.json", {"configuration_files": 3})
    expected = serving.snapshot_serving_artifact(source)
    assert {"vocab.json", "added_tokens.json"}.issubset(
        entry["path"] for entry in expected["content"]["files"]
    )


@pytest.mark.parametrize("mutation", ["truncated", "bad_offsets", "wrong_shape", "extra_payload"])
def test_safetensors_structure_is_checked_without_model_loading(tmp_path, mutation):
    source = artifact(tmp_path / "source")
    weight = source / "model.safetensors"
    if mutation == "truncated":
        weight.write_bytes(b"\0" * 7)
    elif mutation == "extra_payload":
        weight.write_bytes(weight.read_bytes() + b"\0")
    else:
        header = {
            "weight": {
                "dtype": "F32",
                "shape": [2] if mutation == "wrong_shape" else [1],
                "data_offsets": [2, 6] if mutation == "bad_offsets" else [0, 4],
            }
        }
        raw = json.dumps(header).encode()
        weight.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    with pytest.raises(serving.ServingArtifactError):
        serving.snapshot_serving_artifact(source)


def test_default_existing_destination_is_never_merged_or_overwritten(tmp_path):
    source = artifact(tmp_path / "source")
    destination = artifact(tmp_path / "final", generation={"temperature": 0.123})
    old = serving.snapshot_serving_artifact(destination)
    with pytest.raises(serving.ServingPublicationError, match="already exists") as caught:
        publish(source, destination, serving.snapshot_serving_artifact(source), tmp_path)
    assert serving.snapshot_serving_artifact(destination)["content"] == old["content"]
    assert not Path(caught.value.report["stage"]).exists()


def test_explicit_replacement_preserves_unique_backup_and_quiescence_boundary(tmp_path):
    source = artifact(tmp_path / "source")
    destination = artifact(tmp_path / "final", generation={"temperature": 0.123})
    old = serving.snapshot_serving_artifact(destination)
    expected = serving.snapshot_serving_artifact(source)
    with pytest.raises(serving.ServingArtifactError, match="quiescence"):
        publish(source, destination, expected, tmp_path, replace=True, expected_old_identity=old)
    result = publish(
        source,
        destination,
        expected,
        tmp_path,
        replace=True,
        expected_old_identity=old,
        target_quiescent=True,
        quiescence_evidence="Owned fixture had no running consumers.",
    )
    assert serving.snapshot_serving_artifact(Path(result["backup"]))["content"] == old["content"]
    assert serving.snapshot_serving_artifact(destination)["content"] == expected["content"]
    assert result["target_quiescence"]["independent_verification"] == "not_performed"


@pytest.mark.parametrize("fault", ["copy", "source_changed", "stage_changed"])
def test_copy_and_source_faults_keep_old_target_and_failed_stage(tmp_path, monkeypatch, fault):
    source = artifact(tmp_path / "source")
    destination = artifact(tmp_path / "final", generation={"temperature": 0.123})
    old = serving.snapshot_serving_artifact(destination)
    expected = serving.snapshot_serving_artifact(source)
    original_copy = serving._copy_regular
    changed = False

    def faulty_copy(source_fd, stage_fd, entry):
        nonlocal changed
        original_copy(source_fd, stage_fd, entry)
        if not changed:
            changed = True
            if fault == "copy":
                raise OSError("injected copy failure")
            if fault == "source_changed":
                write_json(source / "generation_config.json", {"temperature": 0.001})
            else:
                fd = os.open(entry["path"], os.O_WRONLY | os.O_TRUNC, dir_fd=stage_fd)
                os.write(fd, b"bad")
                os.close(fd)

    monkeypatch.setattr(serving, "_copy_regular", faulty_copy)
    with pytest.raises(serving.ServingPublicationError) as caught:
        publish(
            source,
            destination,
            expected,
            tmp_path,
            replace=True,
            expected_old_identity=old,
            target_quiescent=True,
            quiescence_evidence="Owned synthetic fixture.",
        )
    assert Path(caught.value.report["stage"]).is_dir()
    assert serving.snapshot_serving_artifact(destination)["content"] == old["content"]
    assert caught.value.report["status"] == "failed"
    assert caught.value.report["rollback"] == "old_incumbent_untouched"


def test_raced_new_destination_is_not_overwritten_even_when_empty(tmp_path, monkeypatch):
    source = artifact(tmp_path / "source")
    destination = tmp_path / "final"
    original = serving._rename_noreplace

    def raced(parent_fd, source_name, destination_name):
        if ".stage-" in source_name:
            destination.mkdir()
        return original(parent_fd, source_name, destination_name)

    monkeypatch.setattr(serving, "_rename_noreplace", raced)
    with pytest.raises(serving.ServingPublicationError) as caught:
        publish(source, destination, serving.snapshot_serving_artifact(source), tmp_path)
    assert destination.is_dir() and list(destination.iterdir()) == []
    assert Path(caught.value.report["stage"]).is_dir()


@pytest.mark.parametrize("rollback_failure", [False, True])
def test_interruption_after_backup_is_recoverable_and_rollback_failure_is_honest(
    tmp_path, monkeypatch, rollback_failure
):
    source = artifact(tmp_path / "source")
    destination = artifact(tmp_path / "final", generation={"temperature": 0.123})
    old = serving.snapshot_serving_artifact(destination)
    original = serving._rename_noreplace

    def interrupted(parent_fd, source_name, destination_name):
        if source_name == "final":
            original(parent_fd, source_name, destination_name)
            raise KeyboardInterrupt("after backup rename")
        if rollback_failure and ".backup-" in source_name:
            raise OSError("injected rollback failure")
        return original(parent_fd, source_name, destination_name)

    monkeypatch.setattr(serving, "_rename_noreplace", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        publish(
            source,
            destination,
            serving.snapshot_serving_artifact(source),
            tmp_path,
            replace=True,
            expected_old_identity=old,
            target_quiescent=True,
            quiescence_evidence="Owned synthetic fixture.",
        )
    record = caught.value.publication_record
    assert record["status"] == "failed" and Path(record["stage"]).is_dir()
    if rollback_failure:
        assert not destination.exists()
        assert (
            "manual_recovery" in record["rollback"]
            and "rollback failure" in record["rollback_error"]
        )
        assert (
            serving.snapshot_serving_artifact(Path(record["backup"]))["content"] == old["content"]
        )
    else:
        assert record["rollback"] == "old_incumbent_restored"
        assert serving.snapshot_serving_artifact(destination)["content"] == old["content"]
    assert json.loads(Path(record["record_path"]).read_text())["status"] == "failed"


def test_post_publish_failure_preserves_new_failed_directory_and_restores_old(
    tmp_path, monkeypatch
):
    source = artifact(tmp_path / "source")
    destination = artifact(tmp_path / "final", generation={"temperature": 0.123})
    old = serving.snapshot_serving_artifact(destination)
    original = serving._rename_noreplace
    interrupted = False

    def stop_after_publication(parent_fd, source_name, destination_name):
        nonlocal interrupted
        original(parent_fd, source_name, destination_name)
        if ".stage-" in source_name and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after publication rename")

    monkeypatch.setattr(serving, "_rename_noreplace", stop_after_publication)
    with pytest.raises(KeyboardInterrupt) as caught:
        publish(
            source,
            destination,
            serving.snapshot_serving_artifact(source),
            tmp_path,
            replace=True,
            expected_old_identity=old,
            target_quiescent=True,
            quiescence_evidence="Owned synthetic fixture.",
        )
    record = caught.value.publication_record
    assert Path(record["failed_publication_path"]).is_dir()
    assert record["rollback"] == "old_incumbent_restored"
    assert serving.snapshot_serving_artifact(destination)["content"] == old["content"]


def test_unowned_destination_appearing_after_backup_is_never_reclaimed(tmp_path, monkeypatch):
    source = artifact(tmp_path / "source")
    destination = artifact(tmp_path / "final", generation={"temperature": 0.123})
    old = serving.snapshot_serving_artifact(destination)
    original = serving._rename_noreplace

    def competing(parent_fd, source_name, destination_name):
        if ".stage-" in source_name:
            destination.mkdir()
            (destination / "foreign.txt").write_text("must remain")
        return original(parent_fd, source_name, destination_name)

    monkeypatch.setattr(serving, "_rename_noreplace", competing)
    with pytest.raises(serving.ServingPublicationError) as caught:
        publish(
            source,
            destination,
            serving.snapshot_serving_artifact(source),
            tmp_path,
            replace=True,
            expected_old_identity=old,
            target_quiescent=True,
            quiescence_evidence="Owned fixture; controlled competing writer is a negative case.",
        )
    assert (destination / "foreign.txt").read_text() == "must remain"
    assert "manual_recovery" in caught.value.report["rollback"]
    assert (
        serving.snapshot_serving_artifact(Path(caught.value.report["backup"]))["content"]
        == old["content"]
    )


@pytest.mark.parametrize("target", ["scope", "outside", "overlap", "broad_scope"])
def test_publication_scope_never_authorizes_broad_or_unowned_targets(tmp_path, target):
    source = artifact(tmp_path / "source")
    expected = serving.snapshot_serving_artifact(source)
    destination = tmp_path / "final"
    scope = tmp_path
    if target == "scope":
        destination = tmp_path
    elif target == "outside":
        destination = tmp_path.parent / "outside-final"
    elif target == "overlap":
        destination = source / "nested"
    else:
        scope = Path("/tmp")
    with pytest.raises(serving.ServingArtifactError):
        publish(source, destination, expected, scope)


def test_stale_old_identity_cannot_authorize_replacement(tmp_path):
    source = artifact(tmp_path / "source")
    destination = artifact(tmp_path / "final")
    old = serving.snapshot_serving_artifact(destination)
    write_json(destination / "generation_config.json", {"temperature": 0.999})
    with pytest.raises(serving.ServingPublicationError):
        publish(
            source,
            destination,
            serving.snapshot_serving_artifact(source),
            tmp_path,
            replace=True,
            expected_old_identity=old,
            target_quiescent=True,
            quiescence_evidence="Owned synthetic fixture.",
        )
    assert json.loads((destination / "generation_config.json").read_text())["temperature"] == 0.999


def test_posthoc_snapshot_does_not_claim_old_evaluation_identity(tmp_path):
    source = artifact(tmp_path / "source")
    before = serving.snapshot_serving_artifact(source)
    write_json(source / "generation_config.json", {"temperature": 0.9})
    with pytest.raises(serving.ServingArtifactError):
        serving.verify_serving_artifact(
            source, before, serving._generation_hash(before), load_metadata=False
        )
    after = serving.snapshot_serving_artifact(source)
    result = serving.verify_serving_artifact(
        source, after, generation_hash(source), load_metadata=False
    )
    assert "prior_measurement" in after["proof"]
    assert "not_verified_here" in result["measurement_binding"]


def test_cooperating_publications_cannot_reenter_same_destination(tmp_path, monkeypatch):
    source = artifact(tmp_path / "source")
    expected = serving.snapshot_serving_artifact(source)
    original = serving._copy_regular
    attempted = False

    def reentrant(source_fd, stage_fd, entry):
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(
                serving.ServingPublicationError, match="another guarded publication"
            ):
                publish(source, tmp_path / "final", expected, tmp_path)
        return original(source_fd, stage_fd, entry)

    monkeypatch.setattr(serving, "_copy_regular", reentrant)
    assert publish(source, tmp_path / "final", expected, tmp_path)["status"] == "published"


def test_caller_cannot_mutate_frozen_expectation_during_copy(tmp_path, monkeypatch):
    source = artifact(tmp_path / "source")
    expected = serving.snapshot_serving_artifact(source)
    frozen = expected["identity_sha256"]
    original = serving._copy_regular
    changed = False

    def mutate_caller_argument(source_fd, stage_fd, entry):
        nonlocal changed
        if not changed:
            changed = True
            expected["content"]["files"].clear()
        return original(source_fd, stage_fd, entry)

    monkeypatch.setattr(serving, "_copy_regular", mutate_caller_argument)
    result = publish(source, tmp_path / "final", expected, tmp_path)
    assert result["published_verification"]["identity_sha256"] == frozen
    assert result["expected_identity"]["content"]["files"]


def test_real_e5_fresh_record_is_not_upgraded_to_scientific_completion(tmp_path):
    import base64

    from exp_protocol_cards import plan_card

    from awm.exp_protocol import execution, lock, preflight, schema

    template = artifact(tmp_path / "template")
    output = tmp_path / "produced"
    payload = {p.name: base64.b64encode(p.read_bytes()).decode() for p in template.iterdir()}
    script = tmp_path / "produce.py"
    script.write_text(
        "import base64\nfrom pathlib import Path\n"
        f"output = Path({str(output)!r})\n"
        f"payload = {payload!r}\n"
        "for name, data in payload.items():\n"
        "    (output / name).write_bytes(base64.b64decode(data))\n"
    )
    card = plan_card()
    card["setup"].update(
        data=[],
        parent_checkpoint={"path": str(template), "origin": "base_model"},
        method={"family": "other"},
        output_dir=str(output),
        command={
            "argv": [sys.executable, str(script)],
            "script": str(script),
            "cwd": str(tmp_path),
        },
        execution={"output_evidence": "fresh-directory"},
    )
    path = tmp_path / "memory/cards/exp-01.yaml"
    path.parent.mkdir(parents=True)
    schema.dump_card(path, card)
    before = preflight.run_preflight(card, tmp_path, pitfalls=[])
    assert before["summary"]["fail"] == 0
    lock.write_lock(path, card, before["summary"])
    attempt = execution.run_card(path, tmp_path)
    assert attempt["observed_returncode"] == 0 and attempt["artifacts"]["status"] == "observed"
    expected = serving.snapshot_serving_artifact(output)
    publication = publish(output, tmp_path / "final", expected, tmp_path)
    assert (
        attempt["scientific_validation"] == publication["scientific_validation"] == "not_performed"
    )
    assert publication["published_verification"]["model_load"] == "not_performed"


def tiny_tokenizer(path, *, gemma=False):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GemmaTokenizerFast, PreTrainedTokenizerFast

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<image_soft_token>": 4,
        "<start_of_image>": 5,
        "<end_of_image>": 6,
        "hello": 7,
        "<end_of_turn>": 8,
    }
    vocab.update({f"word-{i}": i for i in range(9, 16)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    cls = GemmaTokenizerFast if gemma else PreTrainedTokenizerFast
    kwargs = (
        {
            "extra_special_tokens": {
                "image_token": "<image_soft_token>",
                "boi_token": "<start_of_image>",
                "eoi_token": "<end_of_image>",
            }
        }
        if gemma
        else {}
    )
    tok = cls(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        **kwargs,
    )
    tok.save_pretrained(path)
    return tok


def test_native_metadata_loaders_are_real_but_do_not_load_fake_weights(tmp_path):
    pytest.importorskip("transformers")
    source = artifact(tmp_path / "source")
    expected = serving.snapshot_serving_artifact(source)
    checked = serving.verify_serving_artifact(source, expected, generation_hash(source))
    assert checked["native_metadata_load"]["classes"]["config"] == "GPT2Config"
    assert checked["native_metadata_load"]["classes"]["tokenizer"] == "PreTrainedTokenizerFast"
    # The deliberately synthetic weight tensor is not a GPT2 state dict. This
    # proves the reported boundary, not model loadability or model quality.
    assert checked["model_load"] == checked["scientific_validation"] == "not_performed"


def test_native_legacy_gpt2_vocabulary_and_merges_are_preserved(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2TokenizerFast

    source = artifact(tmp_path / "source")
    symbols = ["<pad>", "<unk>", "<bos>", "<eos>", "tokenizer_file", "custom_generate"]
    symbols += sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(
        models.BPE(
            vocab={symbol: i for i, symbol in enumerate(symbols)}, merges=[], unk_token="<unk>"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = GPT2TokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tok.save_pretrained(source)
    expected = serving.snapshot_serving_artifact(source)
    names = {entry["path"] for entry in expected["content"]["files"]}
    assert {"vocab.json", "merges.txt", "tokenizer.json"}.issubset(names)
    result = serving.publish_serving_artifact(
        source,
        tmp_path / "final",
        expected,
        generation_hash(source),
        session_dir=tmp_path,
        task_id="legacy-tokenizer-fixture",
        reference_id="frozen-selected-artifact",
    )
    assert result["status"] == "published"
    assert (tmp_path / "final/vocab.json").read_bytes() == (source / "vocab.json").read_bytes()


@pytest.mark.parametrize(
    "sampled,shards", [(False, False), (True, False), (False, True), (True, True)]
)
def test_native_e4_selected_save_and_e8_publication_preserve_exact_settings(
    tmp_path, sampled, shards
):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    from awm.exp_protocol.save_contract import GenerationSaveContract

    source = tmp_path / "native-source"
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=8,
            vocab_size=16,
            bos_token_id=2,
            eos_token_id=3,
        )
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = 0.0
    selected = (
        json.dumps(
            {
                "do_sample": sampled,
                "temperature": 0.7 if sampled else 0.0,
                "top_k": 4 if sampled else 0,
                "eos_token_id": [3, 8],
            },
            indent=1,
        )
        + "\n"
    ).encode()
    saves = GenerationSaveContract()
    with saves.saving(model, source, selected_serving_json=selected):
        model.save_pretrained(source, max_shard_size="1KB" if shards else "5GB")
    tiny_tokenizer(source)
    event = saves.records[-1]
    assert event["selected_serving_hash"] == hashlib.sha256(selected).hexdigest()
    expected = serving.snapshot_serving_artifact(source)
    result = serving.publish_serving_artifact(
        source,
        tmp_path / "final",
        expected,
        event["selected_serving_hash"],
        session_dir=tmp_path,
        task_id="native-cpu-task",
        reference_id="exp-07-selected",
    )
    assert result["status"] == "published"
    assert (tmp_path / "final/generation_config.json").read_bytes() == selected
    assert (
        result["source_verification"]["native_metadata_load"]["status"]
        == "loaded_local_metadata_only"
    )
    assert result["scientific_validation"] == "not_performed"
    assert expected["content"]["weights"]["layout"] == ("indexed" if shards else "single")


def test_native_gemma3_profile_and_processor_metadata_without_forward(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers import (
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3ImageProcessor,
        Gemma3Processor,
        Gemma3TextConfig,
        SiglipVisionConfig,
    )

    from awm.exp_protocol.save_contract import GenerationSaveContract

    source = tmp_path / "gemma-source"
    text = Gemma3TextConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=8,
        bos_token_id=2,
        eos_token_id=3,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=4,
    )
    config = Gemma3Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        mm_tokens_per_image=16,
        image_token_index=4,
        boi_token_index=5,
        eoi_token_index=6,
    )
    model = Gemma3ForConditionalGeneration(config)
    selected = b'{"do_sample":false,"temperature":0.0,"eos_token_id":[3,8]}\n'
    with GenerationSaveContract().saving(model, source, selected_serving_json=selected):
        model.save_pretrained(source)
    tokenizer = tiny_tokenizer(source, gemma=True)
    processor = Gemma3Processor(
        image_processor=Gemma3ImageProcessor(size={"height": 16, "width": 16}),
        tokenizer=tokenizer,
        image_seq_length=16,
    )
    processor.save_pretrained(source)
    expected = serving.snapshot_serving_artifact(source)
    verified = serving.verify_serving_artifact(
        source, expected, hashlib.sha256(selected).hexdigest()
    )
    assert verified["native_metadata_load"]["classes"]["processor"] == "Gemma3Processor"
    result = serving.publish_serving_artifact(
        source,
        tmp_path / "final",
        expected,
        hashlib.sha256(selected).hexdigest(),
        session_dir=tmp_path,
        task_id="gemma-cpu-task",
        reference_id="selected-gemma-native-fixture",
    )
    assert result["status"] == "published"
    assert {"preprocessor_config.json", "processor_config.json"}.issubset(result["copied_files"])
    assert result["scientific_validation"] == "not_performed"


def test_cpu_copy_hash_cost(tmp_path, capsys):
    source = artifact(tmp_path / "source")
    payload = b"\0" * (16 * 1024 * 1024)
    safe_file(
        source / "model.safetensors", {"synthetic_weight": ("F32", [len(payload) // 4], payload)}
    )
    started = time.perf_counter()
    expected = serving.snapshot_serving_artifact(source)
    snapshotted = time.perf_counter()
    serving.verify_serving_artifact(source, expected, generation_hash(source), load_metadata=False)
    checked = time.perf_counter()
    result = publish(source, tmp_path / "final", expected, tmp_path)
    ended = time.perf_counter()
    assert result["status"] == "published"
    with capsys.disabled():
        print(
            "SERVING_CPU_COST "
            + json.dumps(
                {
                    "payload_bytes": len(payload),
                    "snapshot_seconds": snapshotted - started,
                    "verify_seconds": checked - snapshotted,
                    "publish_seconds": ended - checked,
                    "native_metadata_loading": "not_performed",
                },
                sort_keys=True,
            )
        )
