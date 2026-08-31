from __future__ import annotations

import importlib.util
import json
import stat
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
VALIDATOR_PATH = REPO / "rollout" / "validate_c1_final_model.py"
SPEC = importlib.util.spec_from_file_location("validate_c1_final_model", VALIDATOR_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _config() -> dict:
    return {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "boi_token_index": 255999,
        "eoi_token_index": 256000,
        "image_token_index": 262144,
        "mm_tokens_per_image": 256,
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "vocab_size": 262208,
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 34,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 131072,
            "sliding_window": 1024,
            "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "image_size": 896,
            "patch_size": 14,
            "vision_use_head": False,
        },
    }


def _tokenizer_config() -> dict:
    return {
        "tokenizer_class": "GemmaTokenizer",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "added_tokens_decoder": {
            str(index): {"content": token, "special": True}
            for index, token in enumerate(("<pad>", "<eos>", "<bos>", "<unk>"))
        },
    }


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]] | None = None,
) -> None:
    tensors = tensors or {
        "language_model.model.embed_tokens.weight": ("F32", [1]),
    }
    header: dict[str, dict[str, object]] = {}
    payload = bytearray()
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = (elements * validator._DTYPE_BITS[dtype] + 7) // 8
        start = len(payload)
        payload.extend(b"\0" * size)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    task = tmp_path / "task"
    final = task / "final_model"
    base = tmp_path / "base" / "snapshots" / validator.EXPECTED_REVISION
    task.mkdir()
    final.mkdir()
    base.mkdir(parents=True)
    config = _config()
    (base / "config.json").write_text(json.dumps(config))
    config["_name_or_path"] = validator.EXPECTED_MODEL_ID
    (final / "config.json").write_text(json.dumps(config))
    tokenizer = _tokenizer_config()
    (base / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    (final / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    (base / "tokenizer.model").write_bytes(b"exact-pinned-vocabulary")
    (final / "tokenizer.model").write_bytes(b"exact-pinned-vocabulary")
    _write_safetensors(base / "model.safetensors")
    _write_safetensors(final / "model.safetensors")
    return task, final, base


def _validate(task: Path, final: Path, base: Path) -> dict:
    return validator.validate(
        final,
        expected_base_model=validator.EXPECTED_MODEL_ID,
        expected_base_revision=validator.EXPECTED_REVISION,
        expected_base_checkpoint=base,
        task_root=task,
        minimum_weight_bytes=1,
    )


def _mutate_json(path: Path, mutation) -> None:
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))


def test_exact_structurally_valid_fixture_emits_bounded_noncausal_evidence(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    evidence = _validate(task, final, base)
    assert evidence["status"] == "passed"
    assert evidence["scope"] == "structural-declarative-only"
    assert evidence["causal_training_lineage_proven"] is False
    assert evidence["expected_base"] == {
        "model_id": validator.EXPECTED_MODEL_ID,
        "revision": validator.EXPECTED_REVISION,
    }
    assert evidence["final_model"]["path_within_task"] == "final_model"
    assert evidence["final_model"]["weights"]["topology"] == "model.safetensors"
    assert validator.EXPECTED_MODEL_ID not in evidence["limitation"]


def test_rejects_conflicting_model_id(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    _mutate_json(
        final / "config.json", lambda value: value.update(_name_or_path="Qwen/Qwen3-4B-Base")
    )
    with pytest.raises(validator.AttestationError, match="conflicting model identity"):
        _validate(task, final, base)


def test_rejects_wrong_architecture(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    _mutate_json(
        final / "config.json", lambda value: value.update(architectures=["LlamaForCausalLM"])
    )
    with pytest.raises(validator.AttestationError, match="architectures"):
        _validate(task, final, base)


def test_rejects_empty_candidate_config_instead_of_defaulting(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    (final / "config.json").write_text("{}")
    with pytest.raises(validator.AttestationError, match="lacks required architecture field"):
        _validate(task, final, base)


@pytest.mark.parametrize(
    "path",
    (
        ("model_type",),
        ("architectures",),
        ("text_config", "hidden_size"),
        ("vision_config", "patch_size"),
    ),
)
def test_rejects_missing_load_bearing_candidate_fields(
    tmp_path: Path, path: tuple[str, ...]
) -> None:
    task, final, base = _fixture(tmp_path)

    def remove(value: dict) -> None:
        current = value
        for component in path[:-1]:
            current = current[component]
        del current[path[-1]]

    _mutate_json(final / "config.json", remove)
    with pytest.raises(validator.AttestationError, match="lacks required architecture field"):
        _validate(task, final, base)


def test_rejects_missing_pinned_base_invariant(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    _mutate_json(base / "config.json", lambda value: value.pop("model_type"))
    with pytest.raises(validator.AttestationError, match="official base lacks pinned invariant"):
        _validate(task, final, base)


def test_allows_only_reviewed_constructor_default_omissions(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)

    def remove_defaults(value: dict) -> None:
        for path in validator.OPTIONAL_DEFAULTED_INVARIANTS:
            del value[path[0]][path[1]]

    _mutate_json(base / "config.json", remove_defaults)
    _mutate_json(final / "config.json", remove_defaults)
    assert _validate(task, final, base)["status"] == "passed"


@pytest.mark.parametrize(
    ("field", "wrong"),
    (("vocab_size", 32000), ("hidden_size", 4096), ("num_hidden_layers", 32)),
)
def test_rejects_mismatched_core_dimensions(tmp_path: Path, field: str, wrong: int) -> None:
    task, final, base = _fixture(tmp_path)

    def mutate(value: dict) -> None:
        value["text_config"][field] = wrong

    _mutate_json(final / "config.json", mutate)
    with pytest.raises(validator.AttestationError, match=field):
        _validate(task, final, base)


@pytest.mark.parametrize(
    "missing", ("model.safetensors", "tokenizer.model", "tokenizer_config.json")
)
def test_rejects_missing_weights_or_tokenizer(tmp_path: Path, missing: str) -> None:
    task, final, base = _fixture(tmp_path)
    (final / missing).unlink()
    with pytest.raises(validator.AttestationError):
        _validate(task, final, base)


def test_rejects_obvious_alternate_path_even_when_it_exists(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    alternate = tmp_path / "alternate" / "Qwen3"
    alternate.mkdir(parents=True)
    _mutate_json(final / "config.json", lambda value: value.update(_name_or_path=str(alternate)))
    with pytest.raises(validator.AttestationError, match="conflicting model identity"):
        _validate(task, final, base)


@pytest.mark.parametrize(
    ("tensors", "message"),
    (
        ({"alternate.weight": ("F32", [1])}, "tensor-name topology"),
        (
            {"language_model.model.embed_tokens.weight": ("F32", [2])},
            "tensor-shape topology",
        ),
        (
            {"language_model.model.embed_tokens.weight": ("F16", [1])},
            "tensor-dtype topology",
        ),
    ),
)
def test_rejects_candidate_tensor_topology_mismatch(
    tmp_path: Path,
    tensors: dict[str, tuple[str, list[int]]],
    message: str,
) -> None:
    task, final, base = _fixture(tmp_path)
    _write_safetensors(final / "model.safetensors", tensors)
    with pytest.raises(validator.AttestationError, match=message):
        _validate(task, final, base)


def test_accepts_consistent_indexed_safetensors_topology(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    (final / "model.safetensors").unlink()
    tensor = "language_model.model.embed_tokens.weight"
    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(final / shard)
    (final / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4}, "weight_map": {tensor: shard}})
    )
    evidence = _validate(task, final, base)
    assert evidence["final_model"]["weights"]["topology"] == ("model.safetensors.index.json")
    assert evidence["final_model"]["weights"]["tensor_topology_match"] is True


def test_rejects_index_shard_assignment_mismatch(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    (final / "model.safetensors").unlink()
    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(final / shard)
    (final / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"different.weight": shard}})
    )
    with pytest.raises(validator.AttestationError, match="tensor set does not match"):
        _validate(task, final, base)


def test_rejects_unindexed_extra_safetensors_shard(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    _write_safetensors(final / "unexpected.safetensors")
    with pytest.raises(validator.AttestationError, match="extra safetensors shards"):
        _validate(task, final, base)


def test_rejects_safetensors_header_above_bound(tmp_path: Path) -> None:
    path = tmp_path / "oversized.safetensors"
    header_length = validator.MAX_SAFETENSORS_HEADER_BYTES + 1
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", header_length))
        handle.seek(8 + header_length)
        handle.write(b"\0")
    with pytest.raises(validator.AttestationError, match="header length"):
        validator._safetensor_topology(path, allow_link=False)


def test_rejects_final_model_directory_symlink(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    checkpoint = task / "checkpoints" / "trained"
    checkpoint.parent.mkdir()
    final.rename(checkpoint)
    final.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(validator.AttestationError, match="not a symlink"):
        _validate(task, final, base)


def test_rejects_unsloth_substitution_even_inside_task(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    local = task / "checkpoints" / "unsloth-gemma"
    local.mkdir(parents=True)
    _mutate_json(final / "config.json", lambda value: value.update(_name_or_path=str(local)))
    with pytest.raises(validator.AttestationError, match="Unsloth"):
        _validate(task, final, base)


def test_allows_a_declared_local_intermediate_checkpoint(tmp_path: Path) -> None:
    task, final, base = _fixture(tmp_path)
    local = task / "checkpoints" / "sft-1"
    local.mkdir(parents=True)
    _mutate_json(final / "config.json", lambda value: value.update(_name_or_path=str(local)))
    assert _validate(task, final, base)["status"] == "passed"


def test_private_attestation_writer_uses_mode_0600(tmp_path: Path) -> None:
    record = tmp_path / "attestation.json"
    validator._atomic_private_json(record, {"schema_version": validator.SCHEMA})
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert json.loads(record.read_text()) == {"schema_version": validator.SCHEMA}
