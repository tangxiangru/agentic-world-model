#!/usr/bin/env python3
"""Fail closed on an accidentally substituted study submission.

This is a structural and declarative check, not a training-lineage proof.  It
checks that ``final_model`` is a self-contained Hugging Face-style Gemma 3 4B
artifact whose declared architecture, tokenizer, and weight topology agree
with the separately attested official checkpoint.  A compatible artifact
could still have been produced by an unobserved process; proving causal
fine-tuning lineage would require an independent training observer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_MODEL_ID = "google/gemma-3-4b-pt"
EXPECTED_REVISION = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
SCHEMA = "awm-final-model-structural-attestation-v2"
MINIMUM_WEIGHT_BYTES = 1_000_000_000
MAX_SAFETENSORS_HEADER_BYTES = 16_000_000
MAX_WEIGHT_INDEX_BYTES = 16_000_000
MAX_TENSOR_COUNT = 100_000
MAX_TENSOR_NAME_BYTES = 4_096
MAX_TENSOR_RANK = 16
MAX_TENSOR_DIMENSION = (1 << 63) - 1

# These values define model identity, tensor dimensions, or multimodal token
# interpretation.  A normal save_pretrained Gemma artifact records all of them.
INVARIANTS: dict[tuple[str, ...], Any] = {
    ("model_type",): "gemma3",
    ("architectures",): ["Gemma3ForConditionalGeneration"],
    ("boi_token_index",): 255999,
    ("eoi_token_index",): 256000,
    ("image_token_index",): 262144,
    ("mm_tokens_per_image",): 256,
    ("text_config", "model_type"): "gemma3_text",
    ("text_config", "vocab_size"): 262208,
    ("text_config", "hidden_size"): 2560,
    ("text_config", "intermediate_size"): 10240,
    ("text_config", "num_hidden_layers"): 34,
    ("text_config", "num_attention_heads"): 8,
    ("text_config", "num_key_value_heads"): 4,
    ("text_config", "head_dim"): 256,
    ("text_config", "max_position_embeddings"): 131072,
    ("text_config", "sliding_window"): 1024,
    ("text_config", "rope_scaling"): {"factor": 8.0, "rope_type": "linear"},
    ("vision_config", "model_type"): "siglip_vision_model",
    ("vision_config", "hidden_size"): 1152,
    ("vision_config", "intermediate_size"): 4304,
    ("vision_config", "num_hidden_layers"): 27,
    ("vision_config", "num_attention_heads"): 16,
    ("vision_config", "image_size"): 896,
    ("vision_config", "patch_size"): 14,
    ("vision_config", "vision_use_head"): False,
}

# These five text-config values are deliberately omitted by the pinned official
# config because they equal Gemma3TextConfig's stable constructor defaults.
# Everything recorded in the pinned config remains mandatory in the candidate.
OPTIONAL_DEFAULTED_INVARIANTS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("text_config", "vocab_size"),
        ("text_config", "num_attention_heads"),
        ("text_config", "num_key_value_heads"),
        ("text_config", "head_dim"),
        ("text_config", "max_position_embeddings"),
    }
)

IDENTITY_KEYS = {
    "_name_or_path",
    "name_or_path",
    "model_id",
    "model_name_or_path",
    "pretrained_model_name_or_path",
    "base_model",
    "base_model_id",
    "base_model_name_or_path",
}
SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "pad_token", "unk_token")

_MISSING = object()
_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "C64": 64,
    "C128": 128,
    "U4": 4,
    "I4": 4,
}


class AttestationError(RuntimeError):
    """The candidate does not satisfy the study's structural contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, allow_link: bool = False) -> bool:
    return path.is_file() and (allow_link or not path.is_symlink())


def _load_json(path: Path, *, max_bytes: int, allow_link: bool = False) -> Any:
    if not _regular(path, allow_link=allow_link):
        raise AttestationError(f"required regular artifact is missing: {path.name}")
    if path.stat().st_size > max_bytes:
        raise AttestationError(f"JSON artifact exceeds the structural-check limit: {path.name}")
    try:
        return json.loads(path.read_text(), object_pairs_hook=_json_object_without_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AttestationError(f"invalid JSON artifact: {path.name}") from exc


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _dig(value: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    current: Any = value
    for component in path:
        if not isinstance(current, dict) or component not in current:
            return default
        current = current[component]
    return current


def _check_invariants(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if "auto_map" in candidate:
        raise AttestationError("candidate requires unpinned custom model code")
    observed: dict[str, Any] = {}
    for path, canonical in INVARIANTS.items():
        label = ".".join(path)
        base_value = _dig(base, path, _MISSING)
        if base_value is _MISSING:
            if path not in OPTIONAL_DEFAULTED_INVARIANTS:
                raise AttestationError(f"official base lacks pinned invariant: {label}")
            base_value = canonical
        if base_value != canonical:
            raise AttestationError(f"official base disagrees with pinned invariant: {label}")
        candidate_value = _dig(candidate, path, _MISSING)
        if candidate_value is _MISSING:
            if path not in OPTIONAL_DEFAULTED_INVARIANTS:
                raise AttestationError(f"candidate lacks required architecture field: {label}")
            candidate_value = canonical
        if candidate_value != base_value:
            raise AttestationError(f"candidate architecture mismatch: {label}")
        observed[label] = candidate_value
    return observed


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))
    else:
        yield path, value


def _check_declared_identities(
    documents: list[dict[str, Any]],
    *,
    base_checkpoint: Path,
    task_root: Path,
) -> list[str]:
    accepted_fields: list[str] = []
    for document in documents:
        for path, value in _walk(document):
            key = path[-1].lower() if path else ""
            if "unsloth" in key or (isinstance(value, str) and "unsloth" in value.lower()):
                raise AttestationError("candidate declares an Unsloth/alternate model substitution")
            if key not in IDENTITY_KEYS or value in (None, ""):
                continue
            if not isinstance(value, str):
                raise AttestationError(
                    f"candidate model identity is not a string: {'.'.join(path)}"
                )
            if value == EXPECTED_MODEL_ID:
                accepted_fields.append(".".join(path))
                continue
            declared = Path(value)
            if declared.is_absolute():
                resolved = declared.resolve(strict=False)
            else:
                resolved = (task_root / declared).resolve(strict=False)
            if resolved == base_checkpoint or (_inside(resolved, task_root) and resolved.exists()):
                accepted_fields.append(".".join(path))
                continue
            raise AttestationError(
                f"candidate declares a conflicting model identity: {'.'.join(path)}"
            )
    return sorted(set(accepted_fields))


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    if value is None:
        return None
    raise AttestationError("special-token declaration has an unsupported shape")


def _added_token_signature(config: dict[str, Any]) -> list[tuple[int, str, bool]]:
    raw = config.get("added_tokens_decoder")
    if not isinstance(raw, dict):
        raise AttestationError("tokenizer_config.json lacks added_tokens_decoder")
    signature: list[tuple[int, str, bool]] = []
    for token_id, value in raw.items():
        if not isinstance(value, dict) or not isinstance(value.get("content"), str):
            raise AttestationError("invalid added token declaration")
        try:
            numeric_id = int(token_id)
        except (TypeError, ValueError) as exc:
            raise AttestationError("invalid added token id") from exc
        signature.append((numeric_id, value["content"], bool(value.get("special"))))
    return sorted(signature)


def _fast_tokenizer_signature(document: dict[str, Any]) -> str:
    model = document.get("model")
    added = document.get("added_tokens")
    if not isinstance(model, dict) or not isinstance(added, list):
        raise AttestationError("tokenizer.json lacks model/added-token topology")
    vocab = model.get("vocab")
    if not isinstance(vocab, (dict, list)):
        raise AttestationError("tokenizer.json lacks a vocabulary")
    semantic = {
        "model_type": model.get("type"),
        "vocab": vocab,
        "added_tokens": [
            {
                "id": row.get("id"),
                "content": row.get("content"),
                "special": row.get("special"),
            }
            for row in added
            if isinstance(row, dict)
        ],
    }
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _check_tokenizer(base: Path, candidate: Path) -> dict[str, Any]:
    base_config = _load_json(base / "tokenizer_config.json", max_bytes=8_000_000, allow_link=True)
    candidate_path = candidate / "tokenizer_config.json"
    candidate_config = _load_json(candidate_path, max_bytes=8_000_000)
    if not isinstance(base_config, dict) or not isinstance(candidate_config, dict):
        raise AttestationError("tokenizer_config.json must contain an object")
    if candidate_config.get("tokenizer_class") != base_config.get("tokenizer_class"):
        raise AttestationError("candidate tokenizer class differs from the pinned base")
    for key in SPECIAL_TOKEN_KEYS:
        if _token_content(candidate_config.get(key)) != _token_content(base_config.get(key)):
            raise AttestationError(f"candidate tokenizer mismatch: {key}")
    base_added = _added_token_signature(base_config)
    if _added_token_signature(candidate_config) != base_added:
        raise AttestationError("candidate added-token vocabulary differs from the pinned base")

    candidate_spm = candidate / "tokenizer.model"
    candidate_fast = candidate / "tokenizer.json"
    if not _regular(candidate_spm) and not _regular(candidate_fast):
        raise AttestationError("candidate has no local tokenizer.model or tokenizer.json")
    token_sources: list[str] = []
    if _regular(candidate_spm):
        base_spm = base / "tokenizer.model"
        if not _regular(base_spm, allow_link=True) or _sha256(candidate_spm) != _sha256(base_spm):
            raise AttestationError(
                "candidate SentencePiece vocabulary differs from the pinned base"
            )
        token_sources.append("tokenizer.model")
    if _regular(candidate_fast):
        base_fast = _load_json(base / "tokenizer.json", max_bytes=128_000_000, allow_link=True)
        final_fast = _load_json(candidate_fast, max_bytes=128_000_000)
        if not isinstance(base_fast, dict) or not isinstance(final_fast, dict):
            raise AttestationError("tokenizer.json must contain an object")
        if _fast_tokenizer_signature(final_fast) != _fast_tokenizer_signature(base_fast):
            raise AttestationError(
                "candidate fast-tokenizer vocabulary differs from the pinned base"
            )
        token_sources.append("tokenizer.json")
    return {
        "config_sha256": _sha256(candidate_path),
        "sources": token_sources,
        "added_token_count": len(base_added),
    }


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise AttestationError("JSON artifact contains a duplicate key")
        value[key] = child
    return value


def _safetensor_topology(
    path: Path, *, allow_link: bool
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], int]:
    if not _regular(path, allow_link=allow_link):
        raise AttestationError(f"required safetensors shard is missing or linked: {path.name}")
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise AttestationError(f"invalid safetensors header: {path.name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 2 or header_length > min(MAX_SAFETENSORS_HEADER_BYTES, size - 8):
                raise AttestationError(f"invalid safetensors header length: {path.name}")
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise AttestationError(f"truncated safetensors header: {path.name}")
            header = json.loads(raw_header, object_pairs_hook=_json_object_without_duplicates)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        struct.error,
    ) as exc:
        raise AttestationError(f"invalid safetensors artifact: {path.name}") from exc
    if not isinstance(header, dict):
        raise AttestationError(f"invalid safetensors metadata: {path.name}")
    topology: dict[str, tuple[str, tuple[int, ...]]] = {}
    ranges: list[tuple[int, int]] = []
    payload_bytes = size - 8 - header_length
    for name, metadata in header.items():
        if name == "__metadata__":
            if not isinstance(metadata, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata.items()
            ):
                raise AttestationError(f"invalid safetensors metadata: {path.name}")
            continue
        if (
            not isinstance(name, str)
            or not name
            or len(name.encode("utf-8")) > MAX_TENSOR_NAME_BYTES
        ):
            raise AttestationError(f"invalid safetensors tensor name: {path.name}")
        if len(topology) >= MAX_TENSOR_COUNT:
            raise AttestationError(f"safetensors tensor-count limit exceeded: {path.name}")
        if not isinstance(metadata, dict):
            raise AttestationError(f"invalid safetensors tensor metadata: {path.name}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
            raise AttestationError(f"unsupported safetensors dtype: {path.name}")
        if (
            not isinstance(shape, list)
            or len(shape) > MAX_TENSOR_RANK
            or any(
                type(dimension) is not int or not 0 <= dimension <= MAX_TENSOR_DIMENSION
                for dimension in shape
            )
        ):
            raise AttestationError(f"invalid safetensors tensor shape: {path.name}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(item) is int for item in offsets)
            or not (0 <= offsets[0] <= offsets[1] <= payload_bytes)
        ):
            raise AttestationError(f"invalid safetensors data offsets: {path.name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected_bytes = (elements * _DTYPE_BITS[dtype] + 7) // 8
        if offsets[1] - offsets[0] != expected_bytes:
            raise AttestationError(f"safetensors shape/dtype size mismatch: {path.name}")
        topology[name] = (dtype, tuple(shape))
        ranges.append((offsets[0], offsets[1]))
    if not topology:
        raise AttestationError(f"safetensors artifact has no tensors: {path.name}")
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise AttestationError(f"safetensors payload has a gap or overlap: {path.name}")
        cursor = end
    if cursor != payload_bytes:
        raise AttestationError(f"safetensors payload is not fully indexed: {path.name}")
    return topology, payload_bytes


def _safe_weight_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or Path(value).name != value
        or not value.endswith(".safetensors")
    ):
        raise AttestationError("weight index contains an unsafe or unexpected shard name")
    return value


def _logical_safetensor_topology(
    root: Path, *, allow_link: bool
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, Any]]:
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    forbidden = [
        path.name
        for path in (root / "pytorch_model.bin", root / "pytorch_model.bin.index.json")
        if path.exists()
    ]
    if forbidden:
        raise AttestationError("PyTorch pickle weights are not accepted; safetensors is required")
    present = [path for path in (index_path, single_path) if path.exists()]
    if len(present) != 1:
        raise AttestationError("model must have exactly one safetensors topology")

    if present[0] == single_path:
        topology, tensor_bytes = _safetensor_topology(single_path, allow_link=allow_link)
        local_shards = {path.name for path in root.glob("*.safetensors")}
        if local_shards != {single_path.name}:
            raise AttestationError("unindexed or extra safetensors shards are present")
        return topology, {
            "topology": single_path.name,
            "shard_count": 1,
            "bytes": single_path.stat().st_size,
            "tensor_count": len(topology),
            "tensor_payload_bytes": tensor_bytes,
        }

    index = _load_json(index_path, max_bytes=MAX_WEIGHT_INDEX_BYTES, allow_link=allow_link)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise AttestationError("weight index has no weight_map")
    if len(weight_map) > MAX_TENSOR_COUNT:
        raise AttestationError("weight index tensor-count limit exceeded")
    normalized_map: dict[str, str] = {}
    for tensor, filename in weight_map.items():
        if (
            not isinstance(tensor, str)
            or not tensor
            or len(tensor.encode("utf-8")) > MAX_TENSOR_NAME_BYTES
        ):
            raise AttestationError("weight index contains an invalid tensor name")
        normalized_map[tensor] = _safe_weight_name(filename)
    shard_paths = [root / name for name in sorted(set(normalized_map.values()))]
    local_shards = {path.name for path in root.glob("*.safetensors")}
    expected_shards = {path.name for path in shard_paths}
    if local_shards != expected_shards:
        raise AttestationError("weight index has missing or extra safetensors shards")

    logical: dict[str, tuple[str, tuple[int, ...]]] = {}
    actual_shard: dict[str, str] = {}
    tensor_bytes = 0
    for shard in shard_paths:
        shard_topology, shard_tensor_bytes = _safetensor_topology(shard, allow_link=allow_link)
        tensor_bytes += shard_tensor_bytes
        for tensor, specification in shard_topology.items():
            if tensor in logical:
                raise AttestationError("a tensor occurs in more than one safetensors shard")
            if len(logical) >= MAX_TENSOR_COUNT:
                raise AttestationError("safetensors tensor-count limit exceeded")
            logical[tensor] = specification
            actual_shard[tensor] = shard.name
    if set(normalized_map) != set(logical):
        raise AttestationError("weight index tensor set does not match its safetensors shards")
    if any(normalized_map[tensor] != actual_shard[tensor] for tensor in logical):
        raise AttestationError("weight index assigns a tensor to the wrong safetensors shard")
    metadata = index.get("metadata") if isinstance(index, dict) else None
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise AttestationError("weight index metadata must be an object")
        total_size = metadata.get("total_size")
        if total_size is not None and (type(total_size) is not int or total_size != tensor_bytes):
            raise AttestationError("weight index total_size does not match tensor payloads")
    return logical, {
        "topology": index_path.name,
        "shard_count": len(shard_paths),
        "bytes": sum(path.stat().st_size for path in shard_paths),
        "tensor_count": len(logical),
        "tensor_payload_bytes": tensor_bytes,
    }


def _check_weights(base: Path, candidate: Path, *, minimum_weight_bytes: int) -> dict[str, Any]:
    if any(candidate.glob("adapter_model.*")) or (candidate / "adapter_config.json").exists():
        raise AttestationError("adapter-only/PEFT artifacts are not a self-contained final model")
    base_topology, base_details = _logical_safetensor_topology(base, allow_link=True)
    candidate_topology, candidate_details = _logical_safetensor_topology(
        candidate, allow_link=False
    )
    if set(candidate_topology) != set(base_topology):
        raise AttestationError("candidate tensor-name topology differs from the pinned base")
    for tensor in sorted(base_topology):
        candidate_dtype, candidate_shape = candidate_topology[tensor]
        base_dtype, base_shape = base_topology[tensor]
        if candidate_shape != base_shape:
            raise AttestationError("candidate tensor-shape topology differs from the pinned base")
        if candidate_dtype != base_dtype:
            raise AttestationError("candidate tensor-dtype topology differs from the pinned base")
    if candidate_details["bytes"] < minimum_weight_bytes:
        raise AttestationError(
            "candidate weights are too small to be a plausible Gemma 3 4B artifact"
        )
    return {
        **candidate_details,
        "base_topology": base_details["topology"],
        "tensor_topology_match": True,
    }


def _atomic_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def validate(
    final_model: Path,
    *,
    expected_base_model: str,
    expected_base_revision: str,
    expected_base_checkpoint: Path,
    task_root: Path,
    minimum_weight_bytes: int = MINIMUM_WEIGHT_BYTES,
    study_input: Path | None = None,
) -> dict[str, Any]:
    if expected_base_model != EXPECTED_MODEL_ID or expected_base_revision != EXPECTED_REVISION:
        raise AttestationError("validator only accepts the pinned official Gemma 3 4B base")
    task_root = task_root.resolve(strict=True)
    base = expected_base_checkpoint.resolve(strict=True)
    if final_model.is_symlink():
        raise AttestationError("final_model itself must be a real directory, not a symlink")
    candidate = final_model.resolve(strict=True)
    if not candidate.is_dir() or not _inside(candidate, task_root):
        raise AttestationError("final_model must resolve to a local directory inside the task")
    if candidate == base:
        raise AttestationError("final_model resolves directly to the official base checkpoint")
    if base.name != EXPECTED_REVISION:
        raise AttestationError("official base checkpoint path does not name the pinned revision")

    base_config = _load_json(base / "config.json", max_bytes=4_000_000, allow_link=True)
    candidate_config_path = candidate / "config.json"
    candidate_config = _load_json(candidate_config_path, max_bytes=4_000_000)
    if not isinstance(base_config, dict) or not isinstance(candidate_config, dict):
        raise AttestationError("config.json must contain an object")
    observed = _check_invariants(base_config, candidate_config)
    candidate_tokenizer_config = _load_json(
        candidate / "tokenizer_config.json", max_bytes=8_000_000
    )
    if not isinstance(candidate_tokenizer_config, dict):
        raise AttestationError("tokenizer_config.json must contain an object")
    identity_fields = _check_declared_identities(
        [candidate_config, candidate_tokenizer_config],
        base_checkpoint=base,
        task_root=task_root,
    )
    tokenizer = _check_tokenizer(base, candidate)
    weights = _check_weights(base, candidate, minimum_weight_bytes=minimum_weight_bytes)

    study_input_sha256: str | None = None
    if study_input is not None:
        if not _regular(study_input):
            raise AttestationError("study-input record is missing, linked, or invalid")
        study_input_sha256 = _sha256(study_input)
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "scope": "structural-declarative-only",
        "causal_training_lineage_proven": False,
        "limitation": (
            "Detects accidental model substitution; it does not prove which process trained the weights."
        ),
        "expected_base": {
            "model_id": EXPECTED_MODEL_ID,
            "revision": EXPECTED_REVISION,
        },
        "final_model": {
            "path_within_task": candidate.relative_to(task_root).as_posix(),
            "config_sha256": _sha256(candidate_config_path),
            "identity_fields_checked": identity_fields,
            "tokenizer": tokenizer,
            "weights": weights,
            "invariants": observed,
        },
        "study_input_sha256": study_input_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_model", type=Path)
    parser.add_argument("--expected-base-model", required=True)
    parser.add_argument("--expected-base-revision", required=True)
    parser.add_argument("--expected-base-checkpoint", required=True, type=Path)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--study-input", type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        evidence = validate(
            args.final_model,
            expected_base_model=args.expected_base_model,
            expected_base_revision=args.expected_base_revision,
            expected_base_checkpoint=args.expected_base_checkpoint,
            task_root=args.task_root,
            study_input=args.study_input,
        )
        _atomic_private_json(args.record, evidence)
    except (AttestationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema_version": SCHEMA, "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
