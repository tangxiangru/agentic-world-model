"""Freeze selected serving bytes and publish them recoverably; never select a model.

No model weights are deserialized, no pickle is executed, and no inference is
performed. Linux no-follow descriptors/no-replace renames protect the supported
ordinary local-directory publication path. Caller-established quiescence remains
an external precondition, not something an input flag proves.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
import struct
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

IDENTITY_SCHEMA = "awm-serving-artifact-v1"
PUBLICATION_SCHEMA = "awm-serving-publication-v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DTYPES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
_PROFILES = {
    "gemma3": ("gemma3-multimodal-v1", "Gemma3ForConditionalGeneration"),
    "gemma3_text": ("native-hf-text-v1", "Gemma3ForCausalLM"),
    "gpt2": ("native-hf-text-v1", "GPT2LMHeadModel"),
    "llama": ("native-hf-text-v1", "LlamaForCausalLM"),
}
_METADATA = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
}
_OTHER_ASSETS = {
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
}
_IGNORED_FILES = {
    "training_args.bin",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state.pth",
    "train_results.json",
    "eval_results.json",
    "all_results.json",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    ".gitattributes",
}
_IGNORED_DIRS = {"logs", "runs", "tensorboard"}
_BROAD_ROOTS = {
    "/",
    "/home",
    "/root",
    "/tmp",
    "/var",
    "/var/tmp",
    "/usr",
    "/opt",
    "/workspace",
    "/workspaces",
    "/rmeng_data",
}


class ServingArtifactError(ValueError):
    def __init__(self, message, *, report=None):
        super().__init__(message)
        self.report = report or {
            "status": "invalid",
            "detail": str(message),
            "scientific_validation": "not_performed",
        }


class UnsupportedServingArtifact(ServingArtifactError):
    def __init__(self, message):
        super().__init__(
            message,
            report={
                "status": "unsupported",
                "detail": str(message),
                "scientific_validation": "not_performed",
            },
        )


class ServingPublicationError(ServingArtifactError):
    """Publication failed; ``report`` includes retained stage/backup/recovery paths."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ServingArtifactError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        _bytes(result)  # Reject nonfinite literals and exponent overflow.
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ServingArtifactError(f"malformed/nonfinite JSON: {exc}") from exc
    return result


def _absolute(path):
    path = Path(path).absolute()
    if any(part in (".", "..") for part in path.parts):
        raise UnsupportedServingArtifact("path traversal/noncanonical paths are unsupported")
    return path


@contextmanager
def _directory(path):
    """Open each ancestor with O_NOFOLLOW; never resolve an unexpected alias."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise UnsupportedServingArtifact("Linux O_NOFOLLOW directory operations are required")
    path = _absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=fd)
            except OSError as exc:
                raise UnsupportedServingArtifact(
                    f"missing, linked or non-directory path: {path}"
                ) from exc
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _birth(st):
    return {"device": st.st_dev, "inode": st.st_ino}


def _fingerprint(st):
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode,
        st.st_nlink,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


def _entry_stat(parent_fd, name):
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _basename(name):
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or PurePosixPath(name).name != name
        or "\\" in name
        or "\0" in name
    ):
        raise UnsupportedServingArtifact(
            f"only flat, traversal-free serving filenames are supported: {name!r}"
        )
    return name


@contextmanager
def _regular(parent_fd, name):
    name = _basename(name)
    before = _entry_stat(parent_fd, name)
    if before is None or not stat.S_ISREG(before.st_mode):
        raise UnsupportedServingArtifact(f"missing or non-regular serving file: {name}")
    if before.st_nlink != 1:
        raise UnsupportedServingArtifact(f"shared hardlinked serving file is unsupported: {name}")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if _fingerprint(before) != _fingerprint(opened):
            raise ServingArtifactError(f"serving file changed during open: {name}")
        yield fd, opened
        after = _entry_stat(parent_fd, name)
        if (
            after is None
            or _fingerprint(after) != _fingerprint(opened)
            or _fingerprint(os.fstat(fd)) != _fingerprint(opened)
        ):
            raise ServingArtifactError(f"serving file changed during inspection: {name}")
    finally:
        os.close(fd)


def _read_exact(fd, count):
    parts = []
    while count:
        block = os.read(fd, min(count, 1 << 20))
        if not block:
            raise ServingArtifactError("truncated file")
        parts.append(block)
        count -= len(block)
    return b"".join(parts)


def _safetensors(fd, size):
    if size < 10:
        raise ServingArtifactError("empty/truncated safetensors weight file")
    header_size = struct.unpack("<Q", _read_exact(fd, 8))[0]
    if not 1 <= header_size <= min(100_000_000, size - 8):
        raise ServingArtifactError("invalid or unsupported safetensors header length")
    header = _json(_read_exact(fd, header_size))
    if not isinstance(header, dict):
        raise ServingArtifactError("safetensors header must be a mapping")
    payload_size = size - 8 - header_size
    ranges, names, data_bytes = [], [], 0
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                raise ServingArtifactError("invalid safetensors metadata")
            continue
        if (
            not name
            or not isinstance(tensor, dict)
            or set(tensor) != {"dtype", "shape", "data_offsets"}
        ):
            raise ServingArtifactError("invalid safetensors tensor declaration")
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise UnsupportedServingArtifact(f"unsupported safetensors dtype: {dtype}")
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
        ):
            raise ServingArtifactError("invalid safetensors shape/offsets")
        start, end = offsets
        expected_bytes = math.prod(shape) * _DTYPES[dtype]
        if start > end or end > payload_size or end - start != expected_bytes:
            raise ServingArtifactError("safetensors shape/offset/payload mismatch")
        if end > start:
            ranges.append((start, end))
        data_bytes += expected_bytes
        names.append(name)
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ServingArtifactError("safetensors payload has overlapping or missing ranges")
        cursor = end
    if not names or data_bytes == 0 or cursor != payload_size:
        raise ServingArtifactError("empty weight shard or unaccounted safetensors payload")
    return {
        "tensor_names": names,
        "tensor_bytes": data_bytes,
        "validation": "header_shapes_offsets_only_values_and_model_alignment_unverified",
    }


def _inspect_file(root_fd, name):
    with _regular(root_fd, name) as (fd, st):
        if st.st_size == 0:
            raise ServingArtifactError(f"empty serving file: {name}")
        structure = None
        parsed = None
        if name.endswith(".safetensors"):
            structure = _safetensors(fd, st.st_size)
        elif name.endswith(".json"):
            if st.st_size > 100_000_000:
                raise UnsupportedServingArtifact(
                    f"JSON metadata exceeds the supported bound: {name}"
                )
            parsed = _json(_read_exact(fd, st.st_size))
            if not isinstance(parsed, dict):
                raise ServingArtifactError(f"metadata must be a JSON object: {name}")
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(fd, 1 << 20), b""):
            digest.update(block)
        return {"path": name, "bytes": st.st_size, "sha256": digest.hexdigest()}, parsed, structure


def _inventory(root_fd):
    serving, ignored = [], []
    for entry in os.scandir(root_fd):
        name = _basename(entry.name)
        st = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(st.st_mode) or not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
            raise UnsupportedServingArtifact(f"unexpected link/special file: {name}")
        if stat.S_ISDIR(st.st_mode):
            if name in _IGNORED_DIRS or re.fullmatch(r"checkpoint-\d+", name):
                ignored.append(name + "/")
                continue
            raise UnsupportedServingArtifact(f"unknown nested serving layout: {name}")
        if (
            name in _IGNORED_FILES
            or re.fullmatch(r"rng_state_\d+\.pth", name)
            or name.startswith("events.out.tfevents.")
            or name.endswith(".log")
        ):
            ignored.append(name)
        elif (
            name in _METADATA | _OTHER_ASSETS
            or name.endswith(".safetensors")
            or name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
            or re.fullmatch(r"pytorch_model(?:-\d+-of-\d+)?\.bin", name)
        ):
            serving.append(name)
        else:
            raise UnsupportedServingArtifact(
                f"unknown potential inference asset cannot be silently omitted: {name}"
            )
    return sorted(serving), sorted(ignored)


def _profile(metadata, names):
    config = metadata["config.json"]
    kind = config.get("model_type")
    if not isinstance(kind, str) or kind not in _PROFILES:
        raise UnsupportedServingArtifact(f"unsupported native model profile: {kind}")
    profile, architecture = _PROFILES[kind]
    if config.get("architectures") != [architecture]:
        raise UnsupportedServingArtifact("unknown or mixed native model architecture declaration")
    if config.get("quantization_config") is not None:
        raise UnsupportedServingArtifact(
            "quantized/custom serving profiles require an explicit adapter"
        )
    for filename, data in metadata.items():
        if filename not in {
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "processor_config.json",
            "preprocessor_config.json",
        }:
            # A token called "tokenizer_file" in a vocabulary is not an asset reference.
            continue
        if (
            data.get("auto_map")
            or data.get("configuration_files")
            or data.get("custom_generate")
            or data.get("trust_remote_code")
        ):
            raise UnsupportedServingArtifact(
                f"remote/custom/config-redirection metadata is unsupported: {filename}"
            )
        for key, value in data.items():
            if (
                key.endswith("_file")
                and value is not None
                and (not isinstance(value, str) or _basename(value) not in names)
            ):
                raise UnsupportedServingArtifact(
                    f"unbound metadata asset reference: {filename}:{key}"
                )
    tokenizer = metadata["tokenizer.json"]
    if not isinstance(tokenizer.get("model"), dict) or not tokenizer["model"].get("type"):
        raise ServingArtifactError("tokenizer.json has no native tokenizer model structure")
    tokenizer_class = metadata["tokenizer_config.json"].get("tokenizer_class")
    if tokenizer_class not in (
        "PreTrainedTokenizerFast",
        "GPT2Tokenizer",
        "GPT2TokenizerFast",
        "GemmaTokenizer",
        "GemmaTokenizerFast",
        "LlamaTokenizer",
        "LlamaTokenizerFast",
    ):
        raise UnsupportedServingArtifact(f"unknown tokenizer class: {tokenizer_class}")
    if kind == "gemma3":
        if (
            not isinstance(config.get("text_config"), dict)
            or config["text_config"].get("model_type") != "gemma3_text"
            or not isinstance(config.get("vision_config"), dict)
            or config["vision_config"].get("model_type") != "siglip_vision_model"
        ):
            raise ServingArtifactError("Gemma3 requires explicit native text and vision metadata")
        if not {"preprocessor_config.json", "processor_config.json"}.issubset(names):
            raise ServingArtifactError(
                "Gemma3 requires both preprocessor_config.json and processor_config.json"
            )
        if metadata["processor_config.json"].get(
            "processor_class"
        ) != "Gemma3Processor" or metadata["preprocessor_config.json"].get(
            "image_processor_type"
        ) not in ("Gemma3ImageProcessor", "Gemma3ImageProcessorFast"):
            raise UnsupportedServingArtifact("unsupported Gemma3 processor metadata")
    elif "preprocessor_config.json" in names or "processor_config.json" in names:
        raise UnsupportedServingArtifact(
            "extra processor assets need a matching multimodal profile"
        )
    return profile, kind, architecture


def _weight_layout(names, metadata, structures, allow_opaque_weights):
    safe = {n for n in names if n.endswith(".safetensors")}
    bins = {n for n in names if n.endswith(".bin")}
    indexes = {n for n in names if n.endswith(".index.json")}
    if (safe and bins) or len(indexes) > 1 or not (safe or bins):
        raise ServingArtifactError("missing or ambiguous mixed weight layout")
    if bins and not allow_opaque_weights:
        raise UnsupportedServingArtifact(
            "pickle weights are opaque; explicit allow_opaque_weights is required"
        )
    weights = safe or bins
    kind = "safetensors" if safe else "pytorch_bin_opaque"
    if not indexes:
        expected = {"model.safetensors"} if safe else {"pytorch_model.bin"}
        if weights != expected:
            raise ServingArtifactError("unindexed/ambiguous shard layout")
        return {
            "format": kind,
            "layout": "single",
            "index": None,
            "structure": "safetensors_header_checked"
            if safe
            else "opaque_bytes_only_no_pickle_load",
        }
    index_name = next(iter(indexes))
    expected_index = "model.safetensors.index.json" if safe else "pytorch_model.bin.index.json"
    if index_name != expected_index or (
        "model.safetensors" in weights or "pytorch_model.bin" in weights
    ):
        raise ServingArtifactError("mixed indexed and single-file weights")
    index = metadata[index_name]
    mapping = index.get("weight_map")
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(not isinstance(key, str) or not key for key in mapping)
    ):
        raise ServingArtifactError("weight index needs a nonempty tensor-to-shard map")
    referenced = {_basename(value) for value in mapping.values()}
    if referenced != weights:
        raise ServingArtifactError("missing or unindexed weight shards")
    declared_metadata = index.get("metadata", {})
    if not isinstance(declared_metadata, dict):
        raise ServingArtifactError("index metadata must be an object")
    total = declared_metadata.get("total_size")
    if total is not None and (type(total) is not int or total <= 0):
        raise ServingArtifactError("invalid index total_size")
    if safe:
        actual = {}
        tensor_bytes = 0
        for shard in sorted(weights):
            structure = structures[shard]
            tensor_bytes += structure["tensor_bytes"]
            for tensor_name in structure["tensor_names"]:
                if tensor_name in actual:
                    raise ServingArtifactError("duplicate tensor across weight shards")
                actual[tensor_name] = shard
        if actual != mapping or (total is not None and total != tensor_bytes):
            raise ServingArtifactError(
                "index tensor map/total_size differs from actual safetensors headers"
            )
    return {
        "format": kind,
        "layout": "indexed",
        "index": index_name,
        "structure": "safetensors_header_and_index_checked"
        if safe
        else "opaque_bytes_index_names_only",
    }


def snapshot_serving_artifact(path, *, allow_opaque_weights=False):
    """Capture current supported serving bytes; this does not bind past evaluations."""
    if type(allow_opaque_weights) is not bool:
        raise ServingArtifactError("allow_opaque_weights must be an explicit boolean")
    path = _absolute(path)
    with _directory(path) as root_fd:
        names, ignored = _inventory(root_fd)
        required = {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        if not required.issubset(names):
            raise ServingArtifactError(
                "missing required serving metadata: " + ", ".join(sorted(required - set(names)))
            )
        entries, metadata, structures = [], {}, {}
        for name in names:
            entry, parsed, structure = _inspect_file(root_fd, name)
            entries.append(entry)
            if parsed is not None:
                metadata[name] = parsed
            if structure is not None:
                structures[name] = structure
        profile, model_type, architecture = _profile(metadata, set(names))
        weights = _weight_layout(names, metadata, structures, allow_opaque_weights)
        if _inventory(root_fd)[0] != names:
            raise ServingArtifactError("serving layout changed during snapshot")
        # A second content pass prevents a later file mutation from hiding behind
        # sequential inspection of an earlier file in this same snapshot.
        if [_inspect_file(root_fd, n)[0] for n in names] != entries:
            raise ServingArtifactError("serving bytes changed during snapshot")
    content = {
        "profile": profile,
        "model_type": model_type,
        "architecture": architecture,
        "weights": weights,
        "files": entries,
    }
    return {
        "schema_version": IDENTITY_SCHEMA,
        "captured_at": _now(),
        "source_path": str(path),
        "identity_sha256": _sha(_bytes(content)),
        "content": content,
        "inspection": {
            "ignored_nonserving_entries": ignored,
            "native_metadata_load": "not_performed",
            "model_load": "not_performed",
            "inference_engine_validation": "not_performed",
            "scientific_validation": "not_performed",
        },
        "proof": "current_serving_bytes_and_explicit_structure_not_prior_measurement_identity",
    }


def _expected(identity):
    # Take a value snapshot: the caller must not be able to change the expected
    # bytes through a shared mutable dictionary while verification/copy is live.
    identity = _json(_bytes(identity))
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version") != IDENTITY_SCHEMA
        or not isinstance(identity.get("content"), dict)
        or identity.get("identity_sha256") != _sha(_bytes(identity["content"]))
    ):
        raise ServingArtifactError("missing or malformed already-frozen serving identity")
    files = identity["content"].get("files")
    if not isinstance(files, list) or not files:
        raise ServingArtifactError("frozen identity has no serving files")
    names = []
    for entry in files:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "bytes", "sha256"}
            or type(entry["bytes"]) is not int
            or entry["bytes"] <= 0
            or not isinstance(entry["sha256"], str)
            or not _HASH.fullmatch(entry["sha256"])
        ):
            raise ServingArtifactError("malformed frozen file identity")
        names.append(_basename(entry["path"]))
    if names != sorted(set(names)):
        raise ServingArtifactError("frozen file identities must be unique and sorted")
    return identity


def _generation_hash(identity):
    for entry in _expected(identity)["content"]["files"]:
        if entry["path"] == "generation_config.json":
            return entry["sha256"]
    raise ServingArtifactError("frozen identity lacks generation_config.json")


def _load_metadata(path, profile):
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise UnsupportedServingArtifact(
            "native metadata loading requires pinned Transformers 4.57.3"
        ) from exc
    if version != "4.57.3":
        raise UnsupportedServingArtifact(f"unsupported native metadata loader version: {version}")
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig

    options = {"local_files_only": True, "trust_remote_code": False, "token": False}
    try:
        config = AutoConfig.from_pretrained(path, **options)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, **options)
        generation = GenerationConfig.from_pretrained(path, local_files_only=True, token=False)
        classes = {
            "config": type(config).__name__,
            "tokenizer": type(tokenizer).__name__,
            "generation": type(generation).__name__,
        }
        if profile == "gemma3-multimodal-v1":
            processor = AutoProcessor.from_pretrained(path, **options)
            classes["processor"] = type(processor).__name__
    except Exception as exc:
        raise ServingArtifactError(
            f"local native metadata loading failed (no weights loaded): {exc}"
        ) from exc
    return {"status": "loaded_local_metadata_only", "transformers": version, "classes": classes}


def verify_serving_artifact(
    path,
    expected_identity,
    expected_generation_sha256,
    *,
    load_metadata=True,
    allow_opaque_weights=False,
):
    """Match already-frozen bytes/settings; never normalize or infer the selected decoder."""
    if type(load_metadata) is not bool:
        raise ServingArtifactError("load_metadata must be an explicit boolean")
    expected_identity = _expected(expected_identity)
    if not isinstance(expected_generation_sha256, str) or not _HASH.fullmatch(
        expected_generation_sha256
    ):
        raise ServingArtifactError("an explicit frozen selected-generation SHA256 is required")
    if _generation_hash(expected_identity) != expected_generation_sha256:
        raise ServingArtifactError(
            "checkpoint identity is not the explicitly selected generation configuration"
        )
    observed = snapshot_serving_artifact(path, allow_opaque_weights=allow_opaque_weights)
    if observed["content"] != expected_identity["content"]:
        raise ServingArtifactError("serving files differ from the already-frozen selected identity")
    metadata = {"status": "not_performed_explicit_structural_only_mode"}
    if load_metadata:
        metadata = _load_metadata(_absolute(path), observed["content"]["profile"])
        after = snapshot_serving_artifact(path, allow_opaque_weights=allow_opaque_weights)
        if after["content"] != observed["content"]:
            raise ServingArtifactError("serving bytes changed during local metadata loading")
    return {
        "status": "selected_bytes_verified",
        "path": str(_absolute(path)),
        "identity_sha256": observed["identity_sha256"],
        "generation_sha256": expected_generation_sha256,
        "profile": observed["content"]["profile"],
        "weight_structure": observed["content"]["weights"]["structure"],
        "native_metadata_load": metadata,
        "model_load": "not_performed",
        "inference_engine_validation": "not_performed",
        "scientific_validation": "not_performed",
        "measurement_binding": "caller_must_supply_pre_measurement_identity_not_verified_here",
    }


def _rename_noreplace(parent_fd, source_name, destination_name):
    """No overwrite even if another actor populates an initially absent target."""
    source_name, destination_name = _basename(source_name), _basename(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise UnsupportedServingArtifact(
            "Linux renameat2(RENAME_NOREPLACE) is required; no unsafe fallback"
        )
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(destination_name), 1):
        code = ctypes.get_errno()
        if code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            raise UnsupportedServingArtifact("filesystem does not support safe no-replace rename")
        raise OSError(code, os.strerror(code), destination_name)
    os.fsync(parent_fd)


def _check_anchor(path, expected_birth):
    with _directory(path) as fd:
        if _birth(os.fstat(fd)) != expected_birth:
            raise ServingArtifactError("publication parent directory identity changed")


def _mkdir_open(parent_fd, name):
    name = _basename(name)
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        raise UnsupportedServingArtifact(
            f"publication journal parent is linked/non-directory: {name}"
        ) from exc


def _journal(journal_fd, name, record):
    """Replace only this operation's own journal; never touch an incumbent file."""
    temporary = ".journal-" + uuid4().hex + ".tmp"
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=journal_fd
    )
    try:
        raw = _bytes(record)
        position = 0
        while position < len(raw):
            position += os.write(fd, raw[position:])
        os.fsync(fd)
    finally:
        os.close(fd)
    # name is an unpredictable operation-owned UUID, not a caller target.
    os.replace(temporary, name, src_dir_fd=journal_fd, dst_dir_fd=journal_fd)
    os.fsync(journal_fd)


def _copy_regular(source_fd, stage_fd, entry):
    name = entry["path"]
    with _regular(source_fd, name) as (read_fd, st):
        if st.st_size != entry["bytes"]:
            raise ServingArtifactError(f"source file size changed before copy: {name}")
        write_fd = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=stage_fd
        )
        digest = hashlib.sha256()
        try:
            for block in iter(lambda: os.read(read_fd, 1 << 20), b""):
                digest.update(block)
                position = 0
                while position < len(block):
                    position += os.write(write_fd, block[position:])
            os.fsync(write_fd)
        finally:
            os.close(write_fd)
        if digest.hexdigest() != entry["sha256"]:
            raise ServingArtifactError(f"source file changed while copying: {name}")


def _publication_scope(source, destination, session_dir):
    from awm.paths import REPO_ROOT

    source, destination, session_dir = map(_absolute, (source, destination, session_dir))
    if str(session_dir) in _BROAD_ROOTS or session_dir == Path.home():
        raise ServingArtifactError(
            "publication requires a specific task/session scope, not a broad root"
        )
    forbidden = {session_dir, Path.home(), Path.cwd(), Path(REPO_ROOT)}
    if (
        destination in forbidden
        or session_dir not in destination.parents
        or str(destination) in _BROAD_ROOTS
        or destination.name in ("memory", ".git", "data")
    ):
        raise ServingArtifactError(
            "destination must be a narrow serving directory inside the explicit session scope"
        )
    if source == destination or source in destination.parents or destination in source.parents:
        raise ServingArtifactError("source and destination directories must not overlap")
    with _directory(session_dir), _directory(destination.parent), _directory(source):
        pass
    return source, destination, session_dir


def _rollback(
    parent_fd,
    parent,
    destination_name,
    stage_birth,
    backup_name,
    old_birth,
    expected_old_identity,
    allow_opaque_weights,
    record,
):
    """Only move the precise directory objects owned/identified by this operation."""
    destination = _entry_stat(parent_fd, destination_name)
    if destination is not None and stage_birth is not None and _birth(destination) == stage_birth:
        failed_name = Path(record["failed_publication_path"]).name
        _rename_noreplace(parent_fd, destination_name, failed_name)
        destination = None
    backup = _entry_stat(parent_fd, backup_name) if backup_name is not None else None
    if backup is not None:
        if _birth(backup) != old_birth or not stat.S_ISDIR(backup.st_mode):
            raise ServingArtifactError(
                "rollback refused: backup is not the identified old incumbent directory"
            )
        if destination is not None:
            raise ServingArtifactError(
                "rollback refused: destination is occupied by an unowned directory"
            )
        verify_serving_artifact(
            parent / backup_name,
            expected_old_identity,
            _generation_hash(expected_old_identity),
            load_metadata=False,
            allow_opaque_weights=allow_opaque_weights,
        )
        _rename_noreplace(parent_fd, backup_name, destination_name)
        verify_serving_artifact(
            parent / destination_name,
            expected_old_identity,
            _generation_hash(expected_old_identity),
            load_metadata=False,
            allow_opaque_weights=allow_opaque_weights,
        )
        record["rollback"] = "old_incumbent_restored"
    elif old_birth is not None:
        if destination is not None and _birth(destination) == old_birth:
            record["rollback"] = "old_incumbent_untouched"
        else:
            raise ServingArtifactError("rollback could not locate the identified old incumbent")
    else:
        record["rollback"] = "no_identified_old_incumbent; any_unowned_target_left_untouched"


def publish_serving_artifact(
    source,
    destination,
    expected_identity,
    expected_generation_sha256,
    *,
    session_dir,
    task_id,
    reference_id,
    replace=False,
    expected_old_identity=None,
    target_quiescent=False,
    quiescence_evidence=None,
    load_metadata=True,
    allow_opaque_weights=False,
):
    """Stage, verify, then publish; explicit replacements retain a unique backup.

    Quiescence arguments record a caller-established precondition, not a liveness
    proof. SIGKILL can leave an intermediate two-rename state: the journal records
    intended recovery paths before target mutation. No stage/backup/incumbent is
    recursively deleted and no populated target is merged into. On failure inspect
    the exception report and retained paths before retrying.
    """
    import fcntl

    for name, value in (
        ("replace", replace),
        ("target_quiescent", target_quiescent),
        ("load_metadata", load_metadata),
        ("allow_opaque_weights", allow_opaque_weights),
    ):
        if type(value) is not bool:
            raise ServingArtifactError(f"{name} must be an explicit boolean")
    if any(not isinstance(value, str) or not value.strip() for value in (task_id, reference_id)):
        raise ServingArtifactError("explicit task_id and planned reference_id are required")
    if replace and (
        expected_old_identity is None
        or target_quiescent is not True
        or not isinstance(quiescence_evidence, str)
        or not quiescence_evidence.strip()
    ):
        raise ServingArtifactError(
            "replacement requires frozen old identity and caller-established quiescence evidence"
        )
    if not replace and expected_old_identity is not None:
        raise ServingArtifactError(
            "old identity does not imply replacement authority; set replace explicitly"
        )
    source, destination, scope = _publication_scope(source, destination, session_dir)
    expected_identity = _expected(expected_identity)
    if replace:
        expected_old_identity = _expected(expected_old_identity)
    operation = uuid4().hex
    parent = destination.parent
    stage_name = f".{destination.name}.stage-{operation}"
    backup_name = f".{destination.name}.backup-{operation}" if replace else None
    failed_name = f".{destination.name}.failed-{operation}"
    journal_name = operation + ".json"
    journal_path = scope / "memory/serving-publications" / journal_name
    record = {
        "schema_version": PUBLICATION_SCHEMA,
        "operation_id": operation,
        "task_id": task_id,
        "reference_id": reference_id,
        "session_dir": str(scope),
        "source": str(source),
        "destination": str(destination),
        "expected_identity": expected_identity,
        "expected_generation_sha256": expected_generation_sha256,
        "expected_old_identity": expected_old_identity,
        "stage": str(parent / stage_name),
        "backup": str(parent / backup_name) if backup_name else None,
        "failed_publication_path": str(parent / failed_name),
        "record_path": str(journal_path),
        "status": "preparing",
        "phase": "preconditions",
        "started_at": _now(),
        "replacement_requested": replace,
        "target_quiescence": {
            "caller_established": target_quiescent,
            "evidence": quiescence_evidence,
            "independent_verification": "not_performed",
        },
        "scientific_validation": "not_performed",
        "inference_engine_validation": "not_performed",
        "atomicity": (
            "recoverable_two_rename_sequence_not_atomic_exchange"
            if replace
            else "single_no_replace_directory_rename"
        ),
        "rollback": "not_needed",
        "copied_files": [],
    }
    started = time.monotonic()
    stage_birth, old_birth = None, None
    with _directory(scope) as scope_fd, _directory(parent) as parent_fd:
        parent_birth = _birth(os.fstat(parent_fd))
        memory_fd = _mkdir_open(scope_fd, "memory")
        try:
            journal_fd = _mkdir_open(memory_fd, "serving-publications")
        finally:
            os.close(memory_fd)
        lock_fd = None
        try:
            lock_name = ".target-" + _sha(str(destination).encode()) + ".lock"
            lock_fd = os.open(
                lock_name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=journal_fd
            )
            lock_stat = os.fstat(lock_fd)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                raise UnsupportedServingArtifact("publication lock is not an ordinary private file")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ServingArtifactError(
                    "another guarded publication owns this exact destination"
                ) from exc
            _journal(journal_fd, journal_name, record)
            existing = _entry_stat(parent_fd, destination.name)
            if existing is not None:
                if not replace:
                    raise ServingArtifactError(
                        "destination already exists; default publication never replaces or merges"
                    )
                if not stat.S_ISDIR(existing.st_mode):
                    raise UnsupportedServingArtifact(
                        "replacement target must be a regular directory, not an alias"
                    )
                old_birth = _birth(existing)
                record["old_directory_identity"] = old_birth
                verify_serving_artifact(
                    destination,
                    expected_old_identity,
                    _generation_hash(expected_old_identity),
                    load_metadata=False,
                    allow_opaque_weights=allow_opaque_weights,
                )
            elif replace:
                raise ServingArtifactError(
                    "explicit replacement requires the expected existing incumbent"
                )
            record["source_verification"] = verify_serving_artifact(
                source,
                expected_identity,
                expected_generation_sha256,
                load_metadata=load_metadata,
                allow_opaque_weights=allow_opaque_weights,
            )
            _check_anchor(parent, parent_birth)
            os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
            stage_fd = os.open(
                stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            try:
                stage_birth = _birth(os.fstat(stage_fd))
                record["stage_directory_identity"] = stage_birth
                record["phase"] = "copying_stage"
                _journal(journal_fd, journal_name, record)
                with _directory(source) as source_fd:
                    for entry in expected_identity["content"]["files"]:
                        _copy_regular(source_fd, stage_fd, entry)
                        record["copied_files"].append(entry["path"])
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            _check_anchor(parent, parent_birth)
            record["stage_verification"] = verify_serving_artifact(
                parent / stage_name,
                expected_identity,
                expected_generation_sha256,
                load_metadata=load_metadata,
                allow_opaque_weights=allow_opaque_weights,
            )
            record["source_recheck"] = verify_serving_artifact(
                source,
                expected_identity,
                expected_generation_sha256,
                load_metadata=False,
                allow_opaque_weights=allow_opaque_weights,
            )
            if replace:
                verify_serving_artifact(
                    destination,
                    expected_old_identity,
                    _generation_hash(expected_old_identity),
                    load_metadata=False,
                    allow_opaque_weights=allow_opaque_weights,
                )
                current_old = _entry_stat(parent_fd, destination.name)
                if current_old is None or _birth(current_old) != old_birth:
                    raise ServingArtifactError(
                        "incumbent directory identity changed before replacement"
                    )
                record["phase"] = "backup_rename_pending"
                _journal(journal_fd, journal_name, record)
                _check_anchor(parent, parent_birth)
                _rename_noreplace(parent_fd, destination.name, backup_name)
                record["phase"] = "old_incumbent_backed_up"
                _journal(journal_fd, journal_name, record)
            record["phase"] = "publication_rename_pending"
            _journal(journal_fd, journal_name, record)
            _check_anchor(parent, parent_birth)
            _rename_noreplace(parent_fd, stage_name, destination.name)
            record["phase"] = "published_verification_pending"
            _journal(journal_fd, journal_name, record)
            record["published_verification"] = verify_serving_artifact(
                destination,
                expected_identity,
                expected_generation_sha256,
                load_metadata=False,
                allow_opaque_weights=allow_opaque_weights,
            )
            published = _entry_stat(parent_fd, destination.name)
            if published is None or _birth(published) != stage_birth:
                raise ServingArtifactError("published directory object changed")
            record.update(
                status="published",
                phase="complete",
                finished_at=_now(),
                elapsed_seconds=time.monotonic() - started,
            )
            _journal(journal_fd, journal_name, record)
            return record
        except BaseException as exc:
            record.update(
                status="failed",
                failure_phase=record["phase"],
                primary_error=f"{type(exc).__name__}: {exc}",
            )
            try:
                _check_anchor(parent, parent_birth)
                _rollback(
                    parent_fd,
                    parent,
                    destination.name,
                    stage_birth,
                    backup_name,
                    old_birth,
                    expected_old_identity,
                    allow_opaque_weights,
                    record,
                )
            except BaseException as rollback_error:  # noqa: BLE001 - keep original failure
                record["rollback"] = "failed_or_blocked; manual_recovery_required"
                record["rollback_error"] = f"{type(rollback_error).__name__}: {rollback_error}"
            record.update(finished_at=_now(), elapsed_seconds=time.monotonic() - started)
            try:
                _journal(journal_fd, journal_name, record)
            except BaseException as journal_error:  # noqa: BLE001 - keep original failure
                record["journal_error"] = f"{type(journal_error).__name__}: {journal_error}"
            if not isinstance(exc, Exception):
                exc.publication_record = record
                raise
            raise ServingPublicationError(str(exc), report=record) from exc
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(journal_fd)
