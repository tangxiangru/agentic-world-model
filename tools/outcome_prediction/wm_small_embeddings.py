"""Frozen general-English MiniLM text embeddings using CPU ONNX, never labels.

Inputs are explicit caller-approved strings, not arbitrary archive records. This
module does not sanitize or certify them. It is not a code-specialized encoder.
Only preparation may download four pinned public model resources; ordinary
encode_texts calls are local-only. No Torch, remote inference, or label fitting.

Prepare: python -m tools.outcome_prediction.wm_small_embeddings prepare
Verify cached setup: append --local-only. Dependencies: onnxruntime, tokenizers,
huggingface-hub, numpy (installed separately; no project lockfile changes).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSION = 384
MAX_SEQUENCE_TOKENS = 256
CONTENT_TOKENS_PER_CHUNK = 254  # [CLS] and [SEP] occupy two positions.
MAX_CHUNKS = 8
BATCH_SIZE = 32
CPU_THREADS = 1
RESOURCE_SHA256 = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_resources(cache_dir=None, *, local_files_only=True):
    """Resolve/cache only pinned resources, checking exact bytes before loading.

    token=False intentionally avoids reading/sending credentials for this public
    model. local_files_only=True never queries the Hub, even for revisions.
    """
    from huggingface_hub import hf_hub_download

    paths = {}
    for filename, expected in RESOURCE_SHA256.items():
        path = Path(
            hf_hub_download(
                MODEL_ID,
                filename,
                revision=MODEL_REVISION,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=False,
            )
        )
        if _file_sha256(path) != expected:
            raise ValueError(f"Pinned encoder resource hash mismatch: {filename}")
        paths[filename] = path
    return paths


def _load_backend(paths):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    for token, expected in (("[CLS]", 101), ("[SEP]", 102), ("[PAD]", 0)):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError("Pinned tokenizer special-token mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = CPU_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(paths["onnx/model.onnx"]),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("Encoder must use CPU only")
    if {x.name: x.type for x in session.get_inputs()} != {
        "input_ids": "tensor(int64)",
        "attention_mask": "tensor(int64)",
        "token_type_ids": "tensor(int64)",
    }:
        raise ValueError("Pinned encoder input signature mismatch")
    outputs = session.get_outputs()
    if (
        len(outputs) != 1
        or outputs[0].name != "last_hidden_state"
        or outputs[0].type != "tensor(float)"
        or outputs[0].shape[-1] != DIMENSION
    ):
        raise ValueError("Pinned encoder output signature mismatch")
    return tokenizer, session


def _texts(texts):
    if isinstance(texts, (str, bytes)):
        raise TypeError("Pass a sequence of explicit text strings, not one string")
    values = list(texts)
    if any(not isinstance(text, str) for text in values):
        raise TypeError("Only explicit strings may be embedded")
    return values


def _chunks(token_ids):
    retained = token_ids[: CONTENT_TOKENS_PER_CHUNK * MAX_CHUNKS]
    chunks = [
        [101, *retained[start : start + CONTENT_TOKENS_PER_CHUNK], 102]
        for start in range(0, len(retained), CONTENT_TOKENS_PER_CHUNK)
    ]
    return chunks or [[101, 102]]


def _normalize(matrix):
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite encoder output")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Degenerate encoder output")
    return (matrix / norms).astype(np.float32, copy=False)


class FrozenTextEncoder:
    """One reusable cached model session; configuration is fixed, not searched."""

    def __init__(self, cache_dir=None, *, local_files_only=True):
        started = time.perf_counter()
        paths = prepare_resources(cache_dir, local_files_only=local_files_only)
        self._tokenizer, self._session = _load_backend(paths)
        self._metadata = {
            "schema": "frozen-small-text-encoder-v1",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_directory": str(paths["config.json"].parent),
            "resource_sha256": dict(RESOURCE_SHA256),
            "resource_bytes": {name: path.stat().st_size for name, path in paths.items()},
            "model_bytes": paths["onnx/model.onnx"].stat().st_size,
            "total_resource_bytes": sum(path.stat().st_size for path in paths.values()),
            "dimension": DIMENSION,
            "encoder_kind": "general-English sentence/text encoder; not code-specialized",
            "frozen_no_label_training": True,
            "caller_owns_text_sanitization": True,
            "provider": "CPUExecutionProvider",
            "cpu_intra_op_threads": CPU_THREADS,
            "cpu_inter_op_threads": CPU_THREADS,
            "execution": "ORT_SEQUENTIAL; ORT_ENABLE_ALL",
            "portability": "Fixed runtime is deterministic; bitwise equality across platforms is not promised",
            "batch_size": BATCH_SIZE,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "content_tokens_per_chunk": CONTENT_TOKENS_PER_CHUNK,
            "max_chunks": MAX_CHUNKS,
            "truncation": "first 2032 content WordPieces; nonoverlapping chunks; no text rewriting",
            "chunk_pooling": "attention-mask mean including CLS/SEP, then L2 normalize",
            "text_pooling": "equal-weight mean of normalized chunks, then L2 normalize",
            "empty_text": "one CLS/SEP-only chunk",
            "empty_batch": "float32 array with shape (0,384); no model dispatch",
            "preparation_local_files_only": local_files_only,
            "encoding_network_access": False,
            "dependencies": {
                package: metadata.version(package)
                for package in ("numpy", "onnxruntime", "tokenizers", "huggingface-hub")
            },
            "load_seconds": time.perf_counter() - started,
            "last_encoding": None,
        }

    def encode_texts(self, texts):
        values = _texts(texts)
        started = time.perf_counter()
        chunks, owners, original_counts, retained_counts = [], [], [], []
        for index, text in enumerate(values):
            ids = self._tokenizer.encode(text, add_special_tokens=False).ids
            pieces = _chunks(ids)
            chunks.extend(pieces)
            owners.extend([index] * len(pieces))
            original_counts.append(len(ids))
            retained_counts.append(min(len(ids), CONTENT_TOKENS_PER_CHUNK * MAX_CHUNKS))
        vectors = np.zeros((len(values), DIMENSION), dtype=np.float32)
        chunk_counts = np.bincount(owners, minlength=len(values))
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            width = max(map(len, batch))
            assert width <= MAX_SEQUENCE_TOKENS
            ids = np.zeros((len(batch), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, piece in enumerate(batch):
                ids[row, : len(piece)] = piece
                mask[row, : len(piece)] = 1
            states = self._session.run(
                ["last_hidden_state"],
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
            )[0]
            if states.shape != (len(batch), width, DIMENSION):
                raise ValueError("Unexpected token embedding shape")
            expanded = mask[..., None].astype(np.float32)
            pooled = (states * expanded).sum(axis=1) / expanded.sum(axis=1)
            pooled = _normalize(pooled)
            for offset, vector in enumerate(pooled):
                vectors[owners[start + offset]] += vector
        if len(values):
            vectors = _normalize(vectors / chunk_counts[:, None])
        elapsed = time.perf_counter() - started
        self._metadata["last_encoding"] = {
            "texts": len(values),
            "chunks": len(chunks),
            "original_content_tokens": sum(original_counts),
            "retained_content_tokens": sum(retained_counts),
            "truncated_texts": sum(
                a > b for a, b in zip(original_counts, retained_counts, strict=True)
            ),
            "per_text_chunk_counts": chunk_counts.tolist(),
            "elapsed_seconds": elapsed,
            "seconds_per_text": elapsed / len(values) if values else None,
        }
        return vectors

    def metadata(self):
        """Content-free runtime/resource/latency metadata; caller may persist it."""
        return copy.deepcopy(self._metadata)


@lru_cache(maxsize=1)
def get_encoder():
    """Default reusable session; missing cached resources fail without network."""
    return FrozenTextEncoder(local_files_only=True)


def encode_texts(texts):
    return get_encoder().encode_texts(texts)


def encoder_metadata():
    return get_encoder().metadata()


def _value_end(text, start):
    """Skip an unselected JSON value without decoding outcome-bearing columns."""
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            if depth == 0:
                return index
            depth -= 1
        elif char == "," and depth == 0:
            return index
    return len(text)


def _embedding_text_column(raw):
    """Decode only object keys and embedding_text, preserving frozen row order."""
    text = raw.decode("utf-8")
    decoder = json.JSONDecoder()
    position, values = 0, []

    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    def require(index, char):
        index = whitespace(index)
        if index >= len(text) or text[index] != char:
            raise ValueError("Malformed embedding input structure")
        return index + 1

    position = require(position, "[")
    position = whitespace(position)
    if position < len(text) and text[position] == "]":
        position += 1
    else:
        while True:
            position = require(position, "{")
            keys, value = set(), None
            while True:
                position = whitespace(position)
                if position < len(text) and text[position] == "}":
                    position += 1
                    break
                key, position = decoder.raw_decode(text, position)
                if not isinstance(key, str) or key in keys:
                    raise ValueError("Invalid or duplicate embedding input key")
                keys.add(key)
                position = whitespace(require(position, ":"))
                if key == "embedding_text":
                    value, position = decoder.raw_decode(text, position)
                    if not isinstance(value, str):
                        raise TypeError("embedding_text must be an explicit string")
                else:
                    position = _value_end(text, position)
                position = whitespace(position)
                if position < len(text) and text[position] == ",":
                    position += 1
                    if text[whitespace(position) : whitespace(position) + 1] == "}":
                        raise ValueError("Trailing object comma")
                    continue
                position = require(position, "}")
                break
            if "embedding_text" not in keys:
                raise ValueError("Every input row needs embedding_text")
            values.append(value)
            position = whitespace(position)
            if position < len(text) and text[position] == ",":
                position += 1
                continue
            position = require(position, "]")
            break
    if whitespace(position) != len(text):
        raise ValueError("Trailing data after embedding inputs")
    return values


def embed_bundle(bundle, output, *, encoder=None):
    """Export one exclusive private embedding artifact; never open labels.json.

    The raw inputs.json hash is checked against its frozen manifest before
    embedding and again before writing. Only embedding_text is decoded from
    rows. Metadata matches wm_small_benchmark.run's exact hash contract.
    """
    bundle, output = Path(bundle).absolute(), Path(output).absolute()
    for path in (bundle, output, bundle / "inputs.json", bundle / "manifest.json"):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Symlinks are not accepted for embedding bundle/output paths")
    if output.exists():
        raise FileExistsError("Embedding output must be a new directory")
    input_path, manifest_path = bundle / "inputs.json", bundle / "manifest.json"
    raw = input_path.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict) or manifest.get("inputs.json") != input_sha:
        raise ValueError("Frozen embedding input hash mismatch")
    texts = _embedding_text_column(raw)
    encoder = get_encoder() if encoder is None else encoder
    vectors = encoder.encode_texts(texts)
    if vectors.shape != (len(texts), DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError("Invalid exported embedding matrix")
    if _file_sha256(input_path) != input_sha or manifest_path.read_bytes() != manifest_raw:
        raise ValueError("Frozen input changed during embedding")
    info = {
        **encoder.metadata(),
        "input_sha256": input_sha,
        "input_path": str(input_path),
        "input_rows": len(texts),
        "input_column": "embedding_text",
        "bundle_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "embedding_source_sha256": _file_sha256(__file__),
        "labels_read": False,
    }
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.chmod(0o700)
    embedding_path = output / "embeddings.npy"
    descriptor = os.open(embedding_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    info["embeddings_sha256"] = _file_sha256(embedding_path)
    descriptor = os.open(output / "metadata.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(info, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--local-only", action="store_true")
    embed = commands.add_parser("embed")
    embed.add_argument("--bundle", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        info = FrozenTextEncoder(args.cache_dir, local_files_only=args.local_only).metadata()
    else:
        info = embed_bundle(args.bundle, args.output)
    print(json.dumps(info, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
