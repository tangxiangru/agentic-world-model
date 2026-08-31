#!/usr/bin/env python3
"""Validate the study's token-free, exact Gemma base-model cache.

The accepted cache is deliberately an allowlist, not a normal mutable
Hugging Face home.  ``quick`` verifies its complete topology, link targets,
and byte sizes.  ``full`` additionally hashes every artifact (including both
weight shards) and is used by the one-cell release smoke or once per site
cache before a production dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL_ID = "google/gemma-3-4b-pt"
REVISION = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
SCHEMA = "awm-base-model-cache-v1"
MODEL_REL = Path("hub/models--google--gemma-3-4b-pt")

# name: (resolved blob name, bytes, content SHA-256)
MODEL_FILES: dict[str, tuple[str, int, str]] = {
    ".gitattributes": (
        "e6d43da7f2cc0124e691f2c7a2092990d8c69838",
        1760,
        "f5560e52a0e066bf2d8c7d3f0ba64cc64396055e6754ef40aa779dec662747b0",
    ),
    "README.md": (
        "5495cfba86b02a597a8d6da9eb4e40fbe7211084",
        24011,
        "720809c69c7d9664ce64b73307231ab57c17896fa0677e5e24d343020333e050",
    ),
    "added_tokens.json": (
        "e17bde03d42feda32d1abfca6d3b598b9a020df7",
        35,
        "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    ),
    "config.json": (
        "cac8f6c9892491efacb9985453947a3ed6d0dfee",
        815,
        "ae75620513f4a95fb36bcf6c09eeab02938e58051871cfe47ead2a7e79a43848",
    ),
    "generation_config.json": (
        "37a4c871d263a349f50e4a313db3e72164950702",
        215,
        "fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3",
    ),
    "model-00001-of-00002.safetensors": (
        "8f5b423a203c41b5655b2e1b4c58be262cb0e403048c8116e100bdac91318320",
        4961251752,
        "8f5b423a203c41b5655b2e1b4c58be262cb0e403048c8116e100bdac91318320",
    ),
    "model-00002-of-00002.safetensors": (
        "d3d22b1ab51a03678f95f7d5e60f703a50e6554e4c731c884fc0fbf05e6e3f47",
        3639026128,
        "d3d22b1ab51a03678f95f7d5e60f703a50e6554e4c731c884fc0fbf05e6e3f47",
    ),
    "model.safetensors.index.json": (
        "4b95241f208f06d324d17c9675568ec58dafd9fb",
        90558,
        "77f4b67de084c31c7bcd373b039908108eee6c6181607e6d53da730e5f0bc659",
    ),
    "preprocessor_config.json": (
        "b1e00fc184f61b698181821169c6374cd5813e5c",
        570,
        "f688d6bb20c5017601c4011de7ca656da8485b540b05013efdaf986c0fcc918d",
    ),
    "processor_config.json": (
        "453c7966d4b5d0b4a317c585989f64c58c2a6bf0",
        70,
        "3ffd5f11778dc73e2b69b3c00535e4121e1badf7018136263cd17b5b34fbaa53",
    ),
    "special_tokens_map.json": (
        "1a6193244714d3d78be48666cb02cdbfac62ad86",
        662,
        "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    ),
    "tokenizer.json": (
        "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
        33384570,
        "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
    ),
    "tokenizer.model": (
        "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        4689074,
        "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    ),
    "tokenizer_config.json": (
        "84b92a5059f1f96e959c4caf68f7aada2cfb2e77",
        1155389,
        "351d94ef392afce9d2f914e31ed283217d6fd9353af9ef0b156eb1925b1e61b9",
    ),
}

EXTRA_FILES: dict[str, tuple[int, str]] = {
    "hub/CACHEDIR.TAG": (
        191,
        "f6572428f6d5e1575e73a1502895a8731f10757dfbb634909c6e154b849af91d",
    ),
    ("hub/.locks/models--google--gemma-3-4b-pt/cac8f6c9892491efacb9985453947a3ed6d0dfee.lock"): (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    f"{MODEL_REL.as_posix()}/refs/main": (
        41,
        "cd3875fe682263622c3cc1df84a78e8ca3381fc2d5e61b5e6a919adb24f22be2",
    ),
    f"{MODEL_REL.as_posix()}/trees/{REVISION}.json": (
        2331,
        "8e2a1aadcc325f0ced050a2392c29b7ae3157d128849bdffe4a72c62441ea806",
    ),
}


class CacheError(RuntimeError):
    """The cache is not the exact public model-only study input."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def expected_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files": {
            name: {"blob": blob, "bytes": size, "sha256": digest}
            for name, (blob, size, digest) in sorted(MODEL_FILES.items())
        },
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def validate(root: Path, *, full_hash: bool) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise CacheError(f"cache root is missing, linked, or not a directory: {root}")
    root = root.resolve()
    model_root = root / MODEL_REL
    snapshot = model_root / "snapshots" / REVISION
    if model_root.is_symlink() or snapshot.is_symlink() or not snapshot.is_dir():
        raise CacheError("pinned official model snapshot is missing or linked")

    expected_dirs = {
        "hub",
        "hub/.locks",
        "hub/.locks/models--google--gemma-3-4b-pt",
        MODEL_REL.as_posix(),
        f"{MODEL_REL.as_posix()}/blobs",
        f"{MODEL_REL.as_posix()}/refs",
        f"{MODEL_REL.as_posix()}/snapshots",
        f"{MODEL_REL.as_posix()}/snapshots/{REVISION}",
        f"{MODEL_REL.as_posix()}/trees",
    }
    expected_regular = set(EXTRA_FILES)
    expected_regular.update(
        f"{MODEL_REL.as_posix()}/blobs/{blob}" for blob, _size, _digest in MODEL_FILES.values()
    )
    expected_links = {f"{MODEL_REL.as_posix()}/snapshots/{REVISION}/{name}" for name in MODEL_FILES}
    actual_dirs: set[str] = set()
    actual_regular: set[str] = set()
    actual_links: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            actual_links.add(relative)
        elif path.is_dir():
            actual_dirs.add(relative)
        elif path.is_file():
            actual_regular.add(relative)
        else:
            raise CacheError(f"cache contains a special filesystem entry: {relative}")
    if actual_dirs != expected_dirs:
        raise CacheError("cache directory inventory is not the exact model-only allowlist")
    if actual_regular != expected_regular:
        raise CacheError("cache file inventory is not the exact model-only allowlist")
    if actual_links != expected_links:
        raise CacheError("cache link inventory is not the exact pinned snapshot allowlist")

    for name, (blob, size, digest) in MODEL_FILES.items():
        link = snapshot / name
        if os.readlink(link) != f"../../blobs/{blob}":
            raise CacheError(f"unexpected snapshot link target: {name}")
        resolved = link.resolve(strict=True)
        expected_blob = (model_root / "blobs" / blob).resolve(strict=True)
        if resolved != expected_blob or not regular(resolved) or resolved.stat().st_size != size:
            raise CacheError(f"invalid model artifact: {name}")
        if full_hash and sha256_file(resolved) != digest:
            raise CacheError(f"content hash mismatch: {name}")

    for relative, (size, digest) in EXTRA_FILES.items():
        path = root / relative
        if not regular(path) or path.stat().st_size != size:
            raise CacheError(f"invalid cache metadata: {relative}")
        if full_hash and sha256_file(path) != digest:
            raise CacheError(f"cache metadata hash mismatch: {relative}")
    if (model_root / "refs" / "main").read_text().strip() != REVISION:
        raise CacheError("refs/main does not name the pinned revision")

    manifest = expected_manifest()
    return {
        "schema_version": SCHEMA,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "file_count": len(MODEL_FILES),
        "model_bytes": sum(size for _blob, size, _digest in MODEL_FILES.values()),
        "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "verification": "full-content-hash" if full_hash else "topology-size",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--study-input", type=Path)
    args = parser.parse_args(argv)
    try:
        evidence = validate(args.cache, full_hash=args.full_hash)
        if args.record:
            atomic_json(args.record, evidence)
        if args.study_input:
            if not regular(args.study_input):
                raise CacheError(f"study-input is missing, linked, or invalid: {args.study_input}")
            study = json.loads(args.study_input.read_text())
            if not isinstance(study, dict) or "base_model" in study:
                raise CacheError("study-input is invalid or already contains base_model")
            study["base_model"] = evidence
            atomic_json(args.study_input, study)
    except (CacheError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
