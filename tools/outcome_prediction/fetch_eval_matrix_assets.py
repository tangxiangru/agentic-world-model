"""Fetch pinned evaluator sources and selected metadata, never weights or jobs.

Dry-run is stdlib-only and offline. Actual fetch lazily uses huggingface_hub.
Initialize the separately pinned PostTrainBench submodule for AIME task/scorer
sources; the historical HF evaluator kit alone is not a complete executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

DATASET = "JerrrrryL/awm-gsm8k-trajectories"
EVAL_REV = "cc2ac9d884a7d962a6024ab0d5cd8ed3370070de"
META_REV = "446127629d7b271d537390e69bfb2d960a3aa515"
EVAL_PREFIX = "rescore10/eval/"
RUNTIME_README = "rescore10/trajectories/README.md"
DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "experiments/eval_matrix_1k"
PHASES = {"pilot": "1_operational_pilot", "development": "2_development_core",
          "extensions": "3_diagnostic_extensions", "test": "4_locked_test"}
MAX_BYTES = 64 * 1024 * 1024


class AssetError(ValueError):
    """A safe, user-facing asset validation failure."""


def relative_path(value):
    if not isinstance(value, str):
        raise AssetError("Asset path must be text")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or str(p) != value or value == ".":
        raise AssetError("Unsafe or noncanonical asset path")
    return p


def known_hashes(protocol):
    hashes = {}
    for assets in protocol.get("benchmark_files", {}).values():
        for asset in assets:
            path = asset.get("hf_path", asset.get("source_path", asset.get("path", "")))
            if EVAL_PREFIX not in path:
                raise AssetError("Benchmark asset needs an HF rescore10/eval source path")
            path = path[path.index(EVAL_PREFIX):]
            relative_path(path)
            sha = asset.get("sha256", "")
            if not re.fullmatch(r"[a-f0-9]{64}", sha):
                raise AssetError("Invalid protocol asset SHA256")
            if path in hashes and hashes[path] != sha:
                raise AssetError("Conflicting protocol asset hashes")
            hashes[path] = sha
    if not hashes:
        raise AssetError("Protocol has no benchmark asset hashes")
    return hashes


def build_plan(bundle=DEFAULT_BUNDLE, phase="pilot"):
    bundle = Path(bundle).resolve()
    if phase not in {*PHASES, "all"}:
        raise AssetError("Unknown phase")
    source = "experiment_matrix.jsonl" if phase == "all" else f"phases/{PHASES[phase]}.jsonl"
    rows = [json.loads(line) for line in (bundle / source).read_text().splitlines() if line.strip()]
    ids = sorted({row["checkpoint_id"] for row in rows})
    if not ids or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x)
                      for x in ids):
        raise AssetError("Invalid or empty checkpoint selection")
    protocol = json.loads((bundle / "protocol.json").read_text())
    return {"schema": "eval-matrix-asset-plan-v1", "bundle": str(bundle), "phase": phase,
            "repo_id": DATASET, "repo_type": "dataset", "eval_revision": EVAL_REV,
            "metadata_revision": META_REV, "eval_prefix": EVAL_PREFIX,
            "runtime_readme": RUNTIME_README, "expected_sha256": known_hashes(protocol),
            "checkpoint_ids": ids, "metadata_files": [f"checkpoints_meta/{eid}/{name}"
                for eid in ids for name in ("config.json", "generation_config.json")],
            "note": "Sources/metadata only; no weights, labels, execution, or launch approval."}


def eval_file(path):
    p = relative_path(path)
    return (path.startswith(EVAL_PREFIX)
            and not any(part.startswith(".") for part in p.parts)
            and not {"results", "checkpoints", "trajectories"}.intersection(p.parts)
            and p.suffix in {".py", ".json", ".sh", ".jinja", ".md", ".txt", ".toml", ".yaml", ".yml"})


def destination(root, filename):
    relative_path(filename)
    target = root / filename
    for part in (target, *target.parents):
        if part == root:
            break
        if part.is_symlink():
            raise AssetError("Refusing a symlink inside the asset cache")
    if not target.resolve().is_relative_to(root):
        raise AssetError("Asset escapes cache directory")
    return target


def fetch_one(root, revision, filename, expected, downloader):
    target = destination(root, f"{revision}/{filename}")
    receipt = destination(root, f"{revision}/{filename}.receipt.json")
    identity = {"schema": "eval-matrix-asset-receipt-v1", "repo_id": DATASET,
                "repo_type": "dataset", "revision": revision, "filename": filename}
    if target.exists() or receipt.exists():
        if not target.is_file() or not receipt.is_file():
            raise AssetError("Existing unverified cache file is preserved; choose a fresh cache")
        record = json.loads(receipt.read_text())
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if (any(record.get(k) != v for k, v in identity.items()) or record.get("sha256") != sha
                or (expected and sha != expected)):
            raise AssetError("Existing cache receipt/hash mismatch; files were not overwritten")
        return record
    source = Path(downloader(repo_id=DATASET, repo_type="dataset", revision=revision, filename=filename))
    if source.stat().st_size > MAX_BYTES:
        raise AssetError("Asset exceeds the small-file size limit")
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected and sha != expected:
        raise AssetError(f"Protocol SHA256 mismatch: {filename}")
    if filename.startswith("checkpoints_meta/") and not isinstance(json.loads(raw), dict):
        raise AssetError("Checkpoint metadata must be a JSON object")
    record = {**identity, "sha256": sha, "bytes": len(raw)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as f:
        f.write(raw)
    with receipt.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    return record


def fetch_assets(plan, out, *, api=None, downloader=None):
    root = Path(out).resolve()
    if root.is_relative_to(Path(plan["bundle"]).resolve()):
        raise AssetError("Cache output must be outside the proposal bundle")
    if api is None or downloader is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi()
        downloader = downloader or hf_hub_download
    files = []
    for entry in api.list_repo_tree(DATASET, repo_type="dataset", revision=EVAL_REV,
                                   path_in_repo=EVAL_PREFIX.rstrip("/"), recursive=True):
        if getattr(entry, "size", None) is None:
            continue
        if not eval_file(entry.path) or entry.size > MAX_BYTES:
            raise AssetError("Unexpected or oversized file in pinned evaluator tree")
        files.append(entry.path)
    if not files or not set(plan["expected_sha256"]).issubset(files):
        raise AssetError("Pinned evaluator tree is missing required protocol assets")
    jobs = [(EVAL_REV, path) for path in sorted(set(files + [RUNTIME_README]))]
    jobs += [(META_REV, path) for path in plan["metadata_files"]]
    records = []
    for revision, path in jobs:
        records.append(fetch_one(root, revision, path, plan["expected_sha256"].get(path), downloader))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--phase", choices=[*PHASES, "all"], default="pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the pinned plan; no network, auth, or writes")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.bundle, args.phase)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            records = fetch_assets(plan, args.out)
            print(json.dumps({"files_verified": len(records), "cache": str(args.out.resolve()),
                              "note": plan["note"]}))
            print("AIME task/scorer sources: initialize the pinned PostTrainBench submodule separately.")
        return 0
    except AssetError as exc:
        print(f"Asset validation failed: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - avoid leaking credentials in SDK exceptions
        status = getattr(getattr(exc, "response", None), "status_code", None)
        message = "Authentication/read access failed; configure HF access locally." if status in (401, 403) else "Fetch/read failed; check inputs, network, and local cache."
        print(f"{message} ({type(exc).__name__})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
