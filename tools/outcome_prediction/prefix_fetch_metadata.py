"""Fetch small archived checkpoint metadata with per-file provenance receipts.

Never downloads weight shards, modifies cloud objects, or changes authentication.
Existing local files are reused only when their recorded hash and URI match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tools.outcome_prediction.prefix_dataset import DEFAULT_MIRROR, ROOT


def fetch_one(checkpoint, destination, filenames, previous=None):
    exp_id = checkpoint["id"]
    if Path(exp_id).name != exp_id or exp_id in {".", ".."}:
        raise ValueError("Unsafe checkpoint ID")
    result = {"exp_id": exp_id, "files": {}}
    directory = destination / exp_id
    directory.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        uri = checkpoint["gs_path"].rstrip("/") + "/" + name
        target = directory / name
        old = (previous or {}).get("files", {}).get(name, {})
        if target.exists():
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            if (
                old.get("source_uri") == uri
                and old.get("sha256") == sha
                and old.get("status") == "downloaded"
            ):
                result["files"][name] = old
                continue
            result["files"][name] = {
                "status": "unverified_local_file_not_overwritten",
                "source_uri": uri,
            }
            continue
        try:
            completed = subprocess.run(
                ["gcloud", "storage", "cat", uri], capture_output=True, timeout=60, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["files"][name] = {
                "status": "read_failed",
                "source_uri": uri,
                "error": type(exc).__name__,
            }
            continue
        if completed.returncode:
            error = completed.stderr.decode(errors="replace")
            result["files"][name] = {
                "status": "read_failed",
                "source_uri": uri,
                "error": error[:1500],
            }
            continue
        try:
            if not isinstance(json.loads(completed.stdout), dict):
                raise TypeError("metadata must be a JSON object")
        except (ValueError, TypeError, UnicodeDecodeError):
            result["files"][name] = {"status": "invalid_json", "source_uri": uri}
            continue
        # Exclusive create protects concurrent work / unexpected existing files.
        with target.open("xb") as handle:
            handle.write(completed.stdout)
        result["files"][name] = {
            "status": "downloaded",
            "source_uri": uri,
            "sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "bytes": len(completed.stdout),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    return result


def fetch(mirror, destination, filenames, workers=8, limit=None):
    mirror, destination = Path(mirror).resolve(), Path(destination).resolve()
    if destination == mirror or mirror in destination.parents:
        raise ValueError("Metadata cache must be outside read-only mirror")
    manifest_path = mirror / "rescore10/relay/gcs_manifest.json"
    checkpoints = json.loads(manifest_path.read_text())["checkpoints"]
    if limit is not None:
        checkpoints = checkpoints[:limit]
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "manifest.json"
    prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    previous = {item["exp_id"]: item for item in prior.get("checkpoints", [])}
    if (
        prior
        and prior.get("source_manifest_sha256")
        != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Metadata cache belongs to a different manifest")
    results = dict(previous)

    def save():
        receipt = {
            "schema": "checkpoint-metadata-fetch-v1",
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "checkpoints": [results[key] for key in sorted(results)],
        }
        staging = receipt_path.with_suffix(".json.tmp")
        staging.write_text(json.dumps(receipt, indent=2) + "\n")
        staging.replace(receipt_path)

    if not checkpoints:
        save()
        return results
    # Probe once before parallelizing, avoiding hundreds of identical auth errors.
    first = fetch_one(checkpoints[0], destination, filenames, previous.get(checkpoints[0]["id"]))
    results[first["exp_id"]] = first
    save()
    if any("auth" in item.get("error", "").lower() for item in first["files"].values()):
        raise RuntimeError(
            "Google Cloud authentication failed. Run gcloud auth login, then rerun this command."
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                fetch_one, checkpoint, destination, filenames, previous.get(checkpoint["id"])
            )
            for checkpoint in checkpoints[1:]
        ]
        for future in as_completed(futures):
            result = future.result()
            results[result["exp_id"]] = result
            save()
            print(
                json.dumps(
                    {
                        "exp_id": result["exp_id"],
                        "status": {key: value["status"] for key, value in result["files"].items()},
                    }
                ),
                flush=True,
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/analysis/wm_exp_designs/prefix_generation_metadata"
    )
    parser.add_argument("--include-model-metadata", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    filenames = ["generation_config.json"]
    if args.include_model_metadata:
        filenames += ["config.json", "tokenizer_config.json", "special_tokens_map.json"]
    if not 1 <= args.workers <= 32 or (args.limit is not None and args.limit < 1):
        parser.error("workers must be1..32 and limit positive")
    try:
        fetch(args.mirror, args.out, filenames, args.workers, args.limit)
    except RuntimeError as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
