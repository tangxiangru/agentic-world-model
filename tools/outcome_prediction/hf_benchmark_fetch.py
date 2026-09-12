"""Plan or fetch a hash-verified benchmark source subset from a pinned HF dataset.

The default command only prints a plan. ``--fetch`` writes assets, the complete
repository inventory, the explicit selection, and a receipt of verified files
and unavailable selections. Experiment scripts are treated as bytes, never run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote

DATASET = "JerrrrryL/awm-gsm8k-trajectories"
PROJECT = Path(__file__).resolve().parents[2]
MAX_BYTES = 25 * 1024 * 1024
DEFAULT_MIRROR = PROJECT / "data/traj/raw/awm-gsm8k-trajectories-cc2ac9d884a7"


class AssetError(ValueError):
    """A credential-free asset validation failure."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def relative_path(path):
    if not isinstance(path, str):
        raise AssetError("invalid_path")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in path or str(parsed) != path or path == ".":
        raise AssetError("unsafe_path")
    return parsed


def destination(root, path):
    relative_path(path)
    root = Path(root).resolve()
    target = root / path
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise AssetError("symlink_in_destination")
    if not target.resolve().is_relative_to(root):
        raise AssetError("unsafe_destination")
    return target


def token_from_environment(env_file=PROJECT / ".env"):
    """Read credentials into memory; never return them in plans or receipts."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    if Path(env_file).is_file():
        from dotenv import dotenv_values
        return dotenv_values(env_file).get("HF_TOKEN")
    return None


def inventory_from_hf(revision="main", token=None, api=None):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    info = api.dataset_info(DATASET, revision=revision, files_metadata=True)
    if not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise AssetError("invalid_revision")
    return {"schema": "wm-hf-inventory-v1", "repo_id": DATASET, "repo_type": "dataset",
            "requested_revision": revision, "sha": info.sha, "checked_at": utc_now(),
            "last_modified": str(info.last_modified),
            "files": [{"path": s.rfilename, "size": s.size, "blob_id": s.blob_id,
                       "lfs_sha256": s.lfs.sha256 if s.lfs else None} for s in info.siblings]}


def selection_reason(path):
    p = relative_path(path)
    if len(p.parts) == 1 and (p.suffix in {".json", ".jsonl"} or path == "README.md"):
        return "root_manifest"
    if path.startswith("rescore10/results/") and p.suffix == ".json":
        return "ten_run_result"
    if path.startswith("rescore10/eval/"):
        return "evaluator_source"
    if path.startswith("eval_matrix_1k/"):
        if "results" in p.parts and p.suffix == ".json":
            return "matrix_result"
        if p.name == "policies.json" or "reports" in p.parts:
            return "matrix_protocol"
    if path.startswith("checkpoints_meta/") and p.name in {"index.json", "config.json", "generation_config.json"}:
        return "checkpoint_metadata"
    if path.startswith("cells/"):
        if p.name == "solve_out_sanitized.txt":
            return "recipe_reconstruction_local_only"
        if (len(p.parts) >= 4 and p.parts[2] == "wm"
                and (p.parts[3] == "cards" or p.name in {"config.json", "records.jsonl"})):
            return "experiment_recipe"
    if path.startswith("dojo_ab_gsm8k/"):
        if p.name in {"summary.json", "run.json", "dojo_config.json"}:
            return "dojo_metadata"
        if "artifacts" in p.parts and p.name in {"meta.json", "solution.py"}:
            return "dojo_recipe"
    return None


def exclusion_reason(path):
    parts = PurePosixPath(path).parts
    if "trajectories" in parts or "logs" in parts or path.endswith((".log", ".log.gz")):
        return "large_trace_or_runtime_log_not_needed_for_labels"
    if path.startswith("cells/"):
        return "auxiliary_cell_output_outside_recipe_boundary"
    if path.startswith("dojo_ab_gsm8k/"):
        return "auxiliary_dojo_trace_or_output_outside_recipe_boundary"
    if path.startswith("rescore10/relay/"):
        return "checkpoint_transfer_bookkeeping"
    return "outside_benchmark_source_selection"


def build_plan(inventory, assets_root=None, receipt_root=None):
    sha = inventory.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise AssetError("invalid_revision")
    if inventory.get("repo_id", DATASET) != DATASET:
        raise AssetError("wrong_repository")
    selected, excluded = [], []
    seen = set()
    for entry in sorted(inventory["files"], key=lambda x: x["path"]):
        path = entry["path"]
        if path in seen:
            raise AssetError("duplicate_inventory_path")
        seen.add(path)
        reason = selection_reason(path)
        if reason:
            selected.append({**entry, "selection_reason": reason,
                             "local_only": reason == "recipe_reconstruction_local_only"})
        else:
            excluded.append({"path": path, "reason": exclusion_reason(path)})
    return {"schema": "wm-hf-selection-v1", "repo_id": DATASET, "revision": sha,
            "assets_root": str(Path(assets_root or PROJECT / f"data/traj/raw/awm-hf-benchmark-{sha[:12]}").resolve()),
            "receipt_root": str(Path(receipt_root or PROJECT / f"data/analysis/wm_hf_benchmark/source/{sha[:12]}").resolve()),
            "max_download_bytes_per_file": MAX_BYTES,
            "repository_file_count": len(inventory["files"]),
            "selected_file_count": len(selected), "selected": selected, "excluded": excluded,
            "not_selected_reason_counts": dict(Counter(row["reason"] for row in excluded)),
            "scope": "Selected source subset, not a complete repository mirror; no weights or evaluation traces."}


def verify_bytes(raw, entry):
    if len(raw) != entry["size"]:
        raise AssetError("size_mismatch")
    sha256 = hashlib.sha256(raw).hexdigest()
    if entry.get("lfs_sha256"):
        expected, actual, kind = entry["lfs_sha256"], sha256, "lfs_sha256"
    else:
        expected = entry.get("blob_id")
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        kind = "git_blob_sha1"
    if not expected or expected != actual:
        raise AssetError("hash_mismatch")
    return {"sha256": sha256, "bytes": len(raw), "verified_against": kind,
            "expected_hash": expected}


def write_atomic(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_atomic(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def raw_downloader(revision, token=None):
    """Use authenticated raw requests for Git blobs and resolve for LFS objects.

    Response bodies and SDK/HTTP exceptions are never exposed to the caller.
    """
    import requests
    thread_state = threading.local()
    rate_lock = threading.Lock()
    resume_at = [0.0]

    def wait_for_rate_limit():
        while True:
            with rate_lock:
                remaining = resume_at[0] - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))

    def download(entry):
        if not hasattr(thread_state, "session"):
            thread_state.session = requests.Session()
        route = "resolve" if entry.get("lfs_sha256") else "raw"
        url = f"https://huggingface.co/datasets/{DATASET}/{route}/{revision}/{quote(entry['path'], safe='/')}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        for attempt in range(4):
            wait_for_rate_limit()
            try:
                with thread_state.session.get(url, headers=headers, timeout=(10, 45), stream=True) as response:
                    if response.status_code != 200:
                        if response.status_code == 429 and attempt < 3:
                            # HF exposes the fixed-window reset in the RateLimit
                            # header. Share the pause across all eight workers.
                            reset = re.search(r"(?:^|;)t=(\d+)", response.headers.get("RateLimit", ""))
                            delay = min(305, int(reset.group(1)) + 2) if reset else 60
                            with rate_lock:
                                resume_at[0] = max(resume_at[0], time.monotonic() + delay)
                            continue
                        if response.status_code in {500, 502, 503, 504} and attempt < 3:
                            time.sleep(0.5 * 2 ** attempt)
                            continue
                        raise AssetError(f"http_status_{response.status_code}")
                    chunks, size = [], 0
                    for chunk in response.iter_content(1024 * 128):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise AssetError("download_exceeds_size_limit")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except AssetError:
                raise
            except requests.RequestException:
                if attempt == 3:
                    raise AssetError("network_read_failed") from None
                time.sleep(0.5 * 2 ** attempt)
        raise AssetError("network_read_failed")

    return download


def fetch_one(entry, assets_root, mirrors, downloader):
    path = entry["path"]
    target = destination(assets_root, path)
    rejected = []
    candidates = [(target, "verified_cache")]
    candidates.extend((Path(root) / path, "verified_local_mirror") for root in mirrors)
    for candidate, method in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.stat().st_size != entry["size"]:
                raise AssetError("size_mismatch")
            raw = candidate.read_bytes()
            hashes = verify_bytes(raw, entry)
        except (OSError, AssetError):
            rejected.append({"method": method, "reason": "local_copy_did_not_verify"})
            continue
        if candidate != target:
            write_atomic(target, raw)
        return {"path": path, "local_path": str(target), "method": method,
                **hashes, "rejected_local_copies": rejected}
    if entry.get("local_only"):
        raise AssetError("no_verified_local_copy_local_only")
    if entry["size"] > MAX_BYTES:
        raise AssetError("not_downloaded_size_limit")
    raw = downloader(entry)
    hashes = verify_bytes(raw, entry)
    write_atomic(target, raw)
    return {"path": path, "local_path": str(target), "method": "verified_hf_download",
            **hashes, "rejected_local_copies": rejected}


def fetch_assets(inventory, plan, *, mirrors=(), downloader=None, workers=8, progress=None):
    if plan["revision"] != inventory["sha"]:
        raise AssetError("plan_inventory_revision_mismatch")
    downloader = downloader or raw_downloader(plan["revision"], token_from_environment())
    receipt_root = Path(plan["receipt_root"])
    write_json(destination(receipt_root, "inventory.json"), inventory)
    write_json(destination(receipt_root, "selection_plan.json"), plan)
    records, unavailable = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, entry, plan["assets_root"], mirrors, downloader): entry
                   for entry in plan["selected"]}
        for index, future in enumerate(as_completed(futures), 1):
            entry = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- scrub arbitrary transport exception secrets
                # Only our fixed safe error codes may enter a public receipt.
                reason = str(exc) if isinstance(exc, AssetError) else "local_or_network_read_failed"
                unavailable.append({"path": entry["path"], "reason": reason,
                                    "selection_reason": entry["selection_reason"]})
            if progress and (index % 250 == 0 or index == len(futures)):
                progress({"completed": index, "selected": len(futures), "verified": len(records),
                          "unavailable": len(unavailable)})
    records.sort(key=lambda x: x["path"])
    unavailable.sort(key=lambda x: x["path"])
    receipt = {"schema": "wm-hf-fetch-receipt-v1", "repo_id": DATASET, "revision": plan["revision"],
               "finished_at": utc_now(), "assets_root": plan["assets_root"],
               "inventory_sha256": hashlib.sha256((receipt_root / "inventory.json").read_bytes()).hexdigest(),
               "selection_plan_sha256": hashlib.sha256((receipt_root / "selection_plan.json").read_bytes()).hexdigest(),
               "repository_file_count": plan["repository_file_count"],
               "selected_file_count": len(plan["selected"]), "verified_file_count": len(records),
               "unavailable_file_count": len(unavailable), "files": records, "unavailable": unavailable,
               "scope": plan["scope"],
               "source_mapping": {r["path"]: r["local_path"] for r in records}}
    receipt["records"] = [{**r, "status": "verified", "size": r["bytes"], "source": r["method"]}
                          for r in records] + [{**r, "status": "missing", "size": None,
                                               "sha256": None, "source": None} for r in unavailable]
    write_json(destination(receipt_root, "fetch_receipt.json"), receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--inventory", type=Path, help="Use an existing pinned inventory offline")
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--mirror", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fetch", action="store_true", help="Fetch selected assets and write provenance receipts")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.workers <= 16:
            raise AssetError("workers_must_be_between_1_and_16")
        inventory = json.loads(args.inventory.read_text()) if args.inventory else inventory_from_hf(args.revision, token_from_environment())
        plan = build_plan(inventory, args.assets_root, args.receipt_root)
        if not args.fetch:
            print(json.dumps({k: v for k, v in plan.items() if k not in {"selected", "excluded"}}, indent=2))
            return 0
        receipt = fetch_assets(inventory, plan, mirrors=args.mirror or [DEFAULT_MIRROR],
                               workers=args.workers, progress=lambda value: print(json.dumps(value), flush=True))
        print(json.dumps({"revision": receipt["revision"], "verified": receipt["verified_file_count"],
                          "unavailable": receipt["unavailable_file_count"], "receipt_root": plan["receipt_root"]}))
        return 0 if not receipt["unavailable"] else 1
    except AssetError as exc:
        print(json.dumps({"error": str(exc)}))
    except Exception:  # noqa: BLE001 -- never expose raw authentication/transport exceptions
        print(json.dumps({"error": "inventory_or_fetch_failed_check_local_access_and_inputs"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
