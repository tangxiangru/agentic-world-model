"""Operator tooling for the git-as-bus workflow: queue, reconcile, harvest, cancel.

Two agents share one branch. The planner edits manifests under
``experiments/posttrainbench/`` and the queue file, ``queue.yaml``, which is
the desired state: one entry per manifest, ``want: submitted`` or
``want: cancelled``; ``want: staged`` records a committed manifest that must
not reach Slurm yet. The operator, on the cluster, runs
``awm ptb reconcile --apply`` on the configured cadence and commits what it wrote
under ``results/ptb/``: a copy of every receipt, one bundle per finished
cell, and one line per action in ``ops-log.md``. Neither writes the other's
paths, so the branch never needs a merge.

Ownership follows AGENTS.md: the only job ids that may be cancelled are those
in a receipt this repository tracks, only while they are still PENDING; a
running cell is never cancelled by the operator. Nothing here retries: a
retry is the planner adding an entry.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from awm import paths, ptb_results
from awm import ptb_experiments as ptb

QUEUE_PATH = Path("experiments/posttrainbench/queue.yaml")
RESULTS_ROOT = Path("results/ptb")
OPS_LOG = "ops-log.md"
STATUS = "status.json"

TERMINAL_STATES = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
     "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
)
#: The only state the operator cancels from. CONFIGURING already holds resources.
CANCELLABLE_STATES = frozenset({"PENDING"})

PER_FILE_CAP = 2 * 1024 * 1024
ALWAYS_FILES = ("metrics.json", "runtime_provenance.json", "time_taken.txt", "cli_version.txt")
GLOB_FILES = ("judgement_*.json", "final_eval_*.txt")
GZIP_FILES = ("solve_parsed.txt",)
LISTED_ONLY = ("solve_out.txt", "system_monitor.log")
#: What the private WMA sidecar leaves on the shared results volume: its log, and the
#: transcripts it writes under wma_private/. Both are harvested; both are also the earliest
#: in-flight evidence, because the scientist's task tree stays node-local until the job ends.
SIDECAR_LOG = "wma_sidecar.log"
PRIVATE_DIR = "wma_private"
#: A running cell's snapshot lives beside the bundle the harvest will write, and is removed by it.
INFLIGHT_SUFFIX = ".inflight"
INFLIGHT_STATES = frozenset({"RUNNING", "COMPLETING"})
SKIP_DIRS = frozenset(
    {"final_model", ".cache", "__pycache__", ".git", "node_modules", "hf_cache", "wandb",
     ".venv", "venv"}
)
BINARY_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".parquet", ".arrow", ".npy",
     ".npz", ".pkl", ".pickle", ".zip", ".tar", ".gz", ".tgz", ".7z", ".so", ".whl", ".sif"}
)
LOG_TAIL_LINES = 200

# Seams for tests and for the cluster: sacct, the results volume, the validator, the launcher.
job_state = ptb._job_state
result_for_job = ptb.result_for_job
audit_result = ptb.audit_result
submit_batch = ptb.submit


class OpsError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ the queue

def load_queue(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    """The planner's desired state, validated; entries in priority order."""
    if not path.is_file():
        raise OpsError(f"no queue file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise OpsError("queue.yaml must be a mapping with schema_version: 1")
    entries = data.get("entries")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise OpsError("queue.yaml entries must be a list")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"queue entry {index + 1}"
        if not isinstance(entry, dict):
            raise OpsError(f"{where} must be a mapping")
        manifest = str(entry.get("manifest", ""))
        if not manifest.startswith("experiments/posttrainbench/") or not manifest.endswith(".yaml"):
            raise OpsError(f"{where}: manifest must be a committed experiments/posttrainbench/*.yaml")
        if not (repo_root / manifest).is_file():
            raise OpsError(f"{where}: manifest {manifest} does not exist")
        if manifest in seen:
            raise OpsError(f"{where}: manifest {manifest} is listed twice")
        seen.add(manifest)
        if entry.get("want") not in ("submitted", "cancelled", "staged"):
            raise OpsError(f"{where}: want must be submitted, cancelled, or staged")
        if entry.get("pilot") not in (None, "first"):
            raise OpsError(f"{where}: pilot must be absent or 'first'")
        why = entry.get("why")
        if not isinstance(why, str) or not why.strip():
            raise OpsError(f"{where}: why must say, in one line, why this entry is here")
    return entries


# ---------------------------------------------------------------- receipts

RECEIPT_NAME = re.compile(r"^(?P<kind>pilot|formal(?:-retry\d+)?)-(?P<stamp>.+)\.json$")


def _receipt_kind(name: str) -> str:
    """``formal-<iso stamp>.json`` -> formal; ``formal-retry2-...`` -> formal-retry2; ``pilot-...`` -> pilot.

    The stamp is an ISO timestamp with its own hyphens, so the kind is matched, not split off.
    """
    match = RECEIPT_NAME.match(name)
    return match.group("kind") if match else Path(name).stem


def data_receipts(batch: str) -> list[Path]:
    return sorted((paths.data_root() / "ptb" / "batches" / batch).glob("*.json"))


def tracked_receipts(batch: str, repo_root: Path) -> list[Path]:
    return sorted((repo_root / RESULTS_ROOT / batch).glob("*.json"))


def _bundle_status(repo_root: Path, batch: str, cell: str) -> dict[str, Any] | None:
    status = repo_root / RESULTS_ROOT / batch / cell / STATUS
    if not status.is_file():
        return None
    try:
        data = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ------------------------------------------------------------------ the plan

@dataclass(frozen=True)
class Action:
    kind: str  # submit | copy_receipt | cancel | harvest | peek | wait | blocked
    batch: str
    detail: str
    cell: str | None = None
    job_id: str | None = None
    manifest: str | None = None
    pilot: bool = False
    receipt: str | None = None  # receipt file name within the batch, or a source path for copy_receipt
    job_name: str | None = None
    state: str | None = None

    def line(self) -> str:
        where = f"{self.batch}/{self.cell}" if self.cell else self.batch
        job = f" job={self.job_id}" if self.job_id else ""
        pilot = " (pilot)" if self.pilot else ""
        return f"{self.kind} {where}{job}{pilot}: {self.detail}"


def plan(entries: list[dict[str, Any]], repo_root: Path) -> list[Action]:
    """What the queue, the receipts and Slurm say should happen now. Reads only."""
    actions: list[Action] = []
    for entry in entries:
        manifest = ptb.load_manifest(repo_root / entry["manifest"])
        batch = str(manifest["batch_id"])
        tracked = {p.name: p for p in tracked_receipts(batch, repo_root)}
        for source in data_receipts(batch):
            if source.name not in tracked:
                actions.append(Action("copy_receipt", batch, f"{source.name} is not tracked yet",
                                      receipt=str(source)))
                tracked[source.name] = source
        receipts = [(name, ptb.load_receipt(path)) for name, path in sorted(tracked.items())]
        states: dict[str, str] = {}
        for _, receipt in receipts:
            for job in receipt["jobs"]:
                states.setdefault(job["job_id"], job_state(job["job_id"]))
        for name, receipt in receipts:
            if receipt.get("state") != "submitted":
                failure = receipt.get("failure") or {}
                actions.append(Action(
                    "blocked", batch,
                    f"receipt {name} is in state {receipt.get('state')!r}: "
                    f"{failure.get('reason') or failure.get('stderr') or failure.get('error') or 'see the receipt'}",
                    receipt=name))
            if str(receipt.get("kind", "")).startswith("context-smoke-"):
                # Runtime validation has no scientific result bundle.
                continue
            for job in receipt["jobs"]:
                state = states[job["job_id"]]
                if state in TERMINAL_STATES:
                    status = _bundle_status(repo_root, batch, job["cell_id"])
                    if status is None or status.get("job_id") != job["job_id"]:
                        actions.append(Action("harvest", batch, state, cell=job["cell_id"],
                                              job_id=job["job_id"], receipt=name,
                                              job_name=job.get("job_name"), state=state))
                elif state in INFLIGHT_STATES:
                    # analysis does not wait for the job: snapshot what the shared volume holds
                    actions.append(Action("peek", batch, state, cell=job["cell_id"],
                                          job_id=job["job_id"], receipt=name,
                                          job_name=job.get("job_name"), state=state))
        if entry["want"] == "staged":
            if any(not str(r.get("kind", "")).startswith("context-smoke-") for _, r in receipts):
                actions.append(Action(
                    "blocked", batch,
                    "staged entry already has a receipt; choose submitted or cancelled explicitly"))
            continue
        if entry["want"] == "cancelled":
            for name, receipt in receipts:
                done = {c["job_id"] for c in receipt.get("cancellations") or []}
                for job in receipt["jobs"]:
                    state = states[job["job_id"]]
                    if job["job_id"] in done or state in TERMINAL_STATES:
                        continue
                    if state in CANCELLABLE_STATES:
                        actions.append(Action("cancel", batch, entry["why"], cell=job["cell_id"],
                                              job_id=job["job_id"], receipt=name, state=state))
                    else:
                        actions.append(Action(
                            "wait", batch,
                            f"{state}; a cell that has started is never cancelled by the operator",
                            cell=job["cell_id"], job_id=job["job_id"], state=state))
            continue
        if any(_receipt_kind(name).startswith("formal") for name, _ in receipts):
            continue
        if entry.get("pilot") == "first":
            pilots = [(name, r) for name, r in receipts if _receipt_kind(name) == "pilot"]
            if not pilots:
                actions.append(Action("submit", batch, "pilot first", manifest=entry["manifest"],
                                      pilot=True))
                continue
            _, pilot = pilots[-1]
            unfinished = [j for j in pilot["jobs"] if states[j["job_id"]] not in TERMINAL_STATES]
            if unfinished:
                actions.append(Action("wait", batch, f"pilot is {states[unfinished[0]['job_id']]}",
                                      cell=unfinished[0]["cell_id"], job_id=unfinished[0]["job_id"]))
                continue
            statuses = [(j, _bundle_status(repo_root, batch, j["cell_id"])) for j in pilot["jobs"]]
            if any(s is None or s.get("job_id") != j["job_id"] for j, s in statuses):
                actions.append(Action("wait", batch, "pilot finished; its harvest comes first"))
                continue
            if not all(s.get("complete") for _, s in statuses):
                actions.append(Action("blocked", batch,
                                      "pilot did not validate; the planner decides what happens"))
                continue
        actions.append(Action("submit", batch, "formal cells", manifest=entry["manifest"]))
    return actions


# ---------------------------------------------------------------- harvest

def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return "\n".join(text.splitlines()[-lines:]) + "\n"


def _copy_task_tree(task: Path, out: Path, skipped: list[dict[str, Any]]) -> None:
    for root, dirs, files in os.walk(task):
        root_path = Path(root)
        rel_root = root_path.relative_to(task)
        # symlinked directories are recreated as links and never followed
        kept_dirs = []
        for d in sorted(dirs):
            if d in SKIP_DIRS:
                skipped.append({"path": str(rel_root / d), "reason": "directory skipped by policy"})
                continue
            src = root_path / d
            if src.is_symlink():
                dst = out / rel_root / d
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() and not dst.is_symlink():
                    os.symlink(os.readlink(src), dst)
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs
        for f in sorted(files):
            src = root_path / f
            rel = rel_root / f
            dst = out / rel
            if src.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() and not dst.is_symlink():
                    os.symlink(os.readlink(src), dst)
                continue
            try:
                size = src.stat().st_size
            except OSError:
                continue
            if src.suffix.lower() in BINARY_SUFFIXES:
                skipped.append({"path": str(rel), "size": size, "reason": "binary"})
                continue
            if size > PER_FILE_CAP:
                skipped.append({"path": str(rel), "size": size, "reason": f"over {PER_FILE_CAP} bytes"})
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _copy_sidecar(result_dir: Path, out_dir: Path, skipped: list[dict[str, Any]]) -> dict[str, Any]:
    """The sidecar's log and its transcripts (gzipped), from the result directory into ``out_dir``.

    Returns what was found, for status.json / peek.json: whether the log exists and what it
    last said, and the transcripts by name and size. Nothing here is scientist-visible: the
    sidecar writes these outside the task tree.
    """
    found: dict[str, Any] = {"sidecar_log": False, "sidecar_log_tail": None, "transcripts": []}
    log = result_dir / SIDECAR_LOG
    if log.is_file():
        found["sidecar_log"] = True
        if log.stat().st_size > PER_FILE_CAP:
            (out_dir / f"{SIDECAR_LOG}.tail").write_text(_tail(log) or "", encoding="utf-8")
            skipped.append({"path": SIDECAR_LOG, "size": log.stat().st_size,
                            "reason": f"over {PER_FILE_CAP} bytes; tail kept"})
        else:
            shutil.copy2(log, out_dir / SIDECAR_LOG)
        lines = (_tail(log, 5) or "").splitlines()
        found["sidecar_log_tail"] = lines[-1] if lines else ""
    private = result_dir / PRIVATE_DIR
    if private.is_dir():
        for src in sorted(private.rglob("*")):
            if not src.is_file() or src.is_symlink() or not src.resolve().is_relative_to(private.resolve()):
                continue
            if src.suffix not in (".jsonl", ".json", ".yaml", ".yml"):
                continue
            name = str(src.relative_to(private))
            size = src.stat().st_size
            if src.suffix == ".jsonl":
                found["transcripts"].append({"name": name, "size": size})
            rel = f"{PRIVATE_DIR}/{name}"
            if size > PER_FILE_CAP:
                skipped.append({"path": rel, "size": size, "reason": f"over {PER_FILE_CAP} bytes"})
                continue
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".jsonl":
                with src.open("rb") as fin, gzip.open(str(dst) + ".gz", "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            else:
                shutil.copy2(src, dst)
    return found


def peek_job(
    result_dir: Path | None,
    out_dir: Path,
    *,
    batch: str,
    cell: str,
    job_id: str,
    state: str | None = None,
) -> dict[str, Any]:
    """Snapshot a running cell: the sidecar log, the transcripts so far, the scientist's stdout tail.

    Overwrites ``out_dir`` each time, so the operator's commit carries only what changed. The
    task tree (cards, verdicts, .wma) is not here yet: PTB copies it to the results volume only
    when the job ends. With no result directory yet, peek.json says so and nothing else is written.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    skipped: list[dict[str, Any]] = []
    peek: dict[str, Any] = {
        "schema_version": 1,
        "batch": batch,
        "cell": cell,
        "job_id": job_id,
        "slurm_state": state,
        "result_dir": str(result_dir) if result_dir else None,
        "peeked_at": _now(),
        "sidecar_log": False,
        "sidecar_log_tail": None,
        "transcripts": [],
        "solve_out_lines": None,
        "skipped": skipped,
    }
    if result_dir is not None and result_dir.is_dir():
        peek.update(_copy_sidecar(result_dir, out_dir, skipped))
        solve_out = result_dir / "solve_out.txt"
        if solve_out.is_file():
            tail = _tail(solve_out)
            if tail is not None:
                (out_dir / "solve_out.tail").write_text(tail, encoding="utf-8")
            try:
                with solve_out.open("rb") as f:
                    peek["solve_out_lines"] = sum(1 for _ in f)
            except OSError:
                pass
    (out_dir / "peek.json").write_text(json.dumps(peek, indent=2) + "\n", encoding="utf-8")
    return peek


def harvest_job(
    result_dir: Path | None,
    out_dir: Path,
    *,
    batch: str,
    cell: str,
    job_id: str,
    job_name: str | None = None,
    state: str | None = None,
    slurm_log_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the small, readable part of one cell's result into ``out_dir``; write status.json.

    Weights never; anything over PER_FILE_CAP or binary is listed, not copied;
    solve_parsed.txt is gzipped. With no result directory (the job died before
    PTB wrote one) the bundle is the status and the Slurm log tails.
    """
    if out_dir.exists():
        previous: dict[str, Any] | None = None
        try:
            previous = json.loads((out_dir / STATUS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict) and previous.get("job_id") not in (None, job_id):
            os.rename(out_dir, out_dir.parent / f"{cell}.j{previous['job_id']}")
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    inflight = out_dir.parent / f"{cell}{INFLIGHT_SUFFIX}"
    if inflight.is_dir():
        shutil.rmtree(inflight)  # the bundle supersedes the snapshots
    skipped: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "schema_version": 1,
        "batch": batch,
        "cell": cell,
        "job_id": job_id,
        "job_name": job_name,
        "slurm_state": state,
        "result_dir": str(result_dir) if result_dir else None,
        "complete": False,
        "issues": [],
        "accuracy": None,
        "stderr": None,
        "judge_flags": [],
        "awm_sha": None,
        "harvested_at": _now(),
        "sidecar_log": False,
        "transcripts": [],
        "skipped": skipped,
    }
    if result_dir is not None and result_dir.is_dir():
        for name in ALWAYS_FILES:
            if (result_dir / name).is_file():
                shutil.copy2(result_dir / name, out_dir / name)
        sidecar = _copy_sidecar(result_dir, out_dir, skipped)
        status["sidecar_log"] = sidecar["sidecar_log"]
        status["transcripts"] = sidecar["transcripts"]
        for pattern in GLOB_FILES:
            for src in sorted(result_dir.glob(pattern)):
                if src.stat().st_size > PER_FILE_CAP:
                    skipped.append({"path": src.name, "size": src.stat().st_size,
                                    "reason": f"over {PER_FILE_CAP} bytes"})
                else:
                    shutil.copy2(src, out_dir / src.name)
        for name in GZIP_FILES:
            src = result_dir / name
            if src.is_file():
                with src.open("rb") as fin, gzip.open(out_dir / f"{name}.gz", "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        for name in LISTED_ONLY:
            src = result_dir / name
            if src.is_file():
                skipped.append({"path": name, "size": src.stat().st_size, "reason": "listed only"})
        task = result_dir / "task"
        if task.is_dir():
            _copy_task_tree(task, out_dir / "task", skipped)
            sandbox = task / "awm_sandbox.json"
            if sandbox.is_file():
                try:
                    status["awm_sha"] = json.loads(sandbox.read_text(encoding="utf-8")).get("sha")
                except (OSError, ValueError):
                    pass
        metrics = ptb_results._read_json(result_dir / "metrics.json")
        if isinstance(metrics.get("accuracy"), (int, float)):
            status["accuracy"] = metrics["accuracy"]
            status["stderr"] = metrics.get("stderr")
        status["issues"] = audit_result(result_dir)
        status["complete"] = not status["issues"]
        status["judge_flags"] = ptb_results.judge_flags(result_dir)
    else:
        status["issues"] = ["result directory not found"]
    if slurm_log_dir is not None and job_name:
        for suffix in ("out", "err"):
            tail = _tail(slurm_log_dir / f"{job_name}-{job_id}.{suffix}")
            if tail is not None:
                (out_dir / f"slurm.{suffix}.tail").write_text(tail, encoding="utf-8")
    (out_dir / STATUS).write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


# ------------------------------------------------------------------ cancel

def _scancel_command(job_id: str) -> list[str]:
    env = ptb.read_ptb_env()
    if env.get("POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT") == "1":
        return ["sudo", "scancel", job_id]
    return ["scancel", job_id]


def cancel_job(receipt_path: Path, cell: str, reason: str) -> dict[str, Any]:
    """scancel one PENDING job named in a tracked receipt; record it in that receipt."""
    receipt = ptb.load_receipt(receipt_path)
    jobs = [job for job in receipt["jobs"] if job["cell_id"] == cell]
    if not jobs:
        raise OpsError(f"{receipt_path.name} has no cell {cell}")
    job = jobs[-1]
    state = job_state(job["job_id"])
    if state not in CANCELLABLE_STATES:
        raise OpsError(f"{cell} job {job['job_id']} is {state}; only PENDING jobs are cancelled")
    done = subprocess.run(_scancel_command(job["job_id"]), text=True, capture_output=True, check=False)
    if done.returncode:
        raise OpsError(f"scancel {job['job_id']} failed: {done.stderr.strip()}")
    record = {"cell_id": cell, "job_id": job["job_id"], "reason": reason,
              "state_before": state, "at": _now()}
    receipt.setdefault("cancellations", []).append(record)
    receipt.pop("_path", None)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return record


# ------------------------------------------------------------------- apply

def _worktree_dirty(repo_root: Path) -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, text=True,
                          capture_output=True, check=False).stdout.strip()


def _log(repo_root: Path, line: str) -> None:
    log = repo_root / RESULTS_ROOT / OPS_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.is_file():
        log.write_text("# Operator log: one line per action, newest last\n\n", encoding="utf-8")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"- {_now()} {line}\n")


def apply(actions: list[Action], repo_root: Path) -> list[str]:
    """Do what ``plan`` said, in the order that keeps the launcher's worktree gate happy.

    Submits first (they need a clean tree and write only to the gitignored data
    volume), then receipt copies, cancellations and harvests, which dirty the
    tree for the operator to commit. Returns the log lines written.
    """
    written: list[str] = []
    submits = [a for a in actions if a.kind == "submit"]
    others = [a for a in actions if a.kind not in ("submit", "wait", "blocked")]
    if submits and (dirty := _worktree_dirty(repo_root)):
        line = f"blocked submit: the worktree is not clean; commit first\n    {dirty.splitlines()[0]} ..."
        _log(repo_root, line)
        written.append(line)
        submits = []
    # Every submit runs before anything is written under results/: the launcher freezes
    # the source only from a clean tree, and copying the first receipt in would have
    # blocked the second manifest of the same round (seen 2026-09-02, round 01).
    submitted: list[tuple[Action, Path]] = []
    blocked_lines: list[str] = []
    for action in submits:
        manifest = ptb.load_manifest(repo_root / str(action.manifest))
        try:
            receipt_path = submit_batch(manifest, pilot=action.pilot)
        except ptb.ExperimentError as exc:
            blocked_lines.append(f"blocked submit {action.batch}{' (pilot)' if action.pilot else ''}: "
                                 f"{str(exc).splitlines()[0]}")
            blocked = repo_root / RESULTS_ROOT / action.batch / "blocked.md"
            blocked.parent.mkdir(parents=True, exist_ok=True)
            blocked.write_text(f"# {action.batch}: submission blocked\n\n{_now()}\n\n```\n{exc}\n```\n",
                               encoding="utf-8")
            continue
        submitted.append((action, Path(receipt_path)))
    for line in blocked_lines:
        _log(repo_root, line)
        written.append(line)
    for action, receipt_path in submitted:
        dst = repo_root / RESULTS_ROOT / action.batch / receipt_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(receipt_path, dst)
        stale = dst.parent / "blocked.md"
        if stale.is_file():
            stale.unlink()  # an earlier round's refusal, now superseded by the receipt
        jobs = ptb.load_receipt(dst)["jobs"]
        line = (f"submit {action.batch}{' (pilot)' if action.pilot else ''}: "
                f"{len(jobs)} job(s) {','.join(j['job_id'] for j in jobs)} -> {dst.relative_to(repo_root)}")
        _log(repo_root, line)
        written.append(line)
    for action in others:
        if action.kind == "copy_receipt":
            source = Path(str(action.receipt))
            dst = repo_root / RESULTS_ROOT / action.batch / source.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dst)
            line = f"copy_receipt {action.batch}: {source.name}"
        elif action.kind == "cancel":
            receipt_path = repo_root / RESULTS_ROOT / action.batch / str(action.receipt)
            try:
                record = cancel_job(receipt_path, str(action.cell), action.detail)
            except OpsError as exc:
                line = f"cancel {action.batch}/{action.cell} job={action.job_id} did not happen: {exc}"
            else:
                line = f"cancel {action.batch}/{action.cell} job={record['job_id']} ({record['state_before']}): {action.detail}"
        elif action.kind == "peek":
            result_dir = result_for_job(str(action.job_id))
            out_dir = repo_root / RESULTS_ROOT / action.batch / f"{action.cell}{INFLIGHT_SUFFIX}"
            peek = peek_job(result_dir, out_dir, batch=action.batch, cell=str(action.cell),
                            job_id=str(action.job_id), state=action.state)
            if peek["result_dir"] is None:
                what = "no result directory yet"
            else:
                sidecar = ("sidecar: " + (peek["sidecar_log_tail"] or "log empty")) if peek["sidecar_log"] else "no sidecar log"
                what = (f"{sidecar}; {len(peek['transcripts'])} transcript(s); "
                        f"solve_out {peek['solve_out_lines']} lines")
            line = f"peek {action.batch}/{action.cell} job={action.job_id} {action.state}: {what}"
        elif action.kind == "harvest":
            result_dir = result_for_job(str(action.job_id))
            out_dir = repo_root / RESULTS_ROOT / action.batch / str(action.cell)
            status = harvest_job(result_dir, out_dir, batch=action.batch, cell=str(action.cell),
                                 job_id=str(action.job_id), job_name=action.job_name,
                                 state=action.state, slurm_log_dir=ptb.PTB_ROOT / "logs" / "slurm")
            verdict = "complete" if status["complete"] else f"incomplete ({len(status['issues'])} issue(s))"
            flags = ",".join(status["judge_flags"]) or "clean"
            acc = "" if status["accuracy"] is None else f" acc={status['accuracy']:.4f}"
            line = f"harvest {action.batch}/{action.cell} job={action.job_id} {action.state}{acc} {verdict} {flags}"
        else:  # pragma: no cover - plan only emits the kinds above
            continue
        _log(repo_root, line)
        written.append(line)
    return written


# --------------------------------------------------------------------- CLI

def reconcile_cli(args) -> int:
    repo_root = paths.REPO_ROOT
    try:
        entries = load_queue(repo_root / args.queue, repo_root)
        actions = plan(entries, repo_root)
    except (OpsError, ptb.ExperimentError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if not actions:
        print("nothing to do")
        return 0
    for action in actions:
        print(("  " if args.apply else "would ") + action.line())
    if not args.apply:
        return 0
    for line in apply(actions, repo_root):
        print(f"did {line}")
    return 0


def harvest_cli(args) -> int:
    out = paths.REPO_ROOT / RESULTS_ROOT / args.batch / args.cell if args.out is None else Path(args.out)
    status = harvest_job(Path(args.result_dir) if args.result_dir else None, out, batch=args.batch,
                         cell=args.cell, job_id=args.job, job_name=args.job_name, state=args.state,
                         slurm_log_dir=ptb.PTB_ROOT / "logs" / "slurm")
    print(f"wrote {out / STATUS}: {'complete' if status['complete'] else 'incomplete'}, "
          f"{len(status['skipped'])} file(s) listed but not copied")
    return 0 if status["complete"] else 1


def cancel_cli(args) -> int:
    try:
        record = cancel_job(Path(args.receipt), args.cell, args.reason)
    except (OpsError, ptb.ExperimentError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"cancelled {args.cell} job {record['job_id']} ({record['state_before']}); recorded in {args.receipt}")
    return 0
