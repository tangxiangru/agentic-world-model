"""Operator tooling for the git-as-bus workflow: queue, reconcile, harvest, cancel.

Two agents share one branch. The planner edits manifests under
``experiments/posttrainbench/`` and the queue file, ``queue.yaml``, which is
the desired state: one entry per manifest, ``want: submitted`` or
``want: cancelled``. The operator, on the cluster, runs
``awm ptb reconcile --apply`` every few minutes and commits what it wrote
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

from awm import paths, ptb_results, slurm_queue
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
release_batch = ptb.release_held
ownership_snapshot = slurm_queue.collect_snapshot


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
        if entry.get("want") not in ("submitted", "held", "cancelled"):
            raise OpsError(f"{where}: want must be submitted, held, or cancelled")
        if entry.get("pilot") not in (None, "first"):
            raise OpsError(f"{where}: pilot must be absent or 'first'")
        if entry.get("want") == "held" and entry.get("pilot") is not None:
            raise OpsError(f"{where}: a held buffer cannot use a pilot")
        why = entry.get("why")
        if not isinstance(why, str) or not why.strip():
            raise OpsError(f"{where}: why must say, in one line, why this entry is here")
        release_override = entry.get("release_override")
        if release_override is not None:
            if not isinstance(release_override, dict):
                raise OpsError(f"{where}: release_override must be a mapping")
            allowed = {"allow_shared_reservation", "authorized_by", "authorized_at", "reason"}
            unknown = set(release_override) - allowed
            if unknown:
                raise OpsError(
                    f"{where}: release_override has unknown field(s): "
                    f"{', '.join(sorted(unknown))}"
                )
            if entry.get("want") != "submitted":
                raise OpsError(f"{where}: release_override requires want: submitted")
            if release_override.get("allow_shared_reservation") is not True:
                raise OpsError(
                    f"{where}: release_override must set allow_shared_reservation: true"
                )
            for field in ("authorized_by", "authorized_at", "reason"):
                value = release_override.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise OpsError(f"{where}: release_override.{field} must be non-empty")
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


def _bundle_status(
    repo_root: Path, batch: str, cell: str, job_id: str | None = None
) -> dict[str, Any] | None:
    batch_dir = repo_root / RESULTS_ROOT / batch
    candidates = [batch_dir / cell / STATUS, *sorted(batch_dir.glob(f"{cell}.j*/{STATUS}"))]
    for status in candidates:
        if not status.is_file():
            continue
        try:
            data = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if job_id is None or str(data.get("job_id")) == str(job_id):
            return data
    return None


def _status_eligible(status: dict[str, Any]) -> bool:
    """Whether a harvested result may drive a scientific decision.

    ``eligible`` is additive to the original status schema.  Older bundles did
    not carry it, so preserve their old meaning unless they explicitly carry a
    quarantine marker.
    """
    if "eligible" in status:
        return status.get("eligible") is True
    return status.get("complete") is True and status.get("quarantined") is not True


# ------------------------------------------------------------------ the plan

@dataclass(frozen=True)
class Action:
    kind: str  # submit | release | copy_receipt | cancel | harvest | wait | blocked
    batch: str
    detail: str
    cell: str | None = None
    job_id: str | None = None
    manifest: str | None = None
    pilot: bool = False
    keep_held: bool = False
    receipt: str | None = None  # receipt file name within the batch, or a source path for copy_receipt
    job_name: str | None = None
    state: str | None = None
    allow_shared_reservation: bool = False
    release_authorization: str | None = None

    def line(self) -> str:
        where = f"{self.batch}/{self.cell}" if self.cell else self.batch
        job = f" job={self.job_id}" if self.job_id else ""
        pilot = " (pilot)" if self.pilot else ""
        held = " (held)" if self.keep_held else ""
        return f"{self.kind} {where}{job}{pilot}{held}: {self.detail}"


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
            if receipt.get("state") not in ("submitted", "held"):
                failure = receipt.get("failure") or {}
                actions.append(Action(
                    "blocked", batch,
                    f"receipt {name} is in state {receipt.get('state')!r}: "
                    f"{failure.get('reason') or failure.get('stderr') or failure.get('error') or 'see the receipt'}",
                    receipt=name))
            for job in receipt["jobs"]:
                state = states[job["job_id"]]
                if state in TERMINAL_STATES:
                    status = _bundle_status(
                        repo_root, batch, job["cell_id"], job["job_id"]
                    )
                    if status is None:
                        actions.append(Action("harvest", batch, state, cell=job["cell_id"],
                                              job_id=job["job_id"], receipt=name,
                                              job_name=job.get("job_name"), state=state))
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
        formal_receipts = [
            (name, receipt)
            for name, receipt in receipts
            if _receipt_kind(name).startswith("formal")
        ]
        if formal_receipts:
            held_receipts = [
                (name, receipt)
                for name, receipt in formal_receipts
                if receipt.get("state") == "held"
            ]
            if entry["want"] == "submitted" and held_receipts:
                name, _ = held_receipts[-1]
                release_override = entry.get("release_override") or {}
                authorization = None
                if release_override:
                    authorization = (
                        f"authorized_by={release_override['authorized_by']}; "
                        f"authorized_at={release_override['authorized_at']}; "
                        f"reason={release_override['reason']}"
                    )
                actions.append(
                    Action(
                        "release",
                        batch,
                        "ownership and frozen placement must pass before release",
                        receipt=name,
                        allow_shared_reservation=bool(
                            release_override.get("allow_shared_reservation")
                        ),
                        release_authorization=authorization,
                    )
                )
            continue
        if entry["want"] == "held":
            actions.append(
                Action(
                    "submit",
                    batch,
                    "maintain the user-required held pending buffer",
                    manifest=entry["manifest"],
                    keep_held=True,
                )
            )
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
            statuses = [
                (j, _bundle_status(repo_root, batch, j["cell_id"], j["job_id"]))
                for j in pilot["jobs"]
            ]
            if any(s is None or s.get("job_id") != j["job_id"] for j, s in statuses):
                actions.append(Action("wait", batch, "pilot finished; its harvest comes first"))
                continue
            if not all(_status_eligible(s) for _, s in statuses):
                actions.append(Action("blocked", batch,
                                      "pilot did not validate as eligible; the planner decides what happens"))
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
    expected_nodes: set[str] | None = None,
    expected_task: str | None = None,
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
        "eligible": False,
        "issues": [],
        "quarantined": False,
        "quarantine_reasons": [],
        "accuracy": None,
        "stderr": None,
        "judge_flags": [],
        "awm_sha": None,
        "harvested_at": _now(),
        "skipped": skipped,
    }
    if result_dir is not None and result_dir.is_dir():
        for name in ALWAYS_FILES:
            if (result_dir / name).is_file():
                shutil.copy2(result_dir / name, out_dir / name)
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
        status["issues"] = audit_result(result_dir, expected_task=expected_task)
        if expected_nodes:
            provenance = ptb_results._read_json(result_dir / "runtime_provenance.json")
            actual_node = str((provenance.get("slurm") or {}).get("node", ""))
            if not actual_node:
                status["quarantine_reasons"].append(
                    "runtime Slurm node is missing from provenance"
                )
            elif actual_node not in expected_nodes:
                status["quarantine_reasons"].append(
                    f"runtime Slurm node {actual_node} is outside frozen site nodes "
                    f"{','.join(sorted(expected_nodes))}"
                )
        status["complete"] = not status["issues"]
        status["quarantined"] = bool(status["quarantine_reasons"])
        status["eligible"] = status["complete"] and not status["quarantined"]
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
    # Keep the state filter in the controller request, not only our earlier read.
    command = ["scancel", "--ctld", "--state=PENDING", job_id]
    if env.get("POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT") == "1":
        return ["sudo", *command]
    return command


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
    state_after = job_state(job["job_id"])
    if state_after != "CANCELLED":
        raise OpsError(
            f"pending-only cancellation requested for {job['job_id']}, but observed "
            f"{state_after}; cancellation is not confirmed and is not recorded as success"
        )
    record = {"cell_id": cell, "job_id": job["job_id"], "reason": reason,
              "state_before": state, "state_after": state_after,
              "pending_only": True, "at": _now()}
    receipt.setdefault("cancellations", []).append(record)
    receipt.pop("_path", None)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return record


# ------------------------------------------------------------------- apply

def _worktree_dirty(repo_root: Path) -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, text=True,
                          capture_output=True, check=False).stdout.strip()


def _submission_ownership_issue() -> str | None:
    registry = ptb.read_ptb_env().get("POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY", "")
    if not registry:
        return None
    try:
        snapshot = ownership_snapshot(Path(registry))
    except (OSError, slurm_queue.QueueError) as exc:
        return f"cannot inspect ownership registry: {exc}"
    if snapshot.get("ownership_ok"):
        return None
    parts = []
    for key, label in (
        ("unknown_jobs", "unknown job(s)"),
        ("name_mismatches", "name mismatch(es)"),
        ("placement_violations", "placement violation(s)"),
        ("capacity_violations", "capacity violation(s)"),
    ):
        if snapshot.get(key):
            parts.append(f"{len(snapshot[key])} {label}")
    return "OWNERSHIP FAIL: " + (", ".join(parts) or "see gangda-slurm-queue")


def _receipt_expected_nodes(receipt_path: Path) -> set[str]:
    receipt = ptb.load_receipt(receipt_path)
    nodelist = str((receipt.get("site") or {}).get("POST_TRAIN_BENCH_SLURM_NODELIST", ""))
    if not nodelist:
        return set()
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise OpsError(
            f"cannot expand frozen Slurm nodelist {nodelist}: "
            f"{result.stderr.strip() or 'scontrol failed'}"
        )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


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
    runnable_submits = [a for a in submits if not a.keep_held]
    held_submits = [a for a in submits if a.keep_held]
    releases = [a for a in actions if a.kind == "release"]
    others = [a for a in actions if a.kind not in ("submit", "release", "wait", "blocked")]
    if (runnable_submits or releases) and (ownership_issue := _submission_ownership_issue()):
        line = f"blocked submit: {ownership_issue}"
        _log(repo_root, line)
        written.append(line)
        runnable_submits = []
        releases = []
    submits = held_submits + runnable_submits
    if submits and (dirty := _worktree_dirty(repo_root)):
        line = f"blocked submit: the worktree is not clean; commit first\n    {dirty.splitlines()[0]} ..."
        _log(repo_root, line)
        written.append(line)
        submits = []
    submit_outcomes: list[tuple[Action, Path | None, ptb.ExperimentError | None]] = []
    for action in submits:
        manifest = ptb.load_manifest(repo_root / str(action.manifest))
        try:
            receipt_path = submit_batch(
                manifest, pilot=action.pilot, keep_held=action.keep_held
            )
        except ptb.ExperimentError as exc:
            submit_outcomes.append((action, None, exc))
        else:
            submit_outcomes.append((action, Path(receipt_path), None))
    # Do not dirty the source tree until every source-frozen submit has run.
    # Copying the first receipt or writing its blocked.md inside the loop would
    # make the second otherwise-independent submit fail its clean-tree gate.
    for action, receipt_path, error in submit_outcomes:
        if error is not None:
            blocked = repo_root / RESULTS_ROOT / action.batch / "blocked.md"
            blocked.parent.mkdir(parents=True, exist_ok=True)
            blocked.write_text(f"# {action.batch}: submission blocked\n\n{_now()}\n\n```\n{error}\n```\n",
                               encoding="utf-8")
            line = f"blocked submit {action.batch}{' (pilot)' if action.pilot else ''}: {str(error).splitlines()[0]}"
            _log(repo_root, line)
            written.append(line)
            continue
        assert receipt_path is not None
        dst = repo_root / RESULTS_ROOT / action.batch / Path(receipt_path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(receipt_path, dst)
        jobs = ptb.load_receipt(dst)["jobs"]
        line = (f"submit {action.batch}"
                f"{' (pilot)' if action.pilot else ''}"
                f"{' (held)' if action.keep_held else ''}: "
                f"{len(jobs)} job(s) {','.join(j['job_id'] for j in jobs)} -> {dst.relative_to(repo_root)}")
        _log(repo_root, line)
        written.append(line)
    for action in releases:
        source = paths.data_root() / "ptb" / "batches" / action.batch / str(action.receipt)
        tracked = repo_root / RESULTS_ROOT / action.batch / str(action.receipt)
        receipt_path = source if source.is_file() else tracked
        try:
            receipt = release_batch(
                receipt_path,
                allow_shared_reservation=action.allow_shared_reservation,
                authorization=action.release_authorization,
            )
        except ptb.ExperimentError as exc:
            line = f"blocked release {action.batch}: {str(exc).splitlines()[0]}"
        else:
            tracked.parent.mkdir(parents=True, exist_ok=True)
            if receipt_path != tracked:
                shutil.copy2(receipt_path, tracked)
            line = (
                f"release {action.batch}: {len(receipt['jobs'])} held job(s) "
                f"{','.join(str(job['job_id']) for job in receipt['jobs'])}"
            )
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
                line = f"cancel {action.batch}/{action.cell} job={action.job_id} not confirmed: {exc}"
            else:
                line = f"cancel {action.batch}/{action.cell} job={record['job_id']} ({record['state_before']}): {action.detail}"
        elif action.kind == "harvest":
            result_dir = result_for_job(str(action.job_id))
            out_dir = repo_root / RESULTS_ROOT / action.batch / str(action.cell)
            receipt_path = repo_root / RESULTS_ROOT / action.batch / str(action.receipt)
            try:
                expected_task = ptb.receipt_task(ptb.load_receipt(receipt_path), str(action.job_id), str(action.cell))
            except (ptb.ExperimentError, OSError, ValueError):
                expected_task = None  # Preserve artifacts, but completion must fail closed.
            status = harvest_job(result_dir, out_dir, batch=action.batch, cell=str(action.cell),
                                 job_id=str(action.job_id), job_name=action.job_name,
                                 state=action.state, slurm_log_dir=ptb.PTB_ROOT / "logs" / "slurm",
                                 expected_nodes=_receipt_expected_nodes(receipt_path), expected_task=expected_task)
            if status["quarantined"]:
                verdict = f"quarantined ({len(status['quarantine_reasons'])} reason(s))"
            elif status["complete"]:
                verdict = "complete"
            else:
                verdict = f"incomplete ({len(status['issues'])} issue(s))"
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
    matches = []
    for path in (paths.REPO_ROOT / RESULTS_ROOT / args.batch).glob("*.json"):
        try:
            receipt = ptb.load_receipt(path)
            if receipt.get("batch_id") != args.batch:
                continue
            task = ptb.receipt_task(receipt, str(args.job), str(args.cell))
            matches.append((path, task))
        except (ptb.ExperimentError, OSError, ValueError):
            continue
    expected_task = matches[0][1] if len(matches) == 1 else None
    expected_nodes = _receipt_expected_nodes(matches[0][0]) if len(matches) == 1 else None
    status = harvest_job(Path(args.result_dir) if args.result_dir else None, out, batch=args.batch,
                         cell=args.cell, job_id=args.job, job_name=args.job_name, state=args.state,
                         slurm_log_dir=ptb.PTB_ROOT / "logs" / "slurm",
                         expected_task=expected_task, expected_nodes=expected_nodes)
    verdict = ("quarantined" if status["quarantined"] else
               "complete" if status["complete"] else "incomplete")
    print(f"wrote {out / STATUS}: {verdict}, "
          f"{len(status['skipped'])} file(s) listed but not copied")
    return 0 if status["eligible"] else 1


def cancel_cli(args) -> int:
    try:
        record = cancel_job(Path(args.receipt), args.cell, args.reason)
    except (OpsError, ptb.ExperimentError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"cancelled {args.cell} job {record['job_id']} ({record['state_before']}); recorded in {args.receipt}")
    return 0
