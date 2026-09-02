"""Validated launch and audit helpers for committed PostTrainBench batches."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from awm import paths

PTB_ROOT = paths.REPO_ROOT / "third_party" / "PostTrainBench"
SUBMIT = PTB_ROOT / "src" / "commit_utils" / "slurm" / "submit.sh"

APPROVED_BASE_MODELS = (
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
    "google/gemma-3-4b-pt",
)
APPROVED_TASKS = ("gsm8k", "aime2025")
APPROVED_AGENT_SETUPS = (
    ("claude_vertex_max", "claude-opus-5[1m]", "max", 1_000_000),
    ("claude_vertex_xhigh", "claude-opus-5[1m]", "xhigh", 1_000_000),
    ("claude_vertex_high", "claude-opus-5[1m]", "high", 1_000_000),
    ("claude_vertex_max_200k", "claude-opus-5", "max", 200_000),
    # The matching Vertex profile plus a read-only checkout of this repository at
    # AWM_MOUNT and `awm sandbox setup` before the prompt; the cell's `awm` block
    # says what it ships.
    ("claude_vertex_max_awm", "claude-opus-5[1m]", "max", 1_000_000),
    ("claude_vertex_high_awm", "claude-opus-5[1m]", "high", 1_000_000),
)

#: Where an `_awm` scaffold expects the checkout inside the sandbox.
AWM_MOUNT = "/home/ben/awm"
#: The tree whose id names a protocol variant (`awm.protocol_tree` in a manifest cell).
PROTOCOL_TREE_PATH = "skills/exp_protocol"
#: Trees a scientist must never see; no `awm.paths` entry may ship or contain them.
#: Listing a parent (`awm`, `skills`) is rejected too, so a study ships subtrees by name.
AWM_FORBIDDEN_TREES = (
    "skills/exp_protocol_meta",
    "skills/wma",
    "skills/wma_meta",
    "awm/wma",
    "doc",
)
#: What a cell of the experiment-protocol line ships: the CLI, the protocol, nothing else.
EXP_PROTOCOL_SHIP = (
    "awm/__init__.py",
    "awm/cli.py",
    "awm/paths.py",
    "awm/sandbox.py",
    "awm/exp_protocol",
    "skills/exp_protocol",
)


class ExperimentError(ValueError):
    pass


@dataclass(frozen=True)
class Launch:
    cell_id: str
    command: tuple[str, ...]
    environment: dict[str, str]
    #: {sha, paths, dir, digest} of the checkout an `_awm` cell ships; None otherwise.
    checkout: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_git_tracked(repo: Path, relative: str) -> bool:
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def load_manifest(filename: Path) -> dict[str, Any]:
    filename = filename.resolve()
    data = yaml.safe_load(filename.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ExperimentError("manifest root must be a mapping")
    validate_manifest(data)
    data["_path"] = str(filename)
    return data


def _check_awm_path(cell_id: str, path: str) -> None:
    parts = path.split("/")
    if not path or path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ExperimentError(
            f"cell {cell_id}: awm.paths entry {path!r} must be a relative path inside the repository"
        )
    for tree in AWM_FORBIDDEN_TREES:
        if path == tree or tree.startswith(path + "/") or path.startswith(tree + "/"):
            raise ExperimentError(
                f"cell {cell_id}: awm.paths entry {path!r} would ship {tree}, "
                "which a scientist must not see"
            )


def _ships_protocol(shipped: list[str] | tuple[str, ...]) -> bool:
    return any(
        path == PROTOCOL_TREE_PATH or path.startswith(PROTOCOL_TREE_PATH + "/")
        for path in shipped
    )


def _validate_awm_block(cell_id: str, agent: Any, block: Any) -> None:
    """An `_awm` scaffold needs to be told what to ship; any other scaffold would ignore it."""
    wants = str(agent).endswith("_awm")
    if block is None:
        if wants:
            raise ExperimentError(
                f"cell {cell_id} uses the {agent} scaffold and must declare an awm block "
                "(sha, paths, setup)"
            )
        return
    if not wants:
        raise ExperimentError(
            f"cell {cell_id} declares an awm block but the {agent} scaffold would ignore it"
        )
    if not isinstance(block, dict):
        raise ExperimentError(f"cell {cell_id}: awm must be a mapping with sha, paths, setup")
    sha = str(block.get("sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ExperimentError(
            f"cell {cell_id}: awm.sha must be a full commit id (40 hex characters), not {sha!r}"
        )
    shipped = block.get("paths")
    if (
        not isinstance(shipped, list)
        or not shipped
        or not all(isinstance(path, str) for path in shipped)
    ):
        raise ExperimentError(
            f"cell {cell_id}: awm.paths must be a non-empty list of repository paths"
        )
    for path in shipped:
        _check_awm_path(cell_id, path)
    setup = block.get("setup")
    if not isinstance(setup, str) or not setup.strip() or "\n" in setup:
        raise ExperimentError(
            f"cell {cell_id}: awm.setup must be one line of arguments for `awm sandbox setup`"
        )
    tree = block.get("protocol_tree")
    if not _ships_protocol(shipped) and tree is not None:
        raise ExperimentError(
            f"cell {cell_id}: awm.protocol_tree was declared but awm.paths does not ship "
            f"{PROTOCOL_TREE_PATH}"
        )
    if tree is not None and not re.fullmatch(r"[0-9a-f]{40}", str(tree)):
        raise ExperimentError(
            f"cell {cell_id}: awm.protocol_tree must be the full git tree id of "
            f"skills/exp_protocol at awm.sha (git rev-parse <sha>:skills/exp_protocol), not {tree!r}"
        )


def _git_has_commit(sha: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=paths.REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _awm_issues(cell_id: str, block: dict[str, Any]) -> list[str]:
    """What `check` can say about a cell's checkout before anything is materialised."""
    sha = str(block.get("sha", ""))
    if not _git_has_commit(sha):
        return [f"cell {cell_id}: awm commit {sha} is not in this repository (git fetch origin first)"]
    issues = []
    for path in block.get("paths") or []:
        listed = subprocess.run(
            ["git", "ls-tree", "--name-only", sha, "--", path],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not listed:
            issues.append(f"cell {cell_id}: awm path {path!r} does not exist at commit {sha[:12]}")
    declared = block.get("protocol_tree")
    if _ships_protocol(block.get("paths") or []) and not declared:
        issues.append(
            f"cell {cell_id}: awm.protocol_tree must declare the full git tree id of "
            f"{PROTOCOL_TREE_PATH} at awm.sha"
        )
    if declared:
        actual = protocol_tree_at(sha)
        if actual != declared:
            issues.append(
                f"cell {cell_id}: skills/exp_protocol at commit {sha[:12]} is tree "
                f"{(actual or 'absent')[:12]}, not the declared protocol_tree {str(declared)[:12]}"
            )
    return issues


def protocol_tree_at(sha: str) -> str | None:
    """The git tree id of ``skills/exp_protocol`` at ``sha``: the protocol variant's identity.

    A manifest names the commit it ships (which must carry the CLI and the
    setup step) and, separately, the protocol tree it means to test; two
    commits with the same tree are the same variant, and a commit whose tree
    differs from the declared one is a different variant, whatever its
    message says.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{sha}:{PROTOCOL_TREE_PATH}"],
        cwd=paths.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(file).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def materialize_awm_checkout(sha: str, shipped: list[str]) -> dict[str, Any]:
    """``git archive`` the given paths of ``sha`` onto the data volume, once per (sha, paths).

    The directory is what the scaffold sees at AWM_MOUNT, read-only. A marker
    file records the commit, the paths and a content digest; a directory whose
    marker matches is reused, so dry-run, submit and a retry all bind the same
    bytes. The forbidden trees are checked again on the extracted files.
    """
    for path in shipped:
        _check_awm_path("<checkout>", path)
    root = paths.data_root() / "ptb" / "awm-checkouts"
    key = hashlib.sha256("\n".join(shipped).encode()).hexdigest()[:12]
    target = root / f"{sha}-{key}"
    marker_name = ".awm-checkout.json"
    expected_protocol_tree = protocol_tree_at(sha) if _ships_protocol(shipped) else None

    def _existing() -> dict[str, Any] | None:
        marker = target / marker_name
        if not marker.is_file():
            return None
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not (
            info.get("sha") == sha and info.get("paths") == list(shipped) and info.get("complete")
        ):
            return None
        if "protocol_tree" not in info and expected_protocol_tree is not None:
            # The bytes are fixed by (sha, paths); only the marker is behind (an older
            # launcher wrote it without the tree). Upgrade the marker in place: a running
            # cell may have this directory bind-mounted, and replacing it under that
            # mount leaves the scientist with a stale file handle (pilot 90462, 2026-09-02).
            info["protocol_tree"] = expected_protocol_tree
            info["marker_upgraded_at"] = datetime.now(timezone.utc).isoformat()
            marker.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        elif info.get("protocol_tree") != expected_protocol_tree:
            raise ExperimentError(
                f"checkout {target} is complete but its marker has protocol_tree "
                f"{info.get('protocol_tree')!r}, expected {expected_protocol_tree!r}; "
                "refusing to rewrite a corrupt marker for a directory a running cell may be using"
            )
        return {
            "sha": sha,
            "paths": list(shipped),
            "dir": str(target),
            "digest": info["digest"],
            "protocol_tree": info.get("protocol_tree"),
        }

    if (found := _existing()) is not None:
        return found
    if target.exists():
        # Only a half-written directory is ever removed: one without a complete marker.
        # A complete checkout is never deleted here, whatever its marker says, because a
        # cell may be running on it; (sha, paths) name the directory, so a complete marker
        # that disagrees with them is corruption to investigate, not to overwrite.
        marker = target / marker_name
        if marker.is_file():
            try:
                info = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ExperimentError(
                    f"checkout {target} has an unreadable marker; refusing to replace a "
                    "directory a running cell may be using"
                ) from exc
            if info.get("complete"):
                raise ExperimentError(
                    f"checkout {target} is complete but its marker names a different commit or "
                    "paths; refusing to replace a directory a running cell may be using"
                )
        shutil.rmtree(target)  # half-written; never bound into a sandbox
    if not _git_has_commit(sha):
        raise ExperimentError(f"awm commit {sha} is not in this repository; git fetch origin first")
    archive = subprocess.run(
        ["git", "archive", "--format=tar", sha, "--", *shipped],
        cwd=paths.REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if archive.returncode:
        raise ExperimentError(
            f"git archive {sha[:12]} failed: {archive.stderr.decode(errors='replace').strip()}"
        )
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{sha[:12]}-", dir=root))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(staging, filter="data")
            else:  # Python < 3.12
                tar.extractall(staging)
        for tree in AWM_FORBIDDEN_TREES:
            if (staging / tree).exists():
                raise ExperimentError(f"archive of {sha[:12]} contains {tree}; refusing to ship it")
        digest = _tree_digest(staging)
        protocol_tree = expected_protocol_tree
        marker = {
            "sha": sha,
            "paths": list(shipped),
            "digest": digest,
            "protocol_tree": protocol_tree,
            "complete": True,
            "materialized_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / marker_name).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        try:
            os.rename(staging, target)
        except OSError:
            if (found := _existing()) is not None:  # someone else won the race with the same bytes
                shutil.rmtree(staging, ignore_errors=True)
                return found
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "sha": sha,
        "paths": list(shipped),
        "dir": str(target),
        "digest": digest,
        "protocol_tree": protocol_tree,
    }


def _batch_tasks(contract: dict[str, Any]) -> tuple[str, ...]:
    if "tasks" in contract:
        if contract.get("task") is not None:
            raise ExperimentError("contract.task must be absent when contract.tasks is given")
        tasks = tuple(contract.get("tasks") or ())
    else:
        tasks = (contract.get("task"),)
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in APPROVED_TASKS for task in tasks)
    ):
        raise ExperimentError(
            f"the batch's tasks must be a non-empty subset of {list(APPROVED_TASKS)!r}"
        )
    return tasks


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_replication(
    replication: Any, cells: list[dict[str, Any]], settings: list[tuple[Any, ...]]
) -> None:
    """``contract.replication`` describes the batch's own shape; the cells must match it."""
    if not isinstance(replication, dict):
        raise ExperimentError("contract.replication must be a mapping")
    n_settings = replication.get("settings")
    repeats = replication.get("repeats")
    if not _positive_int(n_settings) or not _positive_int(repeats):
        raise ExperimentError(
            "contract.replication must give positive integer settings and repeats"
        )
    counts = Counter(settings)
    if len(counts) != n_settings or set(counts.values()) != {repeats}:
        raise ExperimentError(
            f"replication batch must contain {n_settings} unique settings "
            f"repeated exactly {repeats} times"
        )
    per_task = replication.get("settings_per_task")
    if per_task is not None:
        by_task = Counter(setting[0] for setting in counts)
        if any(count != per_task for count in by_task.values()):
            raise ExperimentError(
                f"replication batch must select exactly {per_task} settings per task"
            )
    expected_replicates = set(range(1, repeats + 1))
    for setting in counts:
        replicates = [
            cell.get("replicate")
            for cell, cell_setting in zip(cells, settings, strict=True)
            if cell_setting == setting
        ]
        if set(replicates) != expected_replicates or len(replicates) != repeats:
            raise ExperimentError(
                f"each replicated setting must carry replicates 1..{repeats}, once each"
            )


def validate_manifest(data: dict[str, Any]) -> None:
    """A batch is any non-empty set of cells drawn from the approved setting space.

    Each cell is checked on its own: an approved task, an approved agent setup,
    a base model the contract pins. The launcher does not insist on a shape for
    the whole. The full 4x4 matrix of the first batch and the 16x2 replication of
    the second were decisions about those batches; a protocol-iteration round of
    two repeats of one setting goes through the same launcher, receipts, and
    registry. The contract constants (budget, containers, provider route, CLI
    version, judge profile) stay frozen so results remain comparable.
    """
    if data.get("schema_version") != 1:
        raise ExperimentError("schema_version must be 1")
    ownership = data.get("ownership") or {}
    branch = str(ownership.get("branch", ""))
    spec = str(ownership.get("spec", ""))
    if not branch or branch != _slug(branch):
        raise ExperimentError("ownership.branch must be a non-empty Slurm-safe branch name")
    if not spec.startswith("doc/spec/") or not (paths.REPO_ROOT / spec).is_file():
        raise ExperimentError("ownership.spec must name an existing committed spec under doc/spec")
    contract = data.get("contract") or {}
    tasks = _batch_tasks(contract)
    expected = {
        "agent_budget_hours": 10,
        "gpus": 1,
        "cpus": 16,
        "memory": "128G",
        "scratch_gb": 400,
        "context_windows": [200_000, 1_000_000],
        "agent_cli_version": "2.1.219",
        "judge_profile": "official",
        "research_judge_profile": "claude-opus-5[1m]-xhigh",
        "require_complete": True,
        "cli_auto_update": False,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ExperimentError(f"contract.{key} must be {value!r}")
    if not _positive_int(contract.get("run_index")):
        raise ExperimentError("contract.run_index must be a positive integer")
    if contract.get("agent_auth") != {
        "provider": "vertex",
        "project": "sercan-v1",
        "region": "global",
    }:
        raise ExperimentError("contract.agent_auth must freeze the approved Vertex route")
    base_models = contract.get("base_models") or {}
    if (
        not isinstance(base_models, dict)
        or not base_models
        or any(model not in APPROVED_BASE_MODELS for model in base_models)
    ):
        raise ExperimentError(
            "contract.base_models may only pin the approved starting models, and must pin at least one"
        )
    for model, metadata in base_models.items():
        if not isinstance(metadata, dict) or not re.fullmatch(
            r"[0-9a-f]{40}", str(metadata.get("revision", ""))
        ):
            raise ExperimentError(f"contract.base_models[{model!r}] must pin a commit revision")
    container = contract.get("container") or {}
    if container.get("name") != "opus_5" or not re.fullmatch(
        r"[0-9a-f]{64}", str(container.get("sha256", ""))
    ):
        raise ExperimentError("contract.container must pin opus_5 and a SHA-256 digest")
    evaluation_container = contract.get("evaluation_container") or {}
    if evaluation_container.get("name") != "vllm_debug.sif" or not re.fullmatch(
        r"[0-9a-f]{64}", str(evaluation_container.get("sha256", ""))
    ):
        raise ExperimentError(
            "contract.evaluation_container must pin vllm_debug.sif and a SHA-256 digest"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("official_judge_container_sha256", ""))):
        raise ExperimentError("contract must pin the official judge container SHA-256")

    cells = data.get("cells")
    if not isinstance(cells, list) or not cells or not all(isinstance(c, dict) for c in cells):
        raise ExperimentError("the batch must contain at least one cell")
    ids = [str(cell.get("id")) for cell in cells]
    if len(set(ids)) != len(ids):
        raise ExperimentError("cell ids must be unique")
    approved_setups = set(APPROVED_AGENT_SETUPS)
    settings: list[tuple[Any, ...]] = []
    for cell, cell_id in zip(cells, ids, strict=True):
        task = _cell_task(contract, cell)
        if task not in tasks:
            raise ExperimentError(
                f"cell {cell_id} names task {task!r}, not one of the batch's tasks {list(tasks)!r}"
            )
        setup = (
            cell.get("agent"),
            cell.get("agent_model"),
            cell.get("effort"),
            cell.get("context_tokens"),
        )
        if setup not in approved_setups:
            raise ExperimentError(f"cell {cell_id} is not an approved agent setup: {setup!r}")
        base = cell.get("base_model")
        if base not in base_models:
            raise ExperimentError(
                f"cell {cell_id} uses base model {base!r}, which contract.base_models does not pin"
            )
        _validate_awm_block(cell_id, cell.get("agent"), cell.get("awm"))
        settings.append((task, *setup, base))
    replication = contract.get("replication")
    if replication is not None:
        _validate_replication(replication, cells, settings)
    pilot = data.get("pilot")
    if pilot is not None:
        pilot_cells = _pilot_cell_ids(data)
        if (
            not pilot_cells
            or len(set(pilot_cells)) != len(pilot_cells)
            or any(cell_id not in ids for cell_id in pilot_cells)
        ):
            raise ExperimentError("pilot must name existing, distinct cells of this batch")
        if pilot.get("agent_budget_hours") != 1:
            raise ExperimentError("pilot.agent_budget_hours must be 1")
    records = data.get("context_validation") or {}
    for _, _, model, effort, _, _ in settings:
        profile = f"{model}:{effort}"
        if profile not in records:
            raise ExperimentError(f"missing context-validation path for {profile}")

def _cell(data: dict[str, Any], cell_id: str) -> dict[str, Any]:
    try:
        return next(cell for cell in data["cells"] if cell["id"] == cell_id)
    except StopIteration as exc:
        raise ExperimentError(f"unknown cell: {cell_id}") from exc


def _cell_task(contract: dict[str, Any], cell: dict[str, Any]) -> str:
    task = cell.get("task", contract.get("task"))
    if not isinstance(task, str) or not task:
        raise ExperimentError(f"cell {cell.get('id', '<unknown>')} must identify a task")
    return task


def _pilot_cell_ids(data: dict[str, Any]) -> list[str]:
    pilot = data.get("pilot") or {}
    if "cells" in pilot:
        cells = pilot.get("cells")
        if not isinstance(cells, list) or not all(isinstance(cell, str) for cell in cells):
            raise ExperimentError("pilot.cells must be a list of cell ids")
        return list(cells)
    cell = pilot.get("cell")
    return [cell] if isinstance(cell, str) else []


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", value)


def _run_identity(data: dict[str, Any], cell_id: str, purpose: str) -> tuple[str, str, str]:
    branch = data["ownership"]["branch"]
    batch = data["batch_id"]
    run_index = data["contract"]["run_index"]
    job_name = _slug(f"{branch}.ptb.{batch}.{cell_id}.{purpose}.r{run_index}")
    if len(job_name) > 128:
        raise ExperimentError(f"Slurm job name exceeds 128 characters: {job_name}")
    experiment_name = _slug(f"_{branch}_{batch}_{cell_id}_{purpose}_r{run_index}")
    return branch, job_name, experiment_name


def _command_option(launch: Launch, option: str) -> str:
    try:
        return launch.command[launch.command.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise ExperimentError(f"{launch.cell_id} launch is missing {option}") from exc


def build_launches(
    data: dict[str, Any],
    *,
    pilot: bool = False,
    cell_ids: list[str] | None = None,
    hold: bool = False,
    purpose: str | None = None,
) -> list[Launch]:
    contract = data["contract"]
    if pilot and not _pilot_cell_ids(data):
        raise ExperimentError("the manifest declares no pilot cell")
    selected = (
        [_cell(data, cell_id) for cell_id in _pilot_cell_ids(data)]
        if pilot
        else list(data["cells"])
    )
    if cell_ids:
        selected = [_cell(data, cell_id) for cell_id in cell_ids]
    hours = data["pilot"]["agent_budget_hours"] if pilot else contract["agent_budget_hours"]
    run_purpose = purpose or (f"pilot-{hours}h" if pilot else "formal")
    launches = []
    for cell in selected:
        branch, job_name, experiment_name = _run_identity(data, cell["id"], run_purpose)
        command = (
            "bash",
            str(SUBMIT),
            "--eval",
            _cell_task(contract, cell),
            "--agent",
            cell["agent"],
            "--model",
            cell["base_model"],
            "--hours",
            str(hours),
            "--agent-config",
            cell["agent_model"],
            "--gpus",
            str(contract["gpus"]),
            "--run-branch",
            branch,
            "--job-name",
            job_name,
            "--experiment-name",
            experiment_name,
            "--judge-profile",
            "official",
        )
        if hold:
            command = (*command, "--hold")
        context_profile = f"{cell['agent_model']}:{cell['effort']}"
        record = (paths.REPO_ROOT / data["context_validation"][context_profile]).resolve()
        environment = {
            # The submit shell can carry an unrelated ambient HF_HOME.  Pin the
            # experiment's canonical shared cache from PTB's .env so
            # set_env_vars.sh cannot preserve the wrong inherited value.
            "HF_HOME": read_ptb_env().get("HF_HOME", ""),
            "POST_TRAIN_BENCH_BASE_MODEL_REVISION": contract["base_models"][cell["base_model"]][
                "revision"
            ],
            "POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD": str(record),
            "POST_TRAIN_BENCH_REQUIRE_CONTEXT_VALIDATION": "1",
            "POST_TRAIN_BENCH_REQUIRE_COMPLETE": "1",
            "POST_TRAIN_BENCH_JUDGE_PROFILE": "official",
            "POST_TRAIN_BENCH_SLURM_GPU_MODE": "gres",
            "POST_TRAIN_BENCH_SKIP_CLI_UPDATE": "1",
            "POST_TRAIN_BENCH_CONTAINER_SHA256": contract["container"]["sha256"],
            "POST_TRAIN_BENCH_EVALUATION_CONTAINER_SHA256": contract["evaluation_container"][
                "sha256"
            ],
            "POST_TRAIN_BENCH_OFFICIAL_JUDGE_CONTAINER_SHA256": contract[
                "official_judge_container_sha256"
            ],
            "POST_TRAIN_BENCH_BATCH_ID": data["batch_id"],
            "POST_TRAIN_BENCH_CELL_ID": cell["id"],
            "POST_TRAIN_BENCH_RUN_PURPOSE": run_purpose,
            "POST_TRAIN_BENCH_SPEC_PATH": data["ownership"]["spec"],
            "POST_TRAIN_BENCH_EXPECTED_CONTEXT_TOKENS": str(cell["context_tokens"]),
        }
        if record.is_file():
            environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256"] = _sha256(record)
        checkout = None
        if cell.get("awm"):
            checkout = materialize_awm_checkout(cell["awm"]["sha"], cell["awm"]["paths"])
            # The receipt names the variant fully: the tree shipped and the setup it runs.
            checkout = {**checkout, "setup": cell["awm"]["setup"]}
            # Read on the host by run_task.sh: one read-only bind for the agent sandbox.
            environment["POST_TRAIN_BENCH_EXTRA_BINDS"] = f"{checkout['dir']}:{AWM_MOUNT}:ro"
            # Forwarded into the sandbox by the scaffold's env_passthrough.txt.
            environment["AWM_SANDBOX_SETUP"] = cell["awm"]["setup"]
            environment["AWM_CHECKOUT_SHA"] = cell["awm"]["sha"]
        launches.append(Launch(cell["id"], command, environment, checkout))
    if len({launch.cell_id for launch in launches}) != len(launches):
        raise ExperimentError("launch result ids are not unique")
    return launches


def local_issues(
    data: dict[str, Any],
    *,
    require_context: bool = True,
    cell_ids: list[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    contract = data["contract"]
    selected_cells = (
        [_cell(data, cell_id) for cell_id in cell_ids] if cell_ids else list(data["cells"])
    )
    selected_tasks = {_cell_task(contract, cell) for cell in selected_cells}
    for task in selected_tasks:
        required_assets = [
            f"src/eval/tasks/{task}/evaluate.py",
            f"src/eval/tasks/{task}/test_data.json",
            f"src/eval/tasks/{task}/info.json",
        ]
        for relative in required_assets:
            if not (PTB_ROOT / relative).is_file():
                issues.append(f"missing PTB task asset: {relative}")
            elif not _is_git_tracked(PTB_ROOT, relative):
                issues.append(f"untracked PTB task asset cannot enter frozen source: {relative}")
    for cell in selected_cells:
        if cell.get("awm"):
            issues.extend(_awm_issues(str(cell["id"]), cell["awm"]))
        for name in ("solve.sh", "api_keys.json", "profile.env"):
            relative = f"agents/{cell['agent']}/{name}"
            if not (PTB_ROOT / relative).is_file():
                issues.append(f"missing agent asset: {relative}")
            elif not _is_git_tracked(PTB_ROOT, relative):
                issues.append(f"untracked agent asset cannot enter frozen source: {relative}")
    env = read_ptb_env()
    containers = Path(env.get("POST_TRAIN_BENCH_CONTAINERS_DIR", PTB_ROOT / "containers"))
    hf_home = Path(env.get("HF_HOME", ""))
    selected_base_models = {cell["base_model"] for cell in selected_cells}
    for model in selected_base_models:
        metadata = contract["base_models"][model]
        cache_name = "models--" + model.replace("/", "--")
        snapshot = hf_home / "hub" / cache_name / "snapshots" / metadata["revision"]
        issues.extend(_base_model_snapshot_issues(model, metadata["revision"], snapshot))
    expected_images = {
        f"{contract['container']['name']}.sif": contract["container"]["sha256"],
        contract["evaluation_container"]["name"]: contract["evaluation_container"]["sha256"],
        contract["official_judge_container"]: contract["official_judge_container_sha256"],
    }
    for image, expected_digest in expected_images.items():
        path = containers / image
        if not path.is_file():
            issues.append(f"missing container: {path}")
            continue
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            issues.append(
                f"container digest mismatch: {path} actual={actual_digest} expected={expected_digest}"
            )
    if require_context:
        selected_profiles = {
            f"{cell['agent_model']}:{cell['effort']}": int(cell["context_tokens"])
            for cell in selected_cells
        }
        for profile, expected_context in selected_profiles.items():
            relative = data["context_validation"][profile]
            model, effort = profile.rsplit(":", 1)
            path = paths.REPO_ROOT / relative
            if not path.is_file():
                issues.append(
                    f"missing {expected_context}-token provider validation for {profile}: {path}"
                )
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if (
                    record.get("requested_model") != model
                    or record.get("provider") != "vertex"
                    or record.get("project") != contract["agent_auth"]["project"]
                    or record.get("region") != contract["agent_auth"]["region"]
                    or record.get("cli_version") != contract["agent_cli_version"]
                    or record.get("container_sha256") != contract["container"]["sha256"]
                    or record.get("effort") != effort
                    or record.get("verified") is not True
                    or int(record.get("requested_context_tokens", 0)) != expected_context
                    or int(record.get("resolved_context_tokens", 0)) != expected_context
                ):
                    issues.append(
                        f"invalid {expected_context}-token provider validation for {profile}: {path}"
                    )
            except (OSError, ValueError, TypeError) as exc:
                issues.append(
                    f"unreadable {expected_context}-token provider validation for {profile}: {exc}"
                )
    return issues


def _base_model_snapshot_issues(model: str, revision: str, snapshot: Path) -> list[str]:
    config = snapshot / "config.json"
    index = snapshot / "model.safetensors.index.json"
    monolithic_weights = snapshot / "model.safetensors"
    if not config.is_file() or not (index.is_file() or monolithic_weights.is_file()):
        return [f"missing pinned base-model snapshot: {model}@{revision} ({snapshot})"]
    if monolithic_weights.is_file() and not index.is_file():
        if monolithic_weights.stat().st_size == 0:
            return [
                (
                    f"incomplete pinned base-model snapshot: {model}@{revision} "
                    "missing ['model.safetensors']"
                )
            ]
        return []
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map is missing or empty")
        missing_weights: list[str] = []
        for filename in weight_map.values():
            if not isinstance(filename, str):
                missing_weights.append(repr(filename))
            elif (
                Path(filename).is_absolute()
                or ".." in Path(filename).parts
                or not (snapshot / filename).is_file()
                or (snapshot / filename).stat().st_size == 0
            ):
                missing_weights.append(filename)
        missing_weights = sorted(set(missing_weights))
        if missing_weights:
            return [
                (
                    f"incomplete pinned base-model snapshot: {model}@{revision} "
                    f"missing {missing_weights}"
                )
            ]
    except (KeyError, OSError, ValueError, TypeError) as exc:
        return [f"invalid base-model snapshot index for {model}: {exc}"]
    return []


def read_ptb_env() -> dict[str, str]:
    values: dict[str, str] = {}
    filename = PTB_ROOT / ".env"
    if not filename.is_file():
        return values
    for raw in filename.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def site_issues() -> list[str]:
    env = read_ptb_env()
    partition = env.get("POST_TRAIN_BENCH_SLURM_PARTITION", "")
    nodelist = env.get("POST_TRAIN_BENCH_SLURM_NODELIST", "")
    subqueue = env.get("POST_TRAIN_BENCH_SLURM_SUBQUEUE", "")
    issues: list[str] = []
    if env.get("POST_TRAIN_BENCH_SLURM_GPU_MODE") != "gres":
        issues.append("site must use POST_TRAIN_BENCH_SLURM_GPU_MODE=gres")
    submit_as_root = env.get("POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT", "0")
    if submit_as_root not in ("0", "1"):
        issues.append("POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT must be 0 or 1")
    elif submit_as_root == "1":
        run_as_user = env.get("POST_TRAIN_BENCH_SLURM_RUN_AS_USER", "")
        current_user = pwd.getpwuid(os.getuid()).pw_name
        if not run_as_user:
            issues.append(
                "root Slurm submission requires POST_TRAIN_BENCH_SLURM_RUN_AS_USER"
            )
        elif run_as_user != current_user or run_as_user == "root":
            issues.append(
                "root Slurm submission must drop back to the invoking non-root user"
            )
        sudo = subprocess.run(
            ["sudo", "-n", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if sudo.returncode:
            issues.append("root Slurm submission requires non-interactive sudo")
    try:
        part = subprocess.check_output(
            ["scontrol", "show", "partition", partition, "-o"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return issues + [f"cannot inspect Slurm partition {partition}: {exc}"]
    if "OverSubscribe=NO" not in part:
        issues.append(f"partition {partition} is not OverSubscribe=NO")
    try:
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        ).split()
    except (OSError, subprocess.CalledProcessError) as exc:
        return issues + [f"cannot expand Slurm nodelist {nodelist}: {exc}"]
    if subqueue:
        registry_path = env.get("POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY", "")
        if not registry_path:
            issues.append(f"subqueue {subqueue} requires an ownership registry path")
        else:
            try:
                from awm import slurm_queue

                registry = slurm_queue.load_registry(Path(registry_path))
                config = slurm_queue.subqueue_config(registry, subqueue)
            except (OSError, slurm_queue.QueueError) as exc:
                issues.append(f"cannot validate Slurm subqueue {subqueue}: {exc}")
            else:
                expected_nodes = list(config["nodes"])
                if set(nodes) != set(expected_nodes):
                    issues.append(
                        f"site nodelist for subqueue {subqueue} must be "
                        f"{','.join(expected_nodes)}; got {','.join(nodes)}"
                    )
                branch = current_top_branch()
                if branch not in config["branches"]:
                    issues.append(
                        f"current branch {branch} does not belong to Slurm subqueue {subqueue}"
                    )
    elif len(nodes) != 4:
        issues.append(f"legacy site nodelist must resolve to four nodes, got {len(nodes)}")
    for node in nodes:
        line = subprocess.check_output(["scontrol", "show", "node", node, "-o"], text=True)
        cfg = re.search(r"\bCfgTRES=([^ ]+)", line)
        if not cfg or "gres/gpu=8" not in cfg.group(1):
            issues.append(f"{node} does not advertise consumable gres/gpu=8 in CfgTRES")
    return issues


def _release_command(ptb_env: dict[str, str], job_ids: str) -> list[str]:
    command = ["scontrol", "release", job_ids]
    if ptb_env.get("POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT") == "1":
        command.insert(0, "sudo")
    return command


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def current_top_branch() -> str:
    return _git(paths.REPO_ROOT, "branch", "--show-current")


def _worktree_status(repo: Path, *, ignore_submodules: bool = False) -> str:
    status_args = ["status", "--porcelain", "--untracked-files=no"]
    if ignore_submodules:
        status_args.append("--ignore-submodules=all")
    tracked = _git(repo, *status_args)
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
    return "\n".join(part for part in (tracked, untracked) if part)


def source_snapshot() -> dict[str, Any]:

    return {
        "top_branch": current_top_branch(),
        "top_commit": _git(paths.REPO_ROOT, "rev-parse", "HEAD"),
        "ptb_commit": _git(PTB_ROOT, "rev-parse", "HEAD"),
        "top_status": _worktree_status(paths.REPO_ROOT, ignore_submodules=True),
        "ptb_status": _worktree_status(PTB_ROOT),
    }


def assert_source_ownership(data: dict[str, Any], snapshot: dict[str, Any]) -> None:
    expected = data["ownership"]["branch"]
    actual = snapshot.get("top_branch", "")
    if not actual:
        raise ExperimentError("Slurm submission requires a named top-level Git branch")
    if actual != expected:
        raise ExperimentError(
            f"current branch {actual!r} does not match manifest ownership.branch {expected!r}"
        )


def dry_run(data: dict[str, Any], *, pilot: bool = False) -> list[tuple[str, str]]:
    assert_source_ownership(data, {"top_branch": current_top_branch()})
    outputs = []
    for launch in build_launches(data, pilot=pilot, hold=not pilot):
        env = os.environ | launch.environment
        result = subprocess.run(
            [*launch.command, "--dry-run"],
            cwd=PTB_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ExperimentError(result.stderr.strip() or f"dry-run failed for {launch.cell_id}")
        command = result.stdout.strip()
        for required in ("--gres=gpu:1", "--cpus-per-task=16", "--mem=128G"):
            if required not in command:
                raise ExperimentError(f"{launch.cell_id} dry-run missing {required}: {command}")
        required_prefix = f"--job-name={data['ownership']['branch']}."
        if required_prefix not in command:
            raise ExperimentError(
                f"{launch.cell_id} dry-run job name lacks branch prefix: {command}"
            )
        if "--exclusive" in command:
            raise ExperimentError(f"{launch.cell_id} dry-run unexpectedly reserves a whole node")
        outputs.append((launch.cell_id, command))
    return outputs


def submit_context_smokes(data: dict[str, Any], cell_ids: list[str]) -> list[dict[str, str]]:
    assert_source_ownership(data, {"top_branch": current_top_branch()})
    issues = local_issues(data, require_context=False, cell_ids=cell_ids) + site_issues()
    if issues:
        raise ExperimentError("context-smoke gates failed:\n- " + "\n- ".join(issues))
    jobs = []
    for launch in build_launches(data, cell_ids=cell_ids, purpose="context-smoke"):
        env = os.environ | launch.environment
        env["POST_TRAIN_BENCH_REQUIRE_CONTEXT_VALIDATION"] = "0"
        result = subprocess.run(
            [*launch.command, "--runtime-smoke", "--walltime", "00:15:00"],
            cwd=PTB_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ExperimentError(
                result.stderr.strip() or f"context smoke failed for {launch.cell_id}"
            )
        match = re.search(r"Submitted Slurm job (\d+)", result.stdout)
        if not match:
            raise ExperimentError(f"could not parse context-smoke job id: {result.stdout}")
        jobs.append(
            {
                "cell_id": launch.cell_id,
                "job_id": match.group(1),
                "job_name": _command_option(launch, "--job-name"),
            }
        )
    return jobs


def submit(
    data: dict[str, Any],
    *,
    pilot: bool = False,
    cell_ids: list[str] | None = None,
    keep_held: bool = False,
) -> Path:
    if pilot and cell_ids:
        raise ExperimentError("pilot and explicit retry cells are mutually exclusive")
    if pilot and keep_held:
        raise ExperimentError("a pilot cannot be kept held; held buffers are formal cells")
    snapshot = source_snapshot()
    assert_source_ownership(data, snapshot)
    selected_cell_ids = _pilot_cell_ids(data) if pilot else cell_ids
    if selected_cell_ids:
        if len(set(selected_cell_ids)) != len(selected_cell_ids):
            raise ExperimentError("retry cell ids must be unique")
        for cell_id in selected_cell_ids:
            _cell(data, cell_id)
    issues = local_issues(data, cell_ids=selected_cell_ids) + site_issues()
    if issues:
        raise ExperimentError("submission gates failed:\n- " + "\n- ".join(issues))
    if snapshot["top_status"] or snapshot["ptb_status"]:
        raise ExperimentError("formal source freeze requires clean top-level and PTB worktrees")
    dry_run(data, pilot=pilot)
    submitted_at = datetime.now(timezone.utc).isoformat()
    out_dir = paths.ensure(paths.data_root() / "ptb" / "batches" / data["batch_id"])
    if pilot:
        kind = "pilot"
        run_purpose = None
    elif selected_cell_ids:
        previous_retries = sorted(out_dir.glob("formal-retry*-*.json"))
        retry_index = len(previous_retries) + 1
        kind = f"formal-retry{retry_index}"
        run_purpose = kind
    else:
        kind = "formal"
        run_purpose = None
    previous = sorted(out_dir.glob(f"{kind}-*.json"))
    if previous:
        raise ExperimentError(f"refusing duplicate {kind} submission: {previous[-1]}")
    output = out_dir / f"{kind}-{submitted_at.replace(':', '')}.json"
    ptb_env = read_ptb_env()
    launches = build_launches(
        data,
        pilot=pilot,
        cell_ids=selected_cell_ids if not pilot else None,
        hold=not pilot,
        purpose=run_purpose,
    )
    if any(
        "POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256" not in launch.environment
        for launch in launches
    ):
        raise ExperimentError("submission requires immutable context-validation digests")
    receipt = {
        "schema_version": 1,
        "batch_id": data["batch_id"],
        "kind": kind,
        "state": "submitting",
        "manifest": data["_path"],
        "ownership": data["ownership"],
        "subqueue": ptb_env.get("POST_TRAIN_BENCH_SLURM_SUBQUEUE", ""),
        "submitted_at": submitted_at,
        "source": snapshot,
        "contract": data["contract"],
        "cells": [_cell(data, launch.cell_id) for launch in launches],
        "context_validation": {
            launch.cell_id: {
                "path": launch.environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD"],
                "sha256": launch.environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256"],
            }
            for launch in launches
        },
        "awm_checkouts": {
            launch.cell_id: launch.checkout for launch in launches if launch.checkout
        },
        "site": {
            key: ptb_env.get(key, "")
            for key in (
                "POST_TRAIN_BENCH_SLURM_PARTITION",
                "POST_TRAIN_BENCH_SLURM_NODELIST",
                "POST_TRAIN_BENCH_SLURM_RESERVATION",
                "POST_TRAIN_BENCH_SLURM_SUBQUEUE",
            )
        },
        "jobs": [],
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    frozen_environment = {
        "POST_TRAIN_BENCH_FROZEN_TOP_BRANCH": snapshot["top_branch"],
        "POST_TRAIN_BENCH_FROZEN_TOP_COMMIT": snapshot["top_commit"],
        "POST_TRAIN_BENCH_FROZEN_PTB_COMMIT": snapshot["ptb_commit"],
    }
    for launch in launches:
        result = subprocess.run(
            launch.command,
            cwd=PTB_ROOT,
            env=os.environ | launch.environment | frozen_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            receipt["state"] = "submission_failed"
            receipt["failure"] = {
                "cell_id": launch.cell_id,
                "stderr": result.stderr.strip(),
            }
            output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise ExperimentError(
                f"submission failed for {launch.cell_id} after {len(receipt['jobs'])} job(s): "
                f"{result.stderr.strip()} (receipt: {output})"
            )
        match = re.search(r"Submitted Slurm job (\d+)", result.stdout)
        if not match:
            receipt["state"] = "submission_failed"
            receipt["failure"] = {
                "cell_id": launch.cell_id,
                "stdout": result.stdout.strip(),
                "reason": "submitted command returned no parseable job id",
            }
            output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise ExperimentError(
                f"could not parse job id for {launch.cell_id}: {result.stdout} (receipt: {output})"
            )
        receipt["jobs"].append(
            {
                "cell_id": launch.cell_id,
                "job_id": match.group(1),
                "job_name": _command_option(launch, "--job-name"),
            }
        )
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    ownership_registry = ptb_env.get("POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY", "")
    if not pilot:
        receipt["state"] = "held"
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if ownership_registry:
        try:
            from awm import slurm_queue

            slurm_queue.register_receipt(
                output,
                registry_path=Path(ownership_registry),
            )
        except (OSError, slurm_queue.QueueError) as exc:
            receipt["state"] = "ownership_registration_failed"
            receipt["failure"] = {
                "reason": (
                    "all formal jobs remain held because ownership registration failed"
                    if not pilot
                    else "pilot jobs were submitted but ownership registration failed"
                ),
                "error": str(exc),
            }
            output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            disposition = (
                "formal jobs were submitted but remain held"
                if not pilot
                else "pilot jobs were submitted and must be resolved from this receipt"
            )
            raise ExperimentError(
                f"{disposition}; ownership registration failed: {exc} (receipt: {output})"
            ) from exc
    if keep_held:
        receipt["held_at"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return output
    if not pilot:
        job_ids = ",".join(job["job_id"] for job in receipt["jobs"])
        release = subprocess.run(
            _release_command(ptb_env, job_ids), text=True, capture_output=True, check=False
        )
        if release.returncode:
            receipt["state"] = "release_failed"
            receipt["failure"] = {
                "reason": "all formal jobs remain held because atomic release failed",
                "stderr": release.stderr.strip(),
            }
            output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise ExperimentError(
                f"formal jobs were submitted but remain held; release failed: "
                f"{release.stderr.strip()} (receipt: {output})"
            )
        receipt["released_at"] = datetime.now(timezone.utc).isoformat()
    receipt["state"] = "submitted"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return output


def _expanded_nodes(nodelist: str) -> set[str]:
    if not nodelist or nodelist == "(null)":
        return set()
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ExperimentError(
            f"cannot expand Slurm nodelist {nodelist}: "
            f"{result.stderr.strip() or 'scontrol failed'}"
        )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def release_held(receipt_path: Path) -> dict[str, Any]:
    """Release one registered held receipt after ownership and placement revalidation."""
    receipt_path = Path(receipt_path)
    receipt = load_receipt(receipt_path)
    if receipt.get("state") != "held":
        raise ExperimentError(
            f"receipt {receipt_path} is {receipt.get('state')!r}, expected 'held'"
        )
    ptb_env = read_ptb_env()
    registry_path = ptb_env.get("POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY", "")
    if not registry_path:
        raise ExperimentError("held release requires the Slurm ownership registry")
    try:
        from awm import slurm_queue

        snapshot = slurm_queue.collect_snapshot(Path(registry_path))
    except (OSError, slurm_queue.QueueError) as exc:
        raise ExperimentError(f"cannot inspect ownership before held release: {exc}") from exc
    if not snapshot.get("ownership_ok"):
        raise ExperimentError("OWNERSHIP FAIL: held jobs remain held")

    frozen_nodelist = str(
        (receipt.get("site") or {}).get("POST_TRAIN_BENCH_SLURM_NODELIST", "")
    )
    frozen_nodes = _expanded_nodes(frozen_nodelist)
    if not frozen_nodes:
        raise ExperimentError("held receipt has no frozen Slurm nodes")
    reservation = str(
        (receipt.get("site") or {}).get("POST_TRAIN_BENCH_SLURM_RESERVATION", "")
    )
    if not reservation:
        raise ExperimentError("held receipt has no frozen Slurm reservation")
    reservation_result = subprocess.run(
        ["scontrol", "show", "reservation", reservation, "-o"],
        text=True,
        capture_output=True,
        check=False,
    )
    if reservation_result.returncode:
        raise ExperimentError(
            f"cannot inspect frozen reservation {reservation}: "
            f"{reservation_result.stderr.strip() or 'scontrol failed'}"
        )
    reservation_nodes_match = re.search(r"\bNodes=(\S+)", reservation_result.stdout)
    reservation_nodes = _expanded_nodes(
        reservation_nodes_match.group(1) if reservation_nodes_match else ""
    )
    if reservation_nodes != frozen_nodes:
        raise ExperimentError(
            f"reservation {reservation} is not native two-node isolation: "
            f"{','.join(sorted(reservation_nodes)) or '(none)'} vs frozen "
            f"{','.join(sorted(frozen_nodes))}"
        )
    for job in receipt["jobs"]:
        job_id = str(job["job_id"])
        if _job_state(job_id) != "PENDING":
            raise ExperimentError(f"held job {job_id} is not PENDING; refusing release")
        result = subprocess.run(
            ["scontrol", "show", "job", job_id, "-o"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ExperimentError(
                f"cannot inspect held job {job_id}: {result.stderr.strip() or 'scontrol failed'}"
            )
        reason = re.search(r"\bReason=(\S+)", result.stdout)
        requested = re.search(r"\bReqNodeList=(\S+)", result.stdout)
        if not reason or reason.group(1) != "JobHeldUser":
            raise ExperimentError(
                f"held job {job_id} reason is "
                f"{reason.group(1) if reason else 'missing'}, expected JobHeldUser"
            )
        actual_nodes = _expanded_nodes(requested.group(1) if requested else "")
        if actual_nodes != frozen_nodes:
            raise ExperimentError(
                f"held job {job_id} ReqNodeList differs from frozen receipt: "
                f"{','.join(sorted(actual_nodes)) or '(null)'} vs "
                f"{','.join(sorted(frozen_nodes))}"
            )

    job_ids = ",".join(str(job["job_id"]) for job in receipt["jobs"])
    release = subprocess.run(
        _release_command(ptb_env, job_ids), text=True, capture_output=True, check=False
    )
    if release.returncode:
        raise ExperimentError(
            f"held jobs remain held; release failed: {release.stderr.strip() or 'scontrol failed'}"
        )
    receipt.pop("_path", None)
    receipt["state"] = "submitted"
    receipt["released_at"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def audit_result(result_dir: Path) -> list[str]:
    validator = PTB_ROOT / "src" / "utils" / "validate_completed_run.py"
    result = subprocess.run(
        [sys.executable, str(validator), str(result_dir), "--judge-profile", "official"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return []
    issues = []
    for line in result.stdout.splitlines():
        prefix = "COMPLETION ERROR: "
        if line.startswith(prefix):
            issues.append(line.removeprefix(prefix))
    return issues or [result.stderr.strip() or "completion validator failed without details"]


def load_receipt(filename: Path) -> dict[str, Any]:
    data = json.loads(filename.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("jobs"), list):
        raise ExperimentError(f"invalid batch receipt: {filename}")
    data["_path"] = str(filename.resolve())
    return data


def _job_state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-nX", "-j", job_id, "--format=State", "--parsable2"],
        text=True,
        capture_output=True,
        check=False,
    )
    states = [line.strip().split("|")[0] for line in result.stdout.splitlines() if line.strip()]
    return states[0] if states else "UNKNOWN"


def result_for_job(job_id: str) -> Path | None:
    env = read_ptb_env()
    results_root = Path(env.get("POST_TRAIN_BENCH_RESULTS_DIR", PTB_ROOT / "results"))
    matches = [path for path in results_root.glob(f"*/*_{job_id}") if path.is_dir()]
    if len(matches) > 1:
        raise ExperimentError(
            f"multiple result directories found for Slurm job {job_id}: {matches}"
        )
    return matches[0] if matches else None


def receipt_status(receipt: dict[str, Any]) -> list[dict[str, str | None]]:
    status = []
    for job in receipt["jobs"]:
        result_dir = result_for_job(job["job_id"])
        status.append(
            {
                "cell_id": job["cell_id"],
                "job_id": job["job_id"],
                "job_name": job.get("job_name"),
                "state": _job_state(job["job_id"]),
                "result_dir": str(result_dir) if result_dir else None,
            }
        )
    return status


def audit_receipt(receipt: dict[str, Any]) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    expected_source = receipt.get("source") or {}
    ownership = receipt.get("ownership") or {}
    contract = receipt.get("contract") or {}
    cells = {cell["id"]: cell for cell in receipt.get("cells") or []}
    site = receipt.get("site") or {}
    expected_context_validation = receipt.get("context_validation") or {}
    expected_container_digests = {
        "container": (contract.get("container") or {}).get("sha256"),
        "evaluation_container": (contract.get("evaluation_container") or {}).get("sha256"),
        "official_judge_container": contract.get("official_judge_container_sha256"),
    }
    for job in receipt_status(receipt):
        cell_issues = []
        if job["state"] != "COMPLETED":
            cell_issues.append(f"Slurm state is {job['state']}, expected COMPLETED")
        if not job["result_dir"]:
            cell_issues.append("result directory not found")
        else:
            result_dir = Path(job["result_dir"])
            cell_issues.extend(audit_result(result_dir))
            provenance_path = result_dir / "runtime_provenance.json"
            if provenance_path.is_file():
                try:
                    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                    runtime = provenance.get("agent_runtime") or {}
                    source = provenance.get("source") or {}
                    experiment = provenance.get("experiment") or {}
                    slurm = provenance.get("slurm") or {}
                    expected_cell = cells.get(job["cell_id"]) or {}
                    if int(runtime.get("resolved_context_tokens", 0)) != int(
                        expected_cell.get("context_tokens", 0)
                    ):
                        cell_issues.append(
                            "runtime resolved context differs from frozen cell setup"
                        )
                    if runtime.get("provider") != contract.get("agent_auth", {}).get("provider"):
                        cell_issues.append("runtime agent provider differs from frozen contract")
                    if runtime.get("requested_model") != expected_cell.get("agent_model"):
                        cell_issues.append("runtime requested agent model differs from frozen cell")
                    if runtime.get("effort") != expected_cell.get("effort"):
                        cell_issues.append("runtime effort differs from frozen cell")
                    if runtime.get("cli_version") != contract.get("agent_cli_version"):
                        cell_issues.append("runtime agent CLI differs from frozen contract")
                    context_record = (runtime.get("context_validation") or {}).get("record") or {}
                    context_digest = (runtime.get("context_validation") or {}).get("sha256")
                    if context_digest != (
                        expected_context_validation.get(job["cell_id"]) or {}
                    ).get("sha256"):
                        cell_issues.append(
                            "runtime context-validation digest differs from frozen receipt"
                        )
                    for key in ("provider", "project", "region"):
                        if context_record.get(key) != contract.get("agent_auth", {}).get(key):
                            cell_issues.append(
                                f"runtime context validation {key} differs from frozen contract"
                            )
                    if experiment.get("base_model") != expected_cell.get("base_model"):
                        cell_issues.append("runtime base model differs from frozen cell")
                    expected_revision = (
                        contract.get("base_models", {})
                        .get(expected_cell.get("base_model"), {})
                        .get("revision")
                    )
                    if experiment.get("base_model_revision") != expected_revision:
                        cell_issues.append(
                            "runtime base-model revision differs from frozen contract"
                        )
                    if (provenance.get("base_model_cache_snapshot") or {}).get("config_sha256") in (
                        None,
                        "missing",
                    ):
                        cell_issues.append("runtime base-model cache snapshot is incomplete")
                    if provenance.get("judge_profile") != "official":
                        cell_issues.append("runtime provenance judge profile is not official")
                    if slurm.get("partition") != site.get("POST_TRAIN_BENCH_SLURM_PARTITION"):
                        cell_issues.append("runtime Slurm partition differs from frozen site")
                    if str(slurm.get("cpus_per_task")) != str(contract.get("cpus")):
                        cell_issues.append("runtime CPU allocation differs from frozen contract")
                    expected_memory_mb = 128 * 1024
                    if str(slurm.get("memory_per_node")) != str(expected_memory_mb):
                        cell_issues.append("runtime memory allocation differs from frozen contract")
                    if len(slurm.get("gpu_uuids") or []) != 1:
                        cell_issues.append(
                            "runtime provenance does not contain exactly one GPU UUID"
                        )
                    if slurm.get("job_name") != job.get("job_name"):
                        cell_issues.append("runtime Slurm job name differs from frozen receipt")
                    for key in ("top_commit", "ptb_commit"):
                        if source.get(key) != expected_source.get(key):
                            cell_issues.append(f"runtime {key} differs from frozen receipt")
                    if source.get("top_branch") != ownership.get("branch"):
                        cell_issues.append("runtime top branch differs from frozen ownership")
                    if experiment.get("batch_id") != receipt.get("batch_id"):
                        cell_issues.append("runtime batch id differs from frozen receipt")
                    if experiment.get("cell_id") != job["cell_id"]:
                        cell_issues.append("runtime cell id differs from frozen receipt")
                    if experiment.get("spec_path") != ownership.get("spec"):
                        cell_issues.append("runtime spec path differs from frozen ownership")
                    if source.get("top_dirty") is not False or source.get("ptb_dirty") is not False:
                        cell_issues.append("runtime source snapshot is not clean")
                    if source.get("materialization") != "git-archive":
                        cell_issues.append(
                            "runtime source was not materialized from a frozen archive"
                        )
                    for key, expected_digest in expected_container_digests.items():
                        if (
                            expected_digest
                            and (provenance.get(key) or {}).get("sha256") != expected_digest
                        ):
                            cell_issues.append(f"runtime {key} digest differs from frozen receipt")
                except (OSError, ValueError, TypeError) as exc:
                    cell_issues.append(f"invalid runtime provenance: {exc}")
        issues[job["cell_id"]] = cell_issues
    return issues


def submit_research_judges(receipt: dict[str, Any]) -> Path:
    audit = audit_receipt(receipt)
    failures = {cell: issues for cell, issues in audit.items() if issues}
    if failures:
        formatted = [f"{cell}: {', '.join(issues)}" for cell, issues in failures.items()]
        raise ExperimentError(
            "research judges require complete official results:\n- " + "\n- ".join(formatted)
        )
    env = read_ptb_env()
    partition = env.get("POST_TRAIN_BENCH_SLURM_PARTITION", "")
    nodelist = env.get("POST_TRAIN_BENCH_SLURM_NODELIST", "")
    reservation = env.get("POST_TRAIN_BENCH_SLURM_RESERVATION", "")
    log_dir = PTB_ROOT / "logs" / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)
    source_receipt = Path(receipt["_path"])
    output = source_receipt.with_name(source_receipt.stem + "-research-judges.json")
    if output.exists():
        raise ExperimentError(f"refusing duplicate research-judge submission: {output}")
    jobs = []
    frozen_ptb_commit = (receipt.get("source") or {}).get("ptb_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", frozen_ptb_commit):
        raise ExperimentError("official receipt does not contain a valid frozen PTB commit")
    branch = (receipt.get("ownership") or {}).get("branch", "")
    spec = (receipt.get("ownership") or {}).get("spec", "")
    if not branch or (receipt.get("source") or {}).get("top_branch") != branch:
        raise ExperimentError("official receipt has no matching branch ownership")
    cells = {cell["id"]: cell for cell in receipt.get("cells") or []}
    context_validation = receipt.get("context_validation") or {}
    evidence = {
        (
            (context_validation.get(cell_id) or {}).get("path"),
            (context_validation.get(cell_id) or {}).get("sha256"),
        )
        for cell_id, cell in cells.items()
        if cell.get("agent_model") == "claude-opus-5[1m]" and cell.get("effort") == "xhigh"
    }
    if len(evidence) != 1:
        raise ExperimentError("official receipt has no unique Opus 5 xhigh 1M validation evidence")
    research_context_path, research_context_digest = evidence.pop()
    if (
        not research_context_path
        or not re.fullmatch(r"[0-9a-f]{64}", str(research_context_digest or ""))
        or not Path(research_context_path).is_file()
        or _sha256(Path(research_context_path)) != research_context_digest
    ):
        raise ExperimentError("Opus 5 xhigh 1M validation evidence is missing or changed")
    research_environment = {
        "POST_TRAIN_BENCH_FROZEN_TOP_BRANCH": branch,
        "POST_TRAIN_BENCH_FROZEN_PTB_COMMIT": frozen_ptb_commit,
        "POST_TRAIN_BENCH_BATCH_ID": receipt["batch_id"],
        "POST_TRAIN_BENCH_SPEC_PATH": spec,
        "POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD": str(research_context_path),
        "POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256": str(research_context_digest),
    }
    for job in receipt_status(receipt):
        job_name = _slug(f"{branch}.ptb.{receipt['batch_id']}.{job['cell_id']}.research")
        command = [
            "sbatch",
            "--parsable",
            f"--partition={partition}",
            f"--nodelist={nodelist}",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=04:00:00",
            f"--job-name={job_name}",
            f"--chdir={PTB_ROOT}",
            f"--output={log_dir}/ptb-research-{job['cell_id']}-%j.out",
            f"--error={log_dir}/ptb-research-{job['cell_id']}-%j.err",
        ]
        if reservation:
            command.append(f"--reservation={reservation}")
        command.extend(
            [
                str(PTB_ROOT / "src/commit_utils/slurm/research_judge.sbatch"),
                str(job["result_dir"]),
            ]
        )
        result = subprocess.run(
            command,
            cwd=PTB_ROOT,
            env=os.environ
            | research_environment
            | {
                "POST_TRAIN_BENCH_CELL_ID": job["cell_id"],
                "POST_TRAIN_BENCH_RUN_PURPOSE": "research",
                "POST_TRAIN_BENCH_SLURM_JOB_NAME": job_name,
            },
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ExperimentError(
                f"research judge submission failed for {job['cell_id']}: {result.stderr.strip()}"
            )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ExperimentError(f"invalid research judge job id: {result.stdout}")
        jobs.append({"cell_id": job["cell_id"], "job_id": job_id, "job_name": job_name})
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_receipt": str(source_receipt),
                "ownership": receipt["ownership"],
                "batch_id": receipt["batch_id"],
                "profile": "claude-opus-5[1m]-xhigh",
                "ptb_commit": frozen_ptb_commit,
                "context_validation": {
                    "path": research_context_path,
                    "sha256": research_context_digest,
                },
                "jobs": jobs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
