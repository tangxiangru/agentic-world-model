"""Manifest-driven discovery and validation of completed PostTrainBench results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from awm import paths
from awm import ptb_experiments as ptb

JUDGE_FLAGS = (
    "contamination",
    "disallowed_model",
    "disallowed_api_usage",
    "disallowed_ptb_lookup",
    "general_anomaly",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _canonical_judgement(result_dir: Path, stem: str) -> dict[str, Any]:
    rerun = result_dir / f"{stem}_rerun.json"
    canonical = result_dir / f"{stem}.json"
    if rerun.is_file() and rerun.stat().st_size:
        return _read_json(rerun)
    return _read_json(canonical)


def judge_flags(result_dir: Path) -> list[str]:
    verdicts = [
        _canonical_judgement(result_dir, "judgement_gpt5_4"),
        _canonical_judgement(result_dir, "judgement_api"),
        _canonical_judgement(result_dir, "judgement_ptb_lookup"),
        _canonical_judgement(result_dir, "judgement_general"),
    ]
    return [key for key in JUDGE_FLAGS if any(verdict.get(key) is True for verdict in verdicts)]


def _accuracy_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(attempt["accuracy"]) for attempt in attempts
              if isinstance(attempt.get("accuracy"), (int, float))]
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _placement_only_quarantine(attempt: dict[str, Any]) -> bool:
    reasons = attempt.get("quarantine_reasons") or []
    return bool(reasons) and all(
        "Slurm node" in str(reason) or "frozen site nodelist" in str(reason)
        for reason in reasons
    )


def _comparable_run_purpose(value: str) -> bool:
    """Formal manifest analysis never pools pilots or context smokes.

    Empty is accepted for older provenance written before run_purpose existed.
    """
    return not value or value == "formal" or value.startswith("formal-retry")


def _results_root() -> Path:
    env = ptb.read_ptb_env()
    return Path(env.get("POST_TRAIN_BENCH_RESULTS_DIR", ptb.PTB_ROOT / "results")).resolve()


def _receipts_root() -> Path:
    return paths.REPO_ROOT / "results" / "ptb"


def _split_nodelist(value: str) -> list[str]:
    """Split a Slurm hostlist on commas that are not inside brackets."""
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return [part for part in parts if part]


def _expand_nodelist(value: str) -> set[str]:
    """Expand the numeric Slurm hostlists frozen in PTB receipts without live Slurm."""
    expanded: set[str] = set()
    for part in _split_nodelist(value):
        left = part.find("[")
        if left < 0:
            expanded.add(part)
            continue
        right = part.find("]", left)
        if right < 0:
            return set()
        prefix, choices, suffix = part[:left], part[left + 1:right], part[right + 1:]
        for choice in choices.split(","):
            if "-" in choice:
                first, last = choice.split("-", 1)
                if not first.isdigit() or not last.isdigit():
                    return set()
                width = max(len(first), len(last))
                numbers = (str(number).zfill(width) for number in range(int(first), int(last) + 1))
            else:
                numbers = (choice,)
            for number in numbers:
                tails = _expand_nodelist(suffix) if "[" in suffix else {suffix}
                expanded.update(prefix + number + tail for tail in tails)
    return expanded


def _expected_nodes_by_job(batch_id: str) -> dict[str, set[str]]:
    """Read each job's frozen site from the tracked immutable receipts."""
    expected: dict[str, set[str]] = {}
    for receipt_path in sorted((_receipts_root() / batch_id).glob("*.json")):
        receipt = _read_json(receipt_path)
        if receipt.get("batch_id") != batch_id or not isinstance(receipt.get("jobs"), list):
            continue
        nodelist = str((receipt.get("site") or {}).get("POST_TRAIN_BENCH_SLURM_NODELIST", ""))
        nodes = _expand_nodelist(nodelist) if nodelist else set()
        for job in receipt["jobs"]:
            if isinstance(job, dict) and job.get("job_id"):
                expected[str(job["job_id"])] = nodes
    return expected


def discover_attempts(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Discover attempts by frozen provenance instead of directory-name conventions."""
    cells = {str(cell["id"]): cell for cell in manifest["cells"]}
    attempts: dict[str, list[dict[str, Any]]] = {cell_id: [] for cell_id in cells}
    expected_nodes = _expected_nodes_by_job(str(manifest.get("batch_id", "")))
    for provenance_path in _results_root().glob("*/*/runtime_provenance.json"):
        provenance = _read_json(provenance_path)
        experiment = provenance.get("experiment") or {}
        if experiment.get("batch_id") != manifest.get("batch_id"):
            continue
        cell_id = str(experiment.get("cell_id", ""))
        if cell_id not in cells:
            continue
        result_dir = provenance_path.parent
        metrics = _read_json(result_dir / "metrics.json")
        accuracy = metrics.get("accuracy")
        if not isinstance(accuracy, (int, float)):
            accuracy = None
        issues = ptb.audit_result(result_dir)
        slurm = provenance.get("slurm") or {}
        run_purpose = str(experiment.get("run_purpose", ""))
        job_id = str(slurm.get("job_id", ""))
        actual_node = str(slurm.get("node", ""))
        quarantine_reasons: list[str] = []
        if job_id in expected_nodes:
            frozen_nodes = expected_nodes[job_id]
            if not frozen_nodes:
                quarantine_reasons.append("frozen site nodelist in receipt could not be expanded")
            elif not actual_node:
                quarantine_reasons.append("runtime Slurm node is missing from provenance")
            elif actual_node not in frozen_nodes:
                quarantine_reasons.append(
                    f"runtime Slurm node {actual_node} is outside frozen site nodes "
                    f"{','.join(sorted(frozen_nodes))}"
                )
        attempts[cell_id].append(
            {
                "cell_id": cell_id,
                "job_id": job_id,
                "job_name": str(slurm.get("job_name", "")),
                "node": actual_node,
                "created_at": str(provenance.get("created_at", "")),
                "run_purpose": run_purpose,
                "comparable": _comparable_run_purpose(run_purpose),
                "result_dir": str(result_dir),
                "complete": not issues,
                "issues": issues,
                "eligible": not issues and not quarantine_reasons,
                "quarantined": bool(quarantine_reasons),
                "quarantine_reasons": quarantine_reasons,
                "accuracy": accuracy,
                "stderr": metrics.get("stderr"),
                "judge_flags": judge_flags(result_dir),
                "top_commit": str((provenance.get("source") or {}).get("top_commit", "")),
                "ptb_commit": str((provenance.get("source") or {}).get("ptb_commit", "")),
            }
        )
    for cell_attempts in attempts.values():
        cell_attempts.sort(key=lambda item: (item["created_at"], item["job_id"]))
    return attempts


def build_report(manifest: dict[str, Any]) -> dict[str, Any]:
    attempts = discover_attempts(manifest)
    rows = []
    for cell in manifest["cells"]:
        cell_id = str(cell["id"])
        cell_attempts = attempts[cell_id]
        complete_attempts = [
            attempt
            for attempt in cell_attempts
            if attempt["complete"] and attempt["comparable"]
        ]
        eligible_attempts = [attempt for attempt in complete_attempts if attempt["eligible"]]
        latest = cell_attempts[-1] if cell_attempts else None
        completed = eligible_attempts[-1] if eligible_attempts else (
            complete_attempts[-1] if complete_attempts else None
        )
        rows.append(
            {
                "cell_id": cell_id,
                "task": cell.get("task", manifest.get("contract", {}).get("task", "")),
                "base_model": cell.get("base_model", ""),
                "agent": cell.get("agent", ""),
                "agent_model": cell.get("agent_model", ""),
                "effort": cell.get("effort", ""),
                "context_tokens": cell.get("context_tokens"),
                "replicate": cell.get("replicate"),
                "complete": bool(complete_attempts),
                "eligible": bool(eligible_attempts),
                "quarantined": bool(complete_attempts) and not eligible_attempts,
                "completed_attempt": completed,
                "latest_attempt": latest,
                "attempt_count": len(cell_attempts),
            }
        )
    complete_rows = [row for row in rows if row["complete"]]
    eligible_rows = [row for row in rows if row["eligible"]]
    quarantined_rows = [row for row in rows if row["quarantined"]]
    flagged_rows = [
        row
        for row in eligible_rows
        if row["completed_attempt"] and row["completed_attempt"]["judge_flags"]
    ]
    primary_attempts = [
        row["completed_attempt"] for row in eligible_rows if row["completed_attempt"]
    ]
    sensitivity_attempts = primary_attempts + [
        row["completed_attempt"]
        for row in quarantined_rows
        if row["completed_attempt"]
        and _placement_only_quarantine(row["completed_attempt"])
    ]
    return {
        "schema_version": 1,
        "batch_id": manifest.get("batch_id", ""),
        "manifest": manifest.get("_path", ""),
        "spec": (manifest.get("ownership") or {}).get("spec", ""),
        "complete": len(complete_rows),
        "total": len(rows),
        "eligible_complete": len(eligible_rows),
        "clean_complete": len(eligible_rows) - len(flagged_rows),
        "flagged_complete": len(flagged_rows),
        "quarantined_complete": len(quarantined_rows),
        "accuracy_primary": _accuracy_summary(primary_attempts),
        "accuracy_placement_sensitivity": _accuracy_summary(sensitivity_attempts),
        "incomplete_cells": [row["cell_id"] for row in rows if not row["complete"]],
        "rows": rows,
    }


def _model_short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _score_text(row: dict[str, Any]) -> str:
    attempt = row.get("completed_attempt") or row.get("latest_attempt") or {}
    accuracy = attempt.get("accuracy")
    if accuracy is None:
        return "-"
    if row.get("task") == "aime2025":
        return f"{round(float(accuracy) * 30)}/30 ({float(accuracy):.4f})"
    return f"{float(accuracy):.4f}"


def _summary_text(label: str, summary: dict[str, Any]) -> str:
    if not summary["n"]:
        return f"ACCURACY {label} n=0 mean=- range=-"
    return (
        f"ACCURACY {label} n={summary['n']} mean={summary['mean']:.4f} "
        f"range={summary['min']:.4f}..{summary['max']:.4f}"
    )


def render_report(
    report: dict[str, Any],
    *,
    include_incomplete: bool = False,
    task: str | None = None,
    cell_ids: set[str] | None = None,
) -> str:
    lines = [
        f"batch={report['batch_id']}",
        (
            f"COMPLETE {report['complete']}/{report['total']} "
            f"eligible={report['eligible_complete']} clean={report['clean_complete']} "
            f"flagged={report['flagged_complete']} quarantined={report['quarantined_complete']}"
        ),
        f"manifest={report['manifest']}",
        f"spec={report['spec']}",
        _summary_text("primary", report["accuracy_primary"]),
        _summary_text("placement-sensitivity", report["accuracy_placement_sensitivity"]),
    ]
    if report["incomplete_cells"]:
        lines.append(f"incomplete={','.join(report['incomplete_cells'])}")
    for row in report["rows"]:
        if task and row["task"] != task:
            continue
        if cell_ids and row["cell_id"] not in cell_ids:
            continue
        if not include_incomplete and not row["complete"]:
            continue
        attempt = row.get("completed_attempt") or row.get("latest_attempt") or {}
        status = ("QUARANTINED" if row.get("quarantined") else
                  "COMPLETE" if row["complete"] else "INCOMPLETE")
        flags = ",".join(attempt.get("judge_flags") or []) or "clean"
        context = row.get("context_tokens")
        context_text = f"{int(context) // 1000}k" if context else "-"
        lines.append(
            f"{row['cell_id']} {status} task={row['task']} score={_score_text(row)} "
            f"model={_model_short(str(row['base_model']))} effort={row['effort']}/{context_text} "
            f"flags={flags} job={attempt.get('job_id') or '-'}"
        )
        if attempt.get("result_dir"):
            lines.append(f"  result={attempt['result_dir']}")
        if not row["complete"] and attempt.get("issues"):
            lines.append(f"  missing={'; '.join(attempt['issues'])}")
        if attempt.get("quarantine_reasons"):
            lines.append(f"  quarantine={'; '.join(attempt['quarantine_reasons'])}")
    return "\n".join(lines) + "\n"
