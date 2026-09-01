"""Manifest-driven discovery and validation of completed PostTrainBench results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _results_root() -> Path:
    env = ptb.read_ptb_env()
    return Path(env.get("POST_TRAIN_BENCH_RESULTS_DIR", ptb.PTB_ROOT / "results")).resolve()


def discover_attempts(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Discover attempts by frozen provenance instead of directory-name conventions."""
    cells = {str(cell["id"]): cell for cell in manifest["cells"]}
    attempts: dict[str, list[dict[str, Any]]] = {cell_id: [] for cell_id in cells}
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
        attempts[cell_id].append(
            {
                "cell_id": cell_id,
                "job_id": str(slurm.get("job_id", "")),
                "job_name": str(slurm.get("job_name", "")),
                "node": str(slurm.get("node", "")),
                "created_at": str(provenance.get("created_at", "")),
                "run_purpose": str(experiment.get("run_purpose", "")),
                "result_dir": str(result_dir),
                "complete": not issues,
                "issues": issues,
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
        complete_attempts = [attempt for attempt in cell_attempts if attempt["complete"]]
        latest = cell_attempts[-1] if cell_attempts else None
        completed = complete_attempts[-1] if complete_attempts else None
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
                "complete": completed is not None,
                "completed_attempt": completed,
                "latest_attempt": latest,
                "attempt_count": len(cell_attempts),
            }
        )
    complete_rows = [row for row in rows if row["complete"]]
    flagged_rows = [
        row
        for row in complete_rows
        if row["completed_attempt"] and row["completed_attempt"]["judge_flags"]
    ]
    return {
        "schema_version": 1,
        "batch_id": manifest.get("batch_id", ""),
        "manifest": manifest.get("_path", ""),
        "spec": (manifest.get("ownership") or {}).get("spec", ""),
        "complete": len(complete_rows),
        "total": len(rows),
        "clean_complete": len(complete_rows) - len(flagged_rows),
        "flagged_complete": len(flagged_rows),
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
            f"clean={report['clean_complete']} flagged={report['flagged_complete']}"
        ),
        f"manifest={report['manifest']}",
        f"spec={report['spec']}",
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
        status = "COMPLETE" if row["complete"] else "INCOMPLETE"
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
    return "\n".join(lines) + "\n"
