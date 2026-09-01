import json
from pathlib import Path

from awm import ptb_results


def _write_attempt(
    root: Path,
    *,
    cell_id: str,
    job_id: str,
    accuracy: float | None,
    anomaly: bool = False,
) -> Path:
    result = root / f"agent_{cell_id}" / f"task_model_{job_id}"
    result.mkdir(parents=True)
    (result / "runtime_provenance.json").write_text(
        json.dumps(
            {
                "created_at": f"2026-09-01T00:00:{job_id[-2:]}+00:00",
                "experiment": {"batch_id": "batch-v1", "cell_id": cell_id},
                "slurm": {"job_id": job_id, "job_name": f"batch.{cell_id}", "node": "node0"},
                "source": {"top_commit": "a" * 40, "ptb_commit": "b" * 40},
            }
        ),
        encoding="utf-8",
    )
    if accuracy is not None:
        (result / "metrics.json").write_text(
            json.dumps({"accuracy": accuracy, "stderr": 0.01}), encoding="utf-8"
        )
    (result / "judgement_general.json").write_text(
        json.dumps({"general_anomaly": anomaly}), encoding="utf-8"
    )
    return result


def test_report_discovers_results_by_provenance_and_keeps_flags(
    tmp_path: Path, monkeypatch
) -> None:
    complete = _write_attempt(tmp_path, cell_id="g01", job_id="101", accuracy=0.75, anomaly=True)
    _write_attempt(tmp_path, cell_id="a01", job_id="102", accuracy=None)
    monkeypatch.setattr(ptb_results, "_results_root", lambda: tmp_path)
    monkeypatch.setattr(
        ptb_results.ptb,
        "audit_result",
        lambda result: [] if result == complete else ["missing metrics"],
    )
    manifest = {
        "_path": "/repo/manifest.yaml",
        "batch_id": "batch-v1",
        "ownership": {"spec": "doc/spec.md"},
        "contract": {"tasks": ["gsm8k", "aime2025"]},
        "cells": [
            {
                "id": "g01",
                "task": "gsm8k",
                "base_model": "Qwen/Qwen3-1.7B-Base",
                "agent": "claude_vertex_max",
                "agent_model": "claude-opus-5[1m]",
                "effort": "max",
                "context_tokens": 1_000_000,
            },
            {
                "id": "a01",
                "task": "aime2025",
                "base_model": "Qwen/Qwen3-1.7B-Base",
                "agent": "claude_vertex_max",
                "agent_model": "claude-opus-5[1m]",
                "effort": "max",
                "context_tokens": 1_000_000,
            },
        ],
    }

    report = ptb_results.build_report(manifest)

    assert report["complete"] == 1
    assert report["clean_complete"] == 0
    assert report["flagged_complete"] == 1
    assert report["incomplete_cells"] == ["a01"]
    assert report["rows"][0]["completed_attempt"]["judge_flags"] == ["general_anomaly"]
    rendered = ptb_results.render_report(report, include_incomplete=True)
    assert "g01 COMPLETE" in rendered
    assert "a01 INCOMPLETE" in rendered
    assert "result=" in rendered
