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
    run_purpose: str = "formal",
) -> Path:
    result = root / f"agent_{cell_id}" / f"task_model_{job_id}"
    result.mkdir(parents=True)
    (result / "runtime_provenance.json").write_text(
        json.dumps(
            {
                "created_at": f"2026-09-01T00:00:{job_id[-2:]}+00:00",
                "experiment": {
                    "batch_id": "batch-v1",
                    "cell_id": cell_id,
                    "run_purpose": run_purpose,
                },
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
        lambda result, **_kwargs: [] if result == complete else ["missing metrics"],
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
    assert report["eligible_complete"] == 1
    assert report["clean_complete"] == 0
    assert report["flagged_complete"] == 1
    assert report["quarantined_complete"] == 0
    assert report["accuracy_primary"] == {"n": 1, "mean": 0.75, "min": 0.75, "max": 0.75}
    assert report["accuracy_placement_sensitivity"] == report["accuracy_primary"]
    assert report["incomplete_cells"] == ["a01"]
    assert report["rows"][0]["completed_attempt"]["judge_flags"] == ["general_anomaly"]
    rendered = ptb_results.render_report(report, include_incomplete=True)
    assert "g01 COMPLETE" in rendered
    assert "a01 INCOMPLETE" in rendered
    assert "ACCURACY primary n=1 mean=0.7500 range=0.7500..0.7500" in rendered
    assert "result=" in rendered


def test_report_keeps_a_complete_spillover_score_but_quarantines_it(
    tmp_path: Path, monkeypatch
) -> None:
    result = _write_attempt(tmp_path / "raw", cell_id="g01", job_id="101", accuracy=0.75)
    receipts = tmp_path / "tracked"
    batch = receipts / "batch-v1"
    batch.mkdir(parents=True)
    (batch / "formal.json").write_text(
        json.dumps(
            {
                "batch_id": "batch-v1",
                "site": {
                    "POST_TRAIN_BENCH_SLURM_NODELIST": "slurm2-a3nodesetondem-[0-1]"
                },
                "jobs": [{"cell_id": "g01", "job_id": "101"}],
            }
        )
    )
    provenance = json.loads((result / "runtime_provenance.json").read_text())
    provenance["slurm"]["node"] = "slurm2-a3nodeset-0"
    (result / "runtime_provenance.json").write_text(json.dumps(provenance))
    monkeypatch.setattr(ptb_results, "_results_root", lambda: tmp_path / "raw")
    monkeypatch.setattr(ptb_results, "_receipts_root", lambda: receipts)
    monkeypatch.setattr(ptb_results.ptb, "audit_result", lambda _result, **_kwargs: [])
    manifest = {
        "_path": "/repo/manifest.yaml",
        "batch_id": "batch-v1",
        "ownership": {"spec": "doc/spec.md"},
        "contract": {"task": "gsm8k"},
        "cells": [
            {
                "id": "g01",
                "replicate": 1,
                "base_model": "google/gemma-3-4b-pt",
                "agent": "claude_vertex_high_awm",
                "agent_model": "claude-opus-5[1m]",
                "effort": "high",
                "context_tokens": 1_000_000,
            }
        ],
    }

    report = ptb_results.build_report(manifest)

    assert report["complete"] == 1
    assert report["eligible_complete"] == 0
    assert report["clean_complete"] == 0
    assert report["quarantined_complete"] == 1
    assert report["accuracy_primary"] == {"n": 0, "mean": None, "min": None, "max": None}
    assert report["accuracy_placement_sensitivity"] == {
        "n": 1, "mean": 0.75, "min": 0.75, "max": 0.75
    }
    attempt = report["rows"][0]["completed_attempt"]
    assert attempt["accuracy"] == 0.75 and attempt["complete"] is True
    assert attempt["eligible"] is False and attempt["quarantined"] is True
    assert "outside frozen site nodes" in attempt["quarantine_reasons"][0]
    rendered = ptb_results.render_report(report)
    assert "g01 QUARANTINED" in rendered
    assert "score=0.7500" in rendered
    assert "quarantine=runtime Slurm node" in rendered
    assert "ACCURACY primary n=0 mean=- range=-" in rendered
    assert "ACCURACY placement-sensitivity n=1 mean=0.7500" in rendered


def test_slurm_nodelist_expansion_is_offline_and_range_aware() -> None:
    assert ptb_results._expand_nodelist("node-[00-01],other3") == {
        "node-00", "node-01", "other3"
    }


def test_a_completed_pilot_never_enters_formal_coverage_or_accuracy(
    tmp_path: Path, monkeypatch
) -> None:
    pilot = _write_attempt(
        tmp_path, cell_id="g01", job_id="101", accuracy=0.99, run_purpose="pilot-1h"
    )
    monkeypatch.setattr(ptb_results, "_results_root", lambda: tmp_path)
    monkeypatch.setattr(ptb_results, "_receipts_root", lambda: tmp_path / "no-receipts")
    monkeypatch.setattr(ptb_results.ptb, "audit_result", lambda _result, **_kwargs: [])
    manifest = {
        "_path": "/repo/manifest.yaml",
        "batch_id": "batch-v1",
        "ownership": {"spec": "doc/spec.md"},
        "contract": {"task": "gsm8k"},
        "cells": [
            {
                "id": "g01",
                "base_model": "google/gemma-3-4b-pt",
                "agent": "claude_vertex_high_awm",
                "agent_model": "claude-opus-5[1m]",
                "effort": "high",
                "context_tokens": 1_000_000,
                "replicate": 1,
            }
        ],
    }

    report = ptb_results.build_report(manifest)

    assert report["complete"] == 0 and report["eligible_complete"] == 0
    assert report["accuracy_primary"]["n"] == 0
    assert report["rows"][0]["latest_attempt"]["result_dir"] == str(pilot)
    assert report["rows"][0]["latest_attempt"]["comparable"] is False
