from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from awm.experiment import ExperimentBundle, ExperimentError, open_experiments

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples/experiments/local-smoke"


def copy_example(tmp_path: Path, name: str = "exp") -> ExperimentBundle:
    target = tmp_path / name
    shutil.copytree(EXAMPLE, target)
    return ExperimentBundle(target)


def complete_result(bundle: ExperimentBundle) -> None:
    result = yaml.safe_load(bundle.result_path.read_text())
    refs = [m["evidence_ref"] for m in result["result"]["measurements"]]
    result["result"]["artifact_status"] = "loadable"
    result["result"]["selected_as_next_incumbent"] = True
    result["scientist_assessment"] = {
        "outcome": {
            "verdict": "supported",
            "basis": "matched_eval",
            "summary": "Matched accuracy increased from 0.25 to 1.0.",
            "evidence_refs": refs,
        },
        "mechanism": {
            "verdict": "supported",
            "basis": "diagnostic_eval",
            "summary": "The dedicated four-case diagnostic shows all recorded errors are fixed.",
            "evidence_refs": refs,
        },
    }
    result["scientist_decision"] = {
        "action": "adopt",
        "rationale": "The candidate satisfies both the matched outcome and diagnostic checks.",
    }
    bundle.result_path.write_text(yaml.safe_dump(result, sort_keys=False))


def test_complete_lifecycle(tmp_path: Path) -> None:
    bundle = copy_example(tmp_path)
    assert bundle.card["experiment_id"] == "local-smoke"

    manifest = bundle.freeze()
    assert bundle.state["status"] == "frozen"
    assert manifest["card"]["sha256"]
    assert all(item["exists"] for item in manifest["inputs"])

    summary = bundle.run()
    assert summary["execution_status"] == "completed"
    assert summary["measurements"][0]["matched_delta"] == pytest.approx(0.75)
    assert bundle.state["status"] == "awaiting_review"
    assert (bundle.directory / "artifacts/candidate-checkpoint/model.json").is_file()

    complete_result(bundle)
    result = bundle.finalize()
    assert result["scientist_assessment"]["outcome"]["verdict"] == "supported"
    assert bundle.state["status"] == "closed"
    assert open_experiments(tmp_path) == []


def test_frozen_card_is_immutable(tmp_path: Path) -> None:
    bundle = copy_example(tmp_path)
    bundle.freeze()
    bundle.card_path.write_text(bundle.card_path.read_text() + "\n# changed\n")
    with pytest.raises(ExperimentError, match="changed after freeze"):
        bundle.run()


def test_failed_phase_still_requires_scientist_review(tmp_path: Path) -> None:
    bundle = copy_example(tmp_path)
    card = yaml.safe_load(bundle.card_path.read_text())
    card["execution"]["phases"][0]["command"] = ["python3", "-c", "raise SystemExit(7)"]
    bundle.card_path.write_text(yaml.safe_dump(card, sort_keys=False))

    bundle.freeze()
    summary = bundle.run()
    assert summary["execution_status"] == "failed"
    assert bundle.state["status"] == "awaiting_review"
    assert open_experiments(tmp_path)[0]["experiment_id"] == "local-smoke"


def test_scaffold_refuses_to_overwrite_and_needs_grounded_evidence(tmp_path: Path) -> None:
    bundle = ExperimentBundle.scaffold(tmp_path / "fresh", "fresh-001", "Fresh")
    assert bundle.state["status"] == "draft"
    with pytest.raises(ExperimentError, match="refusing to overwrite"):
        ExperimentBundle.scaffold(tmp_path / "fresh")

    card = yaml.safe_load(bundle.card_path.read_text())
    card["observed_problem"]["evidence"] = []
    bundle.card_path.write_text(yaml.safe_dump(card, sort_keys=False))
    with pytest.raises(ExperimentError, match="at least one rollout"):
        _ = bundle.card


def test_result_cannot_claim_mechanism_from_aggregate_score(tmp_path: Path) -> None:
    bundle = copy_example(tmp_path)
    bundle.freeze()
    bundle.run()
    complete_result(bundle)
    result = yaml.safe_load(bundle.result_path.read_text())
    result["scientist_assessment"]["mechanism"]["basis"] = "matched_eval"
    bundle.result_path.write_text(yaml.safe_dump(result, sort_keys=False))

    with pytest.raises(ExperimentError, match="requires a diagnostic_eval"):
        bundle.finalize()
