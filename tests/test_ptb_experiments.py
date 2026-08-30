from pathlib import Path

import pytest

from awm import paths
from awm import ptb_experiments as ptb

MANIFEST = paths.REPO_ROOT / "experiments/posttrainbench/gsm8k-claude5-1m-batch1.yaml"


def test_manifest_is_exact_approved_matrix() -> None:
    data = ptb.load_manifest(MANIFEST)
    launches = ptb.build_launches(data)
    assert [launch.cell_id for launch in launches] == ["b1", "b2", "b3", "b4", "b5", "b6"]
    assert all("official" in launch.command for launch in launches)
    assert all(
        launch.environment["POST_TRAIN_BENCH_REQUIRE_COMPLETE"] == "1" for launch in launches
    )


def test_pilot_is_b6_shape_and_one_hour() -> None:
    (launch,) = ptb.build_launches(ptb.load_manifest(MANIFEST), pilot=True)
    assert launch.cell_id == "b6"
    assert launch.command[launch.command.index("--hours") + 1] == "1"
    assert any("pilot_1h" in argument for argument in launch.command)


def test_manifest_rejects_non_1m_contract() -> None:
    data = ptb.load_manifest(MANIFEST)
    data["contract"]["context_tokens"] = 200_000
    with pytest.raises(ptb.ExperimentError, match="context_tokens"):
        ptb.validate_manifest(data)


def test_result_audit_requires_full_official_flow(tmp_path: Path) -> None:
    issues = ptb.audit_result(tmp_path)
    assert "missing or empty: metrics.json" in issues
    assert "missing or empty: judgement_general.json" in issues


def test_receipt_validation(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"schema_version": 1, "jobs": [{"cell_id": "b6", "job_id": "1"}]}')
    assert ptb.load_receipt(receipt)["jobs"][0]["cell_id"] == "b6"
