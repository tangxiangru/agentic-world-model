"""Exercise history extraction against the recorder's actual measurement schema."""

import json
from pathlib import Path

import pytest

from tools.outcome_prediction import rpm_judge as judge


def row(card_id, at):
    return {
        "example_id": f"recorder-fixture/{card_id}",
        "cell_id": "recorder-fixture",
        "card_id": card_id,
        "first_submitted_at": at,
        "source_card": f"cells/recorder-fixture/wm/cards/{card_id}/card.json",
        "lineage": [{"card_id": card_id, "recipe": {"method": "sft"}}],
        "y": 0.99,
    }


def write_submission(root, prior, measurements, *, number=1, at="2026-09-04T01:00:00Z"):
    path = root / prior["source_card"]
    path.parent.mkdir(parents=True, exist_ok=True)
    card = {
        "schema_version": "awm-experiment-card-v1",
        "card_id": prior["card_id"],
        "setup": {
            "base_model": "google/gemma-3-4b-pt",
            "parent_checkpoint": {"origin": "base_model"},
            "method": {"family": "sft", "hyperparams": {"lr": 1e-5}},
            "data": [],
        },
        "result": {"execution": "completed", "measurements": measurements},
    }
    (path.parent / f"record-{number:02d}.json").write_text(
        json.dumps({"event": "submit", "at": at, "card": card})
    )


def fixture_rows():
    return (
        row("exp-01", "2026-09-04T00:00:00Z"),
        row("exp-02", "2026-09-04T05:00:00Z"),
        row("exp-03", "2026-09-04T06:00:00Z"),
    )


@pytest.mark.parametrize(
    "measurement",
    [
        {"metric": "accuracy", "value": 0.75, "n": 150},
        {"metric": "accuracy", "value": 75, "n": 150},
        {"name": "dev_gsm8k", "value": 0.75, "n": 150},
        {"name": "primary_dev", "metric": "accuracy", "value": 0.75, "n": 150},
    ],
)
def test_recorder_accuracy_is_available_without_a_name_field(tmp_path, measurement):
    prior, a, b = fixture_rows()
    write_submission(tmp_path, prior, [measurement])
    history = judge.observable_history([prior, a, b], tmp_path, a, b)
    assert len(history) == 1
    assert history[0]["observed_accuracy"] == pytest.approx(0.75)
    assert history[0]["observed_accuracy"] != prior["y"]


@pytest.mark.parametrize(
    "measurements, expected",
    [
        (
            [
                {"metric": "accuracy", "value": 0.4, "n": 150},
                {"metric": "accuracy", "value": 0.75, "n": 500},
            ],
            0.75,
        ),
        (
            [
                {"metric": "accuracy", "value": 0.75, "n": 500},
                {"metric": "accuracy", "value": 75, "n": 500},
            ],
            0.75,
        ),
        (
            [
                {"metric": "accuracy", "value": 0.75, "n": 500},
                {"metric": "accuracy_sampling", "value": 0.65, "n": 500},
            ],
            None,
        ),
        ([{"metric": "training_loss", "value": 0.75, "n": 500}], None),
    ],
)
def test_largest_sample_size_does_not_mean_largest_accuracy(tmp_path, measurements, expected):
    prior, a, b = fixture_rows()
    write_submission(tmp_path, prior, measurements)
    history = judge.observable_history([prior, a, b], tmp_path, a, b)
    if expected is None:
        assert history == []
    else:
        assert len(history) == 1
        assert history[0]["observed_accuracy"] == pytest.approx(expected)


def test_metric_schema_preserves_the_strict_earlier_candidate_cutoff(tmp_path):
    prior, a, b = fixture_rows()
    write_submission(tmp_path, prior, [{"metric": "accuracy", "value": 0.25, "n": 150}])
    write_submission(
        tmp_path,
        prior,
        [{"metric": "accuracy", "value": 0.85, "n": 500}],
        number=2,
        at=a["first_submitted_at"],
    )
    write_submission(
        tmp_path,
        prior,
        [{"metric": "accuracy", "value": 0.95, "n": 1000}],
        number=3,
        at="2026-09-04T05:30:00Z",
    )
    history = judge.observable_history([prior, a, b], tmp_path, a, b)
    assert len(history) == 1
    assert history[0]["observed_accuracy"] == 0.25


@pytest.mark.needs_data
def test_real_recorder_parent_measurement_reaches_judge_history():
    root = Path(__file__).resolve().parents[1] / "data/traj/raw/awm-gsm8k-trajectories"
    folder = root / "cells/r0-06/wm/cards"
    if not (folder / "exp-02/record-02.json").exists():
        pytest.skip("Downloaded r0 recorder arm is unavailable")
    rows = []
    for cid in ("exp-02", "exp-03", "exp-04"):
        first = json.loads((folder / cid / "record-01.json").read_text())
        rows.append(
            {
                "example_id": f"r0-06/{cid}",
                "cell_id": "r0-06",
                "card_id": cid,
                "first_submitted_at": first["at"],
                "source_card": f"cells/r0-06/wm/cards/{cid}/card.json",
                "lineage": [{"card_id": cid, "recipe": judge.canonical(first["card"])[0]}],
                "y": 0.99,
            }
        )
    measurements = json.loads((folder / "exp-02/record-02.json").read_text())["card"]["result"][
        "measurements"
    ]
    assert any(m.get("metric") == "accuracy" and "name" not in m for m in measurements)
    history = judge.observable_history(rows, root, rows[1], rows[2])
    assert len(history) == 1
    assert history[0]["observed_accuracy"] == pytest.approx(0.83)
