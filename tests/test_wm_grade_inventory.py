import hashlib
import json

import pytest

from tools.outcome_prediction import wm_grade_inventory as grades


def record(value, official=True):
    return {
        "cell_id": "r0-01",
        "example_id": "r0-01/exp-01",
        "label": {
            "accuracy": value,
            "official_metric": {"accuracy": value} if official else None,
        },
        "model_input": {"must_not_be_decoded": True},
        "prior_observations": [],
    }


def test_zero_is_valid_and_missing_is_not_zero():
    assert grades.valid_official_label(record(0)["label"])
    for value in (None, True, -0.1, 1.1, float("nan"), float("inf")):
        assert not grades.valid_official_label(record(value)["label"])
    assert not grades.valid_official_label(record(0.7, False)["label"])
    assert not grades.valid_official_label({"accuracy": 0.7, "official_metric": {"accuracy": 0.6}})


def test_metadata_is_invariant_to_valid_score_magnitude():
    outputs = [
        grades.summarize([json.dumps(record(v))], {"r0-01": "test"}) for v in (0, 0.01, 0.5, 1)
    ]
    assert all(output == outputs[0] for output in outputs)
    assert set(outputs[0][0]) == {
        "example_id",
        "cell_id",
        "partition",
        "has_valid_official_final_grade",
    }


def test_nonlabel_record_bodies_are_not_decoded():
    raw = json.dumps(record(0.4)).replace('{"must_not_be_decoded": true}', "INVALID_BODY")
    projected = grades.project(raw)
    assert set(projected) == {"cell_id", "example_id", "label"}
    assert grades.summarize([raw], {"r0-01": "test"})[0]["has_valid_official_final_grade"]


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        '{"cell_id":"x","cell_id":"x"}',
        json.dumps(record(0.3)) + " trailing",
        json.dumps(record(0.3))[:-1] + ",}",
    ],
)
def test_malformed_or_duplicate_top_level_keys_fail(raw):
    with pytest.raises(ValueError):
        grades.project(raw)


def test_unknown_duplicate_and_mismatched_identity_fail():
    raw = json.dumps(record(0.4))
    with pytest.raises(ValueError):
        grades.summarize([raw], {})
    with pytest.raises(ValueError):
        grades.summarize([raw, raw], {"r0-01": "test"})
    with pytest.raises(ValueError):
        grades.summarize([raw.replace("r0-01/exp-01", "r0-02/exp-01")], {"r0-01": "test"})


def test_pinned_private_build_and_immutable_output(tmp_path):
    inventory, split, output = (tmp_path / name for name in ("inventory", "split", "metadata"))
    inventory.write_text(json.dumps(record(0.1)) + "\n")
    split.write_text(json.dumps({"cell_partition": {"r0-01": "test"}}))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs = {"inventory_sha256": sha(inventory), "split_sha256": sha(split), "output": output}
    assert grades.build(inventory, split, **kwargs) == {"records": 1, "score_values_emitted": False}
    assert output.stat().st_mode & 0o777 == 0o600
    assert "accuracy" not in output.read_text()
    with pytest.raises(FileExistsError):
        grades.build(inventory, split, **kwargs)
    kwargs["output"] = tmp_path / "new"
    kwargs["inventory_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        grades.build(inventory, split, **kwargs)
    assert not kwargs["output"].exists()
