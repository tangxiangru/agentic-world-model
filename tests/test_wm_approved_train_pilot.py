import copy

import pytest

from tools.outcome_prediction import wm_approved_train_pilot as pilot


class Poison(dict):
    def __getitem__(self, key):
        if key != "cell_id":
            raise AssertionError("Heldout field accessed")
        return super().__getitem__(key)


def fixture():
    split = {
        "train_cell_ids": ["run-a", "run-b"],
        "test_cell_ids": ["test"],
    }
    drafts = []
    for cell, status in [("run-a", "approved"), ("run-b", "draft_needs_review")]:
        drafts.append(
            {
                "cell_id": cell,
                "example_id": cell + "/exp-01",
                "status": status,
                "audit": {"issues": [] if status == "approved" else [{"kind": "pending"}]},
                "model_input": {"task": {"benchmark": "fixture"}},
            }
        )
    records = [
        {k: v for k, v in drafts[0].items() if k in {"cell_id", "example_id", "model_input"}}
    ]
    rows = [{"cell_id": d["cell_id"], "example_id": d["example_id"]} for d in drafts]
    return {"drafts": drafts}, rows, records, {"label_gaps": []}, split


def test_coverage_filters_test_before_other_access_and_lists_omissions():
    prepared, rows, records, joined, split = fixture()
    prepared["drafts"].append(Poison(cell_id="test"))
    rows.append(Poison(cell_id="test"))
    result = pilot.cohort_coverage(prepared, rows, records, joined, split)
    assert result["retained_targets"] == 1
    assert result["train_inventory_targets"] == 2
    assert result["omitted_targets"] == [
        {
            "example_id": "run-b/exp-01",
            "cell_id": "run-b",
            "draft_status": "draft_needs_review",
            "reason": "whole_recipe_not_approved",
            "draft_issue_kinds": ["pending"],
        }
    ]
    assert result["run_coverage"][1]["retained_targets"] == 0


def test_coverage_reports_missing_official_without_reading_any_label():
    prepared, rows, _records, joined, split = fixture()
    rows[0]["label"] = object()
    joined["label_gaps"] = [
        {"example_id": "run-a/exp-01", "reason": "missing_valid_official_final_label"}
    ]
    result = pilot.cohort_coverage(prepared, rows, [], joined, split)
    assert result["omitted_targets"][0]["reason"] == "missing_valid_official_final_label"
    assert result["benchmark_coverage"]["fixture"]["retained_runs"] == 0


def test_coverage_rejects_missing_and_duplicate_train_identity():
    prepared, rows, records, joined, split = fixture()
    with pytest.raises(ValueError, match="coverage"):
        pilot.cohort_coverage(prepared, rows[:-1], records, joined, split)
    with pytest.raises(ValueError, match="coverage"):
        pilot.cohort_coverage(prepared, rows + copy.deepcopy(rows[:1]), records, joined, split)


def test_every_source_benchmark_needs_four_runs():
    assert not pilot.fit_is_permitted({"benchmark_coverage": {}})
    assert not pilot.fit_is_permitted(
        {"benchmark_coverage": {"a": {"retained_runs": 4}, "b": {"retained_runs": 3}}}
    )
    assert pilot.fit_is_permitted(
        {"benchmark_coverage": {"a": {"retained_runs": 4}, "b": {"retained_runs": 4}}}
    )


def test_existing_directory_refused_before_any_read(tmp_path):
    with pytest.raises(FileExistsError):
        pilot.build_pilot("missing", "missing", "missing", tmp_path)


def test_symlink_output_refused_before_any_read(tmp_path):
    destination = tmp_path / "link"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        pilot.build_pilot("missing", "missing", "missing", destination)
