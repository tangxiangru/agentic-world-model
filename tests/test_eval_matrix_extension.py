"""Portable 3K extension checks; no private corpus, HF cache, credentials or GPU."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/outcome_prediction/verify_eval_matrix_extension.py"
SPEC = importlib.util.spec_from_file_location("verify_eval_matrix_extension", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def refresh_hashes(root):
    manifest = M.BASE.read_json(root, "bundle_files.sha256.json")
    for relative in manifest["files"]:
        manifest["files"][relative] = M.BASE.sha256(root / relative)
    write_json(root / "bundle_files.sha256.json", manifest)


def replace_extension(root, rows):
    # Keep the redundant combined/phase copies internally consistent so semantic
    # tests are not merely failing on the first stale SHA or duplicate copy.
    write_rows(root / "experiment_matrix_extension_2k.jsonl", rows)
    (root / "experiment_matrix_all_3k.jsonl").write_bytes(
        (root / "experiment_matrix.jsonl").read_bytes()
        + (root / "experiment_matrix_extension_2k.jsonl").read_bytes()
    )
    for phase in M.NEW_PHASES:
        write_rows(root / f"phases/{phase}.jsonl", [r for r in rows if r["phase"] == phase])
    refresh_hashes(root)


@pytest.fixture(scope="session")
def published():
    if any(not (M.DEFAULT_BUNDLE / relative).is_file() for relative in M.ADDITIVE_FILES):
        pytest.skip("Published extension assets are not present yet")
    return M.DEFAULT_BUNDLE


@pytest.fixture
def bundle(published, tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(published, target)
    return target


def test_published_extension(published):
    report = M.verify_extension(published)
    assert report["ok"], report
    assert report["counts"]["combined_cells"] == 3000
    assert report["counts"]["candidate_checkpoints"] == 516
    assert report["counts"]["core_cells"] == 1232
    assert report["counts"]["execution_stage_counts"] == {"pilot": 196, "development": 1784, "locked_test": 1020}
    assert "not_launch_ready" in report["status"]
    assert any("516" in warning and "not proof" in warning for warning in report["warnings"])


def test_cli_missing_bundle_is_failure_without_network(tmp_path, capsys):
    assert M.main(["--bundle", str(tmp_path / "missing"), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "Original 1K bundle check failed" in report["errors"][0]


def test_new_policy_has_exact_intended_settings(tmp_path):
    generation = {"do_sample": True, "temperature": 1, "top_k": 0, "top_p": 1,
                  "min_p": 0, "repetition_penalty": 1, "max_new_tokens": 4000,
                  "eos_token_id": [1, 106]}
    request = {"temperature": 1, "top_p": 1, "max_tokens": 4000, "n": 1,
               "presence_penalty": 0, "frequency_penalty": 0, "stop": [],
               "extra_body": {"top_k": 0, "min_p": 0, "repetition_penalty": 1,
                              "min_tokens": 0, "ignore_eos": False, "stop_token_ids": [1, 106]}}
    write_json(tmp_path / "configs/G03/generation_config.json", generation)
    write_json(tmp_path / "configs/G03/request_template.json", request)
    M._validate_new_policy(tmp_path, "G03")
    request["extra_body"]["ignore_eos"] = True
    write_json(tmp_path / "configs/G03/request_template.json", request)
    with pytest.raises(M.BASE.InvalidBundle, match="Invalid request parameter types"):
        M._validate_new_policy(tmp_path, "G03")


@pytest.mark.parametrize(("field", "value", "message"), [
    ("n_passes", 9, "ten passes"),
    ("n_questions_per_pass", 100, "question count"),
    ("protocol_path", "alternative.json", "frozen seed/runtime protocol"),
    ("release_status", "ready", "preflight-required"),
    ("historical_reuse_status", "verified", "remain unverified"),
    ("primary_evaluation", True, "core-policy flag mismatch"),
    ("phase", "6_extension_development", "phase assignment mismatch"),
])
def test_semantic_cell_changes_fail_with_consistent_hashes(bundle, field, value, message):
    rows = M.BASE.read_json(bundle, "experiment_matrix_extension_2k.jsonl", lines=True)
    row = next(r for r in rows if r["phase"] == "5_extension_pilot")
    row[field] = value
    replace_extension(bundle, rows)
    report = M.verify_extension(bundle)
    assert not report["ok"]
    assert message in report["errors"][0]


def test_original_bytes_locked_even_after_hash_map_refresh(bundle):
    path = bundle / "experiment_matrix.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    refresh_hashes(bundle)
    assert "Original frozen bytes changed" in M.verify_extension(bundle)["errors"][0]


def test_combined_requires_raw_original_prefix_not_just_same_rows(bundle):
    combined = M.BASE.read_json(bundle, "experiment_matrix_all_3k.jsonl", lines=True)
    # Formatting-only reserialization retains all data but not the frozen prefix.
    (bundle / "experiment_matrix_all_3k.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in combined), encoding="utf-8"
    )
    refresh_hashes(bundle)
    assert "exact original bytes" in M.verify_extension(bundle)["errors"][0]


def test_duplicate_policy_pair_rejected(bundle):
    rows = M.BASE.read_json(bundle, "experiment_matrix_extension_2k.jsonl", lines=True)
    rows[1]["checkpoint_id"] = rows[0]["checkpoint_id"]
    rows[1]["generation_config_id"] = rows[0]["generation_config_id"]
    replace_extension(bundle, rows)
    assert "Duplicate combined checkpoint-policy" in M.verify_extension(bundle)["errors"][0]


def test_phase_files_must_match_entire_rows(bundle):
    relative = "phases/5_extension_pilot.jsonl"
    rows = M.BASE.read_json(bundle, relative, lines=True)
    rows[0]["n_passes"] = 9
    write_rows(bundle / relative, rows)
    refresh_hashes(bundle)
    assert "phase content/partition mismatch" in M.verify_extension(bundle)["errors"][0]


def test_new_policy_request_semantics_not_only_hashes(bundle):
    relative = "configs/G03/request_template.json"
    request = M.BASE.read_json(bundle, relative)
    request["temperature"] = .5
    write_json(bundle / relative, request)
    rows = M.BASE.read_json(bundle, "experiment_matrix_extension_2k.jsonl", lines=True)
    for row in rows:
        if row["generation_config_id"] == "G03":
            row["request_template_sha256"] = M.BASE.sha256(bundle / relative)
    replace_extension(bundle, rows)
    assert "Unexpected extension request parameters" in M.verify_extension(bundle)["errors"][0]


def test_original_session_assignments_are_inherited(bundle):
    rows = M.BASE.read_json(bundle, "selected_checkpoints_all_516.jsonl", lines=True)
    old_ids = set(M.BASE.read_json(bundle, "selected_ids_400.json"))
    row = next(r for r in rows if r["checkpoint_id"] not in old_ids)
    row["split"] = "development" if row["split"] == "locked_session_test" else "locked_session_test"
    write_rows(bundle / "selected_checkpoints_all_516.jsonl", rows)
    refresh_hashes(bundle)
    assert "Original session reservation changed" in M.verify_extension(bundle)["errors"][0]


def test_extension_file_cannot_be_excluded_from_hash_map(bundle):
    manifest = M.BASE.read_json(bundle, "bundle_files.sha256.json")
    manifest["files"].pop("extension_summary.json")
    manifest["excluded"].append("extension_summary.json")
    write_json(bundle / "bundle_files.sha256.json", manifest)
    assert "Required assets absent from hash map" in M.verify_extension(bundle)["errors"][0]


def test_execution_stage_order_is_enforced(bundle):
    plan = M.BASE.read_json(bundle, "execution_plan.json")
    plan["stages"].reverse()
    write_json(bundle / "execution_plan.json", plan)
    refresh_hashes(bundle)
    assert "Execution stages must be ordered" in M.verify_extension(bundle)["errors"][0]


def test_execution_plan_does_not_grant_approval(bundle):
    plan = M.BASE.read_json(bundle, "execution_plan.json")
    plan["status"] = "approved"
    write_json(bundle / "execution_plan.json", plan)
    refresh_hashes(bundle)
    assert "Execution plan is not a proposal" in M.verify_extension(bundle)["errors"][0]


@pytest.mark.parametrize(("field", "value"), [
    ("auto_advance", True), ("require_explicit_approval", False),
    ("auto_launch", True), ("automatic_launch", True),
])
def test_execution_plan_cannot_waive_operator_approval(bundle, field, value):
    plan = M.BASE.read_json(bundle, "execution_plan.json")
    plan[field] = value
    write_json(bundle / "execution_plan.json", plan)
    refresh_hashes(bundle)
    assert "cannot auto-launch/advance or waive approval" in M.verify_extension(bundle)["errors"][0]


def test_summary_numbers_must_match_actual_matrix(bundle):
    summary = M.BASE.read_json(bundle, "extension_summary.json")
    summary["n_total_cells"] = 3001
    write_json(bundle / "extension_summary.json", summary)
    refresh_hashes(bundle)
    assert "summary count disagrees" in M.verify_extension(bundle)["errors"][0]


def test_catalog_cannot_redefine_original_policy(bundle):
    catalog = M.BASE.read_json(bundle, "generation_policies_all_13.json")
    catalog["G01"]["parameters"]["temperature"] = .5
    write_json(bundle / "generation_policies_all_13.json", catalog)
    refresh_hashes(bundle)
    assert "Original seven generation catalog entries changed" in M.verify_extension(bundle)["errors"][0]


def test_new_catalog_parameters_match_actual_configs(bundle):
    catalog = M.BASE.read_json(bundle, "generation_policies_all_13.json")
    catalog["G03"]["parameters"]["temperature"] = .5
    write_json(bundle / "generation_policies_all_13.json", catalog)
    refresh_hashes(bundle)
    assert "Generation catalog parameters/stops mismatch" in M.verify_extension(bundle)["errors"][0]


def test_additive_builder_reproduces_published_bytes_from_tracked_inputs(published, tmp_path):
    builder = PATH.with_name("extend_eval_matrix.py")
    if not builder.is_file():
        pytest.skip("Additive builder not present yet")
    source_hashes = {relative: M.BASE.sha256(published / relative) for relative in M.ORIGINAL_SHA256}
    result = subprocess.run(
        [sys.executable, str(builder), "--bundle", str(published), "--output-dir", str(tmp_path)],
        check=False, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    for relative in M.ADDITIVE_FILES:
        assert (tmp_path / relative).read_bytes() == (published / relative).read_bytes(), relative
    assert not any((tmp_path / relative).exists() for relative in M.ORIGINAL_SHA256)
    assert source_hashes == {relative: M.BASE.sha256(published / relative) for relative in M.ORIGINAL_SHA256}
