"""Dojo checkpoint/script binding without executing archived experiments."""

import hashlib
import json

from tools.outcome_prediction.hf_benchmark_dojo import extract_dojo_scripts

SCRIPT = '''"""Previous measured accuracy: 0.99."""
from transformers import AutoModelForCausalLM
# parent dev accuracy 0.75
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Base")
model.save_pretrained("./checkpoint")
'''


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, *, script=SCRIPT, best_step=2):
    run_dir = root / "dojo_ab_gsm8k/rpm/seed01"
    checkpoint = "/results/step002_123"
    run = {
        "job": 1707,
        "selector": "rpm",
        "seed": 1,
        "rescore10_id": "abgsm8k-rpm-s01",
        "best_checkpoint_gs": "gs://archive/abgsm8k-rpm-s01",
        "best": [0.8, checkpoint, best_step],
        # Deliberately disagree: the extractor must bind the export declaration,
        # never reconstruct selection by sorting potentially incomplete scores.
        "steps_scored": [{"step": 1, "dev_acc": 0.95}, {"step": 2, "dev_acc": 0.8}],
    }
    write_json(run_dir / "run.json", run)
    write_json(
        run_dir / "dojo_config.json",
        {
            "task": {"base_model": "Qwen/Qwen3-4B-Base"},
            "metadata": {"slurm_id": "147"},
            "interpreter": {"working_dir": "/old/147"},
            "contract": "python solution.py",
        },
    )
    step = run_dir / "artifacts/step002"
    write_json(
        step / "meta.json",
        {
            "run": "1707",
            "step": 2,
            "checkpoint": checkpoint,
            "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
            "parent_step": 1,
            "parent_dev_acc": 0.95,
            "exit_code": 0,
            "is_buggy": False,
            "dev_acc": 0.8,
            "analysis": "Measured outcome must never be part of the model input.",
        },
    )
    (step / "solution.py").write_text(script)
    return run_dir


def record(root):
    return extract_dojo_scripts(root)["abgsm8k-rpm-s01"]


def test_exact_selected_step_is_bound_without_score_sorting(tmp_path):
    fixture(tmp_path)
    result = record(tmp_path)
    assert result["status"] == "eligible"
    assert result["provenance"]["selected_step"] == 2
    assert len(result["scripts"]) == 1
    assert result["parent_checkpoint_ids"] == []
    assert result["provenance"]["search_parent_step"] == 1


def test_comments_scores_and_stale_launch_config_are_not_inputs(tmp_path):
    fixture(tmp_path)
    result = record(tmp_path)
    inputs = json.dumps({"scripts": result["scripts"], "launch": result["launch"]})
    assert "0.99" not in inputs and "0.75" not in inputs and "0.8" not in inputs
    assert "Measured outcome" not in inputs and "/old/147" not in inputs
    assert result["launch"]["cwd"] is None
    assert "stale_config_working_directory_omitted" in result["review_flags"]
    assert "semantic_review_not_exhaustive" in result["review_flags"]


def test_missing_explicit_best_does_not_fall_back_to_argmax(tmp_path):
    run_dir = fixture(tmp_path)
    path = run_dir / "run.json"
    run = json.loads(path.read_text())
    del run["best"]
    write_json(path, run)
    result = record(tmp_path)
    assert result["status"] == "candidate"
    assert result["scripts"] == []
    assert "explicit_best_checkpoint_step_missing" in result["exclusion_reasons"]


def test_checkpoint_conflict_is_not_usable(tmp_path):
    run_dir = fixture(tmp_path)
    path = run_dir / "artifacts/step002/meta.json"
    meta = json.loads(path.read_text())
    meta["checkpoint"] = "/results/different"
    write_json(path, meta)
    result = record(tmp_path)
    assert result["status"] == "candidate"
    assert "selected_checkpoint_step_metadata_conflict" in result["exclusion_reasons"]


def test_script_tamper_is_not_usable(tmp_path):
    run_dir = fixture(tmp_path)
    (run_dir / "artifacts/step002/solution.py").write_text(SCRIPT + "\nprint('changed')\n")
    result = record(tmp_path)
    assert result["status"] == "candidate"
    assert "selected_script_checksum_conflict_or_missing" in result["exclusion_reasons"]


def test_surviving_executable_outcome_literal_requires_review(tmp_path):
    fixture(tmp_path, script=SCRIPT + '\nprior_observation = "accuracy: 0.94"\n')
    result = record(tmp_path)
    assert result["status"] == "candidate"
    assert "script_content_review_required" in result["exclusion_reasons"]


def test_runtime_accuracy_lookup_and_math_percent_are_preserved(tmp_path):
    script = (
        SCRIPT
        + """
accuracy = results.get("accuracy", 0.0)
question = "A shop gives a 25% discount."
divisible = 10 % 3 == 0
"""
    )
    fixture(tmp_path, script=script)
    result = record(tmp_path)
    assert result["status"] == "eligible"
    content = result["scripts"][0]["content"]
    assert 'results.get("accuracy", 0.0)' in content
    assert "25%" in content


def test_zero_accuracy_observation_is_still_reviewed(tmp_path):
    fixture(tmp_path, script=SCRIPT + '\nprior_observation = "accuracy: 0.0"\n')
    result = record(tmp_path)
    assert result["status"] == "candidate"
    assert "script_content_review_required" in result["exclusion_reasons"]


def test_string_that_describes_runtime_syntax_is_not_whitelisted(tmp_path):
    fixture(tmp_path, script=SCRIPT + "\nprior = \"accuracy: results.get('accuracy', 0.0)\"\n")
    assert record(tmp_path)["status"] == "candidate"


def test_unresolved_learned_weight_input_is_not_silently_base(tmp_path):
    fixture(tmp_path, script=SCRIPT.replace('"Qwen/Qwen3-4B-Base"', '"/models/earlier_checkpoint"'))
    result = record(tmp_path)
    assert result["status"] == "candidate"
    assert "script_weight_inputs_need_resolution" in result["exclusion_reasons"]


def test_runs_do_not_expand_to_unlabelled_steps_and_need_no_label_reads(tmp_path):
    run_dir = fixture(tmp_path)
    extra = run_dir / "artifacts/step001"
    extra.mkdir()
    (extra / "solution.py").write_text(SCRIPT)
    assert len(extract_dojo_scripts(tmp_path)) == 1
    assert record(tmp_path)["status"] == "eligible"


def test_missing_script_preserves_candidate_inventory_row(tmp_path):
    run_dir = fixture(tmp_path)
    (run_dir / "artifacts/step002/solution.py").unlink()
    result = record(tmp_path)
    assert result["status"] == "candidate"
    assert "selected_step_script_missing_or_invalid" in result["exclusion_reasons"]
