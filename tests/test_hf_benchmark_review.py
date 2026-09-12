"""Independent edge cases for benchmark source, split, and export integrity."""

import copy
import json

import pytest

from tools.outcome_prediction.hf_benchmark import (
    assign_groups,
    choose_representatives,
    digest,
    extract_serving,
    file_digest,
    verify,
)


def test_unscored_ancestor_retains_transitive_weight_group():
    rows = [
        {"checkpoint_id": "root", "session_id": "source-run", "source_split": "locked_test"},
        {"checkpoint_id": "leaf", "session_id": "leaf-run"},
    ]
    scripts = {
        "root": {"session_id": "source-run", "parent_checkpoint_ids": []},
        "unscored": {"session_id": "middle-run", "parent_checkpoint_ids": ["root"]},
        "leaf": {"session_id": "leaf-run", "parent_checkpoint_ids": ["unscored"]},
    }
    assign_groups(rows, scripts)
    assert {r["split"] for r in rows} == {"test"}
    assert len({r["group_id"] for r in rows}) == 1


def test_different_evaluator_protocols_are_not_duplicate_cells():
    common = {
        "track": "ptb_controlled",
        "benchmark": "aime2025",
        "weights_sha256": "same-weights",
        "serving_sha256": "same-serving",
        "exclusion_reasons": [],
        "status": "eligible",
    }
    rows = [
        {**copy.deepcopy(common), "example_id": "a", "protocol_fingerprint": "evaluator-v1"},
        {**copy.deepcopy(common), "example_id": "b", "protocol_fingerprint": "evaluator-v2"},
    ]
    choose_representatives(rows)
    assert all(row["status"] == "eligible" for row in rows)


def test_same_weights_with_different_model_artifacts_are_not_duplicate_cells():
    common = {
        "track": "ptb_controlled",
        "benchmark": "gsm8k",
        "weights_sha256": "same-weights",
        "serving_sha256": "same-sampling-and-template",
        "protocol_fingerprint": "same-evaluator",
        "exclusion_reasons": [],
        "status": "eligible",
    }
    rows = [
        {
            **copy.deepcopy(common),
            "example_id": "a",
            "model_artifacts_fingerprint": "config-with-eos",
        },
        {
            **copy.deepcopy(common),
            "example_id": "b",
            "model_artifacts_fingerprint": "config-without-eos",
        },
    ]
    choose_representatives(rows)
    assert all(row["status"] == "eligible" for row in rows)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_duplicate_exported_label_ids_fail_verification(tmp_path):
    payload = {
        "experiment_scripts": {"files": [{"path": "train.py", "content": "pass"}], "launch": {}},
        "serving_config": {},
    }
    row = {
        "example_id": "one",
        "status": "eligible",
        "split": "train",
        "input_sha256": digest(payload),
        "group_id": "group-one",
        "checkpoint_id": "checkpoint",
    }
    label = {"example_id": "one", "status": "complete", "y": 0.5, "run_accuracies": [0.5] * 10}
    write_rows(tmp_path / "audit/registry.jsonl", [row])
    for split in ["train", "validation", "test"]:
        write_rows(
            tmp_path / f"inputs/{split}.jsonl",
            [{"example_id": "one", "x": payload}] if split == "train" else [],
        )
        write_rows(tmp_path / f"labels/{split}.jsonl", [label, label] if split == "train" else [])
    assert not verify(tmp_path)["passed"]


def serving_fixture(root):
    template = root / "rescore10/eval/templates/qwen3.jinja"
    template.parent.mkdir(parents=True)
    template.write_text("{{ messages }}")
    model = root / "checkpoints_meta/checkpoint/config.json"
    model.parent.mkdir(parents=True)
    model.write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "eos_token_id": 151645}))
    return {
        "protocol": {"seed_formula": "deterministic per question and repeat"},
        "eval_matrix_1k": {
            "preflight": {"family": "qwen", "file_sha256": {"config.json": file_digest(model)}},
            "digests": {"chat_template_sha256": file_digest(template)},
            "server": {
                "generation_config": "vllm",
                "dtype": "bfloat16",
                "cli_args_logged": "non-default args: {'max_model_len': 20480}",
                "resolved_sampling_params": {
                    "n": "1",
                    "temperature": "1.0",
                    "top_k": "0",
                    "top_p": "1.0",
                    "min_p": "0.0",
                    "presence_penalty": "0.0",
                    "frequency_penalty": "0.0",
                    "repetition_penalty": "1.0",
                    "max_tokens": "4000",
                    "min_tokens": "0",
                    "stop": "[]",
                    "stop_token_ids": "[151643, 151645]",
                    "ignore_eos": "False",
                },
            },
        },
    }


@pytest.mark.parametrize(
    "key,value", [("n", "0"), ("max_tokens", "-1"), ("top_p", "1.5"), ("ignore_eos", "17")]
)
def test_impossible_server_resolved_settings_fail_eligibility(tmp_path, key, value):
    data = serving_fixture(tmp_path)
    assert extract_serving(data, tmp_path, "checkpoint", "aime2025")["status"] == "eligible"
    data["eval_matrix_1k"]["server"]["resolved_sampling_params"][key] = value
    result = extract_serving(data, tmp_path, "checkpoint", "aime2025")
    assert result["status"] == "candidate"
