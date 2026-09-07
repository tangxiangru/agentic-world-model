import copy

import pytest

from tools.outcome_prediction import wm_rich_parameter_pilot as pilot


class NoOutcomes(dict):
    def __getitem__(self, key):
        if key in {
            "label",
            "prior_observations",
            "known_previous_checkpoints",
            "problem",
            "hypothesis",
            "result",
            "conclusion",
        }:
            raise AssertionError("Outcome-bearing field accessed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


def row(code=""):
    return NoOutcomes(
        {
            "role": "training",
            "label": object(),
            "model_input": NoOutcomes(
                {
                    "known_previous_checkpoints": object(),
                    "plan": NoOutcomes(
                        {
                            "problem": object(),
                            "hypothesis": object(),
                            "setup": {
                                "method": {"family": "sft", "peft": "lora"},
                                "command": {
                                    "argv": [
                                        "python",
                                        "train.py",
                                        "--lr",
                                        "1e-5",
                                        "--top-p",
                                        "0.9",
                                        "--optimizer",
                                        "adamw_8bit",
                                    ]
                                },
                                "data": [
                                    {
                                        "source": "open-r1/OpenR1-Math-220k",
                                        "built_by": "build.py",
                                        "build_command": [
                                            "python",
                                            "build.py",
                                            "--max-examples",
                                            "12000",
                                        ],
                                        "n_examples": object(),
                                    }
                                ],
                            },
                        }
                    ),
                    "code": [{"role": "training", "status": "reconstructed", "content": code}],
                }
            ),
        }
    )


def test_outcomes_comments_docstrings_logs_and_arbitrary_constants_not_features():
    code = '''
"""Parent accuracy 0.91; 27/30 correct."""
# parent failed at 0.01
parent_accuracy = 0.91
LOSS_CHUNK = 2048
loss_acc = 0.0
print("official score", parent_accuracy, SFTConfig(learning_rate=0.333))
cfg = SFTConfig(learning_rate=1e-5, packing=True, optim="adamw_8bit")
'''
    first, _ = pilot.step_features(row(code))
    mutated = (
        code.replace("0.91", "0.37")
        .replace("27/30", "11/30")
        .replace("0.01", "0.87")
        .replace("2048", "9999")
        .replace("0.333", "0.777")
    )
    second, _ = pilot.step_features(row(mutated))
    assert first == second
    assert not any("0.333" in key or "parent_accuracy" in key or "2048" in key for key in first)
    assert first["command.--lr"] == 1e-5
    assert first["command.--top-p"] == 0.9
    assert first["syntax.training.SFTConfig.learning_rate.literal=1e-05"] == 1.0


def test_lora_and_optimizer_operand_features_are_positive_and_bounded():
    features, _ = pilot.step_features(
        row(
            'cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=["q_proj","v_proj"], parent_accuracy=0.9)\nopt=AdamW(params, lr=args.lr)\n'
        )
    )
    assert features["declared.peft"] == "lora"
    assert features["syntax.training.LoraConfig.target_module=q_proj"] == 1.0
    assert features["syntax.training.LoraConfig.r.literal=16"] == 1.0
    assert features["syntax.training.AdamW.present"] == 1.0
    assert not any("0.9" in key or "args.lr" in key for key in features)


def test_quota_is_explicit_builder_cap_not_realized_count():
    sample = row()
    features, _ = pilot.step_features(sample)
    assert features["data[0].planned_cap.--max-examples"] == 12000
    sample["model_input"]["plan"]["setup"]["data"][0]["n_examples"] = 45678
    assert pilot.step_features(sample)[0] == features
    sample["model_input"]["plan"]["setup"]["data"][0]["build_command"] = [
        "python",
        "evaluate.py",
        "--limit",
        "30",
    ]
    assert not any("planned_cap" in key for key in pilot.step_features(sample)[0])


def test_unknown_dataset_and_shell_builder_have_no_quota():
    sample = row()
    entry = sample["model_input"]["plan"]["setup"]["data"][0]
    entry["build_command"] = ["python", "build.py", "--max-examples", "100", "&&", "touch", "x"]
    assert not any("planned_cap" in key for key in pilot.step_features(sample)[0])
    entry["source"] = "synthetic:self based on open-r1/OpenR1-Math-220k"
    assert not any(key.startswith("data[") for key in pilot.step_features(sample)[0])


def test_self_generated_graph_false_negative_cannot_be_readmitted():
    sample = row()
    sample["example_id"] = "r0-11/exp-04"
    entry = sample["model_input"]["plan"]["setup"]["data"][0]
    entry["source"] = "synthetic:self sft_out_e3/sft_out_rft"
    entry["build_command"] = ["python", "build.py", "--rft", "sft_out_rft"]
    frozen = [{"example_id": sample["example_id"], "closure_ids_private": [sample["example_id"]]}]
    with pytest.raises(ValueError, match="self-generated"):
        pilot.richer_cohort(frozen, [sample])


def test_ordered_closure_and_original_features_preserved_without_ancestor_labels():
    first, second = row(), row("cfg=LoraConfig(r=8)")
    first["example_id"], second["example_id"] = "run/exp-01", "run/exp-02"
    item = {
        "example_id": "run/exp-02",
        "closure_ids_private": ["run/exp-01", "run/exp-02"],
        "features": {"closure_steps": 2.0},
        "last_step_features": {"lr": 1e-5},
        "target": 0.5,
    }
    rich, _ = pilot.richer_cohort([item], [first, second])
    assert rich[0]["features"]["closure_steps"] == 2.0
    assert rich[0]["features"]["step_1.rich.syntax.training.LoraConfig.r.literal=8"] == 1.0
    assert rich[0]["target"] == item["target"]
    assert item == {
        "example_id": "run/exp-02",
        "closure_ids_private": ["run/exp-01", "run/exp-02"],
        "features": {"closure_steps": 2.0},
        "last_step_features": {"lr": 1e-5},
        "target": 0.5,
    }


def test_unknown_code_never_uses_reconstruction_status_as_feature():
    sample = row("not valid python !!!")
    features, audit = pilot.step_features(sample)
    assert not any(key.startswith("syntax") for key in features)
    assert any(item["reason"] == "unparseable_python" for item in audit["omissions"])
    changed = copy.deepcopy(sample)
    changed["model_input"]["code"][0]["status"] = "not_reconstructed"
    assert pilot.step_features(changed)[0] == features
