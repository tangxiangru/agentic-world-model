import copy
import json

import pytest

from tools.outcome_prediction import wm_fresh_llm_baseline as llm
from tools.outcome_prediction.wm_one_step_features import features


def row(key="train/one", cell="train", split="train", benchmark="gsm8k", parent=0.1):
    source = {"parent": {"accuracy": parent, "kind": "published_base", "reference_known": True},
              "model_input": {}, "history": [], "history_complete_to_base": True}
    return {"example_id": key, "cell_id": cell, "split": split, "benchmark": benchmark,
            "parent_reference": parent, "reference_kind": "published_base",
            "views": {"history": features(source, "history")}}


def test_schema_is_numeric151_and_projection_excludes_extraneous_input():
    value = row()
    value.update(scientist_model="SECRET", arbitrary_text="SECRET", target_accuracy=0.88888)
    value["views"]["history"]["parent.accuracy"] = 0.1
    order = llm.schema_keys()
    prompt = llm.prompt_for(value, [], {}, order)
    payload = json.loads(prompt)
    assert len(order) == 151
    assert all(type(x) is float for x in payload["target_features"])
    assert payload["labeled_training_demonstrations"] == []
    assert all(x not in prompt for x in ("SECRET", "train/one", "target_accuracy", "0.88888"))


@pytest.mark.parametrize("bad", [None, "0.1", True, float("nan"), float("inf")])
def test_no_nonnumeric_or_missing_prompt_feature(bad):
    value = row()
    value["views"]["history"]["parent.accuracy"] = bad
    with pytest.raises(ValueError):
        llm.safe_vector(value, llm.schema_keys())


def test_unknown_feature_fails_closed():
    value = row()
    value["views"]["history"]["current.accuracy"] = 0.4
    with pytest.raises(ValueError, match="schema"):
        llm.safe_vector(value, llm.schema_keys())


def test_bank_deterministic_session_round_robin_benchmark_only():
    train = [row(f"s{s}/e{e}", f"s{s}") for s in range(5) for e in range(4)]
    train.append(row("aime/e1", "aime", benchmark="aime2025"))
    bank = llm.select_bank(train, "gsm8k", 7)
    assert bank == llm.select_bank(list(reversed(train)), "gsm8k", 7)
    assert len(bank) == 7
    assert len({r["cell_id"] for r in bank[:5]}) == 5
    assert all(r["benchmark"] == "gsm8k" for r in bank)


def test_test_rows_cannot_enter_bank():
    with pytest.raises(ValueError, match="training"):
        llm.select_bank([row(split="test")], "gsm8k")


def test_only_bank_label_values_are_projected():
    train, target = row(), row("SECRET/TEST", "SECRET", "test")
    label = {train["example_id"]: {"accuracy": 0.0, "delta_accuracy": -0.1,
                                   "raw_outcomes": "FORBIDDEN"}}
    prompt = llm.prompt_for(target, [train], label, llm.schema_keys())
    data = json.loads(prompt)
    assert data["labeled_training_demonstrations"][0]["official_accuracy"] == 0.0
    assert "FORBIDDEN" not in prompt and "SECRET" not in prompt
    altered = copy.deepcopy(label)
    altered["SECRET/TEST"] = {"accuracy": 0.987654321}
    assert llm.prompt_for(target, [train], altered, llm.schema_keys()) == prompt


@pytest.mark.parametrize("bad", [None, True, "0.2", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_predictions_rejected_no_imputation(bad):
    with pytest.raises(ValueError):
        llm.parse_prediction(json.dumps({"predicted_accuracy": bad, "rationale": "test"}))


@pytest.mark.parametrize("valid", [0, 0.5, 1])
def test_valid_zero_and_bounds_retained(valid):
    output = llm.parse_prediction(json.dumps({"predicted_accuracy": valid, "rationale": "test"}))
    assert output["predicted_accuracy"] == valid


def test_no_extra_prose_or_extra_json():
    with pytest.raises(ValueError):
        llm.parse_prediction('Forecast: {"predicted_accuracy":0.1,"rationale":"x"}')
    assert llm.parse_prediction('```json\n{"predicted_accuracy":0.1,"rationale":"x"}\n```')["predicted_accuracy"] == 0.1


@pytest.mark.parametrize("payload", ["null", "[]", "0.3", '"text"'])
def test_non_object_json_rejected(payload):
    with pytest.raises(TypeError, match="JSON object"):
        llm.parse_prediction(payload)


def test_model_call_has_no_data_tools_or_repository_working_directory(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)
        events = [
            {"type": "system", "subtype": "init", "model": "claude-opus-5", "tools": [], "mcp_servers": []},
            {"type": "assistant", "message": {"model": "claude-opus-5", "content": []}},
            {"type": "result", "is_error": False, "result": '{"predicted_accuracy":0,"rationale":"test"}', "total_cost_usd": 0.01},
        ]
        return type("Response", (), {"stdout": "\n".join(json.dumps(e) for e in events), "stderr": "", "returncode": 0})()

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    policy = {"effort": "high", "per_call_budget_usd": 0.5, "model_requested": "claude-opus-5",
              "system": llm.SYSTEM, "max_output_tokens": 2048, "max_cli_network_retries": 1, "timeout_seconds": 240}
    result = llm.call_model("safe-prompt", policy)
    assert result["error"] is None and result["prediction"]["predicted_accuracy"] == 0
    cmd = captured["command"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--safe-mode" in cmd and "--strict-mcp-config" in cmd
    assert "agentic-world-model" not in captured["cwd"]
    assert captured["input"] == "safe-prompt"
    assert captured["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "2048"
