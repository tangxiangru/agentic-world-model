"""Draft recipe compilation preserves execution and isolates private outcomes."""

import ast
import copy
import json
from dataclasses import replace

import pytest

from tools.outcome_prediction.wm_compile import (
    ContentDecision,
    compile_step,
    compile_steps,
    digest,
    text_digest,
)
from tools.outcome_prediction.wm_recipe_input import digest as recipe_digest


def row():
    return {
        "example_id": "run-a/exp-02",
        "model_input": {
            "task": {"benchmark": "fixture", "base_model": "fixture/base", "evaluation_n": 30},
            "plan": {
                "problem": {"statement": "Investigate how context length affects optimization."},
                "hypothesis": {"claim": "Longer contexts may improve mathematical reasoning."},
                "setup": {
                    "method": {"family": "sft", "hyperparams": {"lr": 1e-5, "top_p": 0.95}},
                    "command": {
                        "script": "train.py",
                        "argv": ["python", "train.py", "--lr", "1e-5", "--top-p", "0.95"],
                    },
                    "data": [
                        {"source": "fixture-corpus", "selection": "verified training examples"}
                    ],
                },
                "evaluation": {"protocol": {"n": 30, "temperature": 0.7, "top_p": 0.95}},
            },
            "code": [
                {
                    "role": "training",
                    "script_path": "train.py",
                    "status": "reconstructed",
                    "content": "lr = 1e-5\ntop_p = 0.95\n",
                }
            ],
        },
        "label": {"accuracy": "PRIVATE_TARGET_SENTINEL"},
        "prior_observations": [{"value": "PRIVATE_HISTORY_SENTINEL"}],
        "known_previous_checkpoints": {"accuracy": "PRIVATE_PREVIOUS_SENTINEL"},
    }


def review(finding, **changes):
    decision = ContentDecision(
        finding_id=finding["finding_id"],
        source_sha256=finding["source_sha256"],
        action="approve",
        reviewer="fixture-human-reviewer",
        evidence_sha256=text_digest("independent review evidence"),
        rationale="Fixture review establishes this is a configuration constant, not an observed result.",
        outcome_free=True,
        executable_recipe_preserved=True,
    )
    return replace(decision, **changes)


def first_finding(result, kind=None):
    return next(f for f in result["audit"]["findings"] if kind is None or f["kind"] == kind)


def test_safe_draft_preserves_numeric_parameters_but_never_approves_content():
    source = row()
    original = copy.deepcopy(source)
    result = compile_step(source)
    assert result["status"] == "draft_clear"
    assert result["draft_input"] == source["model_input"]
    assert result["agent_payload"] is None
    assert result["audit"]["semantic_certificate"] is False
    assert result["audit"]["requires_whole_step_content_review"] is True
    assert result["audit"]["draft_input_sha256"] == recipe_digest(result["draft_input"])
    assert source == original
    assert "PRIVATE_" not in json.dumps(result)


def test_external_results_cannot_influence_compilation_or_audit_hash():
    source = row()
    first = compile_step(source)
    source["label"] = {"accuracy": 0.9}
    source["result"] = {"measurements": [{"value": 0.99}]}
    source["prior_observations"] = [{"value": 0.8}]
    source["known_previous_checkpoints"] = {"accuracy": 0.7}
    assert compile_step(source) == first


def test_compiler_never_reads_non_feature_row_fields():
    class GuardedRow(dict):
        def __getitem__(self, key):
            assert key in {"example_id", "model_input"}
            return super().__getitem__(key)

        def __iter__(self):
            raise AssertionError("Whole row traversal is forbidden")

    assert compile_step(GuardedRow(row()))["status"] == "draft_clear"


def test_structured_observation_mutation_does_not_change_draft():
    source = row()
    plan = source["model_input"]["plan"]
    plan["problem"].update(
        {
            "evidence": [{"observation": "accuracy 0.81"}],
            "failure_examples": [{"gold": "37"}],
            "affected_share": 0.5,
        }
    )
    plan["evaluation"]["comparator"] = {
        "ref": "exp-01",
        "value": 0.81,
        "path": "old_metric.json",
        "stderr": 0.03,
    }
    plan["evaluation"]["diagnostic"] = {"what": "prior checkpoint terminated 7/30 times"}
    plan["result"] = {"measurements": [{"accuracy": 0.81}]}
    a = compile_step(source)
    plan["problem"]["evidence"][0]["observation"] = "accuracy 0.19"
    plan["evaluation"]["comparator"]["value"] = 0
    plan["evaluation"]["diagnostic"]["what"] = "different measured outcome"
    plan["result"]["measurements"][0]["accuracy"] = 0.99
    b = compile_step(source)
    assert a["draft_input"] == b["draft_input"]
    assert a["draft_input"]["plan"]["evaluation"]["comparator"] == {"ref": "exp-01"}
    assert len(a["audit"]["removed"]) == 8
    assert a["audit"]["source_input_sha256"] != b["audit"]["source_input_sha256"]
    assert "0.81" not in json.dumps(a["draft_input"])


@pytest.mark.parametrize(
    "text",
    [
        "exp-01 scored 0.75 on the benchmark.",
        "The parent model collapsed during evaluation.",
        "Accuracy: 85%; try changing the prompt next.",
        "The incumbent obtained 21/30 on the benchmark.",
    ],
)
def test_explicit_narrative_results_removed_with_exact_private_provenance(text):
    source = row()
    source["model_input"]["plan"]["problem"]["statement"] = text
    result = compile_step(source)
    assert "statement" not in result["draft_input"]["plan"]["problem"]
    removal = next(r for r in result["audit"]["removed"] if r["path"] == "/plan/problem/statement")
    assert removal["content"] == text
    assert removal["source_sha256"] == digest(text)
    assert removal["spans"] == [{"start": 0, "end": len(text), "sha256": text_digest(text)}]


def test_possible_implicit_narrative_is_not_silently_approved():
    source = row()
    text = "The three decode attempts were neutral or harmful; the ceiling remains."
    source["model_input"]["plan"]["hypothesis"]["mechanism"] = text
    result = compile_step(source)
    assert result["status"] == "needs_review"
    assert result["draft_input"]["plan"]["hypothesis"]["mechanism"] == text
    assert any(f["kind"] == "implicit_outcome_reference" for f in result["audit"]["findings"])


def test_operational_prose_with_outcomes_is_flagged_but_not_deleted():
    source = row()
    source["model_input"]["plan"]["setup"]["method"]["notes"] = (
        "Parent accuracy was 0.83; lr=1e-5 and top_p=0.95."
    )
    result = compile_step(source)
    assert result["status"] == "needs_review"
    assert result["draft_input"]["plan"]["setup"] == source["model_input"]["plan"]["setup"]
    assert result["audit"]["removed"] == []
    finding = first_finding(result, "explicit_metric_literal")
    assert finding["path"] == "/plan/setup/method/notes"
    text = source["model_input"]["plan"]["setup"]["method"]["notes"]
    for span in finding["spans"]:
        assert text_digest(text[span["start"] : span["end"]]) == span["sha256"]


@pytest.mark.parametrize(
    "code",
    [
        "# exp-01 accuracy: 0.8\nlr=1e-5\n",
        '"""Fixing the over-generation that collapsed exp-02."""\nlr=1e-5\n',
        'notes = "parent scored 81%"\n',
        "baseline_accuracy = 0.81\n",
        'history = {"accuracy": 0.81}\n',
        "accuracy = 0\n",
    ],
)
def test_possible_code_results_stay_byte_exact_and_require_review(code):
    source = row()
    source["model_input"]["code"][0]["content"] = code
    result = compile_step(source)
    assert result["status"] == "needs_review"
    assert result["draft_input"]["code"][0]["content"] == code
    assert result["audit"]["findings"]
    assert result["audit"]["removed"] == []


@pytest.mark.parametrize(
    "code",
    [
        "accuracy = correct / total\n",
        'print(f"accuracy={correct/total:.3f}")\n',
        "loss = model(batch).loss\nloss.backward()\n",
        "top_p=0.95\nlr=1e-5\nepochs=2\n",
        'metrics = {"accuracy": correct / total}\n',
    ],
)
def test_dynamic_metric_calculation_is_not_a_measured_result_literal(code):
    source = row()
    source["model_input"]["code"][0]["content"] = code
    result = compile_step(source)
    assert result["status"] == "draft_clear"
    assert result["audit"]["findings"] == []


def test_operational_structured_result_requires_explicit_manual_removal():
    source = row()
    source["model_input"]["plan"]["setup"]["observed_accuracy"] = 0.0
    result = compile_step(source)
    assert result["draft_input"]["plan"]["setup"]["observed_accuracy"] == 0.0
    finding = first_finding(result, "outcome_field_in_operational_setup")
    result = compile_step(source, decisions=[review(finding, action="remove_field")])
    assert "observed_accuracy" not in result["draft_input"]["plan"]["setup"]
    assert result["status"] == "draft_clear"
    assert result["agent_payload"] is None


def test_manual_approval_is_bound_to_exact_content_and_does_not_certify_step():
    source = row()
    source["model_input"]["plan"]["setup"]["threshold"] = "target accuracy: 0.9"
    finding = first_finding(compile_step(source))
    approved = compile_step(source, decisions=[review(finding)])
    assert approved["status"] == "draft_clear"
    assert approved["agent_payload"] is None
    assert approved["audit"]["semantic_certificate"] is False
    assert approved["audit"]["requires_whole_step_content_review"] is True
    source["model_input"]["plan"]["setup"]["threshold"] = "target accuracy: 0.8"
    with pytest.raises(ValueError, match="stale finding"):
        compile_step(source, decisions=[review(finding)])


def test_exact_comment_redaction_preserves_executable_ast_and_hyperparameters():
    source = row()
    code = "# parent accuracy: 0.8\nlr = 1e-5\ntop_p = 0.95\n"
    source["model_input"]["code"][0]["content"] = code
    result = compile_step(source)
    finding = first_finding(result, "explicit_metric_literal")
    end = code.index("\n")
    decision = review(
        finding, action="redact", start=0, end=end, span_sha256=text_digest(code[:end])
    )
    changed = compile_step(source, decisions=[decision])
    new_code = changed["draft_input"]["code"][0]["content"]
    assert new_code == "\nlr = 1e-5\ntop_p = 0.95\n"
    assert ast.dump(ast.parse(new_code)) == ast.dump(ast.parse(code))
    assert changed["status"] == "draft_clear"
    assert changed["audit"]["removed"][0]["content"] == code
    assert changed["agent_payload"] is None


def test_manual_redaction_rescans_remaining_results():
    source = row()
    code = "# accuracy: 0.8\n# parent model collapsed\nlr=1e-5\n"
    source["model_input"]["code"][0]["content"] = code
    finding = first_finding(compile_step(source), "explicit_metric_literal")
    end = code.index("\n")
    decision = review(
        finding, action="redact", start=0, end=end, span_sha256=text_digest(code[:end])
    )
    changed = compile_step(source, decisions=[decision])
    assert changed["status"] == "needs_review"
    assert any(
        f["kind"] == "past_outcome_language" for f in changed["audit"]["unresolved_findings"]
    )


def test_python_unicode_ast_spans_use_character_not_byte_offsets():
    source = row()
    code = 'name = "λ"; baseline_accuracy = 0.8\n'
    source["model_input"]["code"][0]["content"] = code
    finding = first_finding(compile_step(source), "static_metric_assignment")
    span = finding["spans"][0]
    assert code[span["start"] : span["end"]] == "baseline_accuracy = 0.8"
    assert span["sha256"] == text_digest("baseline_accuracy = 0.8")


@pytest.mark.parametrize(
    "changes",
    [
        {"reviewer": ""},
        {"evidence_sha256": "missing"},
        {"outcome_free": False},
        {"executable_recipe_preserved": False},
        {"source_sha256": "a" * 64},
        {"start": 0},
    ],
)
def test_manual_decision_guards(changes):
    source = row()
    source["model_input"]["plan"]["setup"]["notes"] = "accuracy: 0.8"
    finding = first_finding(compile_step(source))
    with pytest.raises(ValueError):
        compile_step(source, decisions=[review(finding, **changes)])


def test_bad_redaction_hash_and_untyped_decisions_rejected():
    source = row()
    source["model_input"]["plan"]["setup"]["notes"] = "accuracy: 0.8"
    finding = first_finding(compile_step(source))
    decision = review(finding, action="redact", start=0, end=3, span_sha256="a" * 64)
    with pytest.raises(ValueError, match="span bounds/hash"):
        compile_step(source, decisions=[decision])
    with pytest.raises(TypeError, match="typed"):
        compile_step(source, decisions=[{"action": "approve"}])


def test_code_path_can_carry_a_possible_result_too():
    source = row()
    source["model_input"]["code"][0]["script_path"] = "train-acc95.py"
    result = compile_step(source)
    assert result["status"] == "needs_review"
    assert any(f["path"] == "/code/0/script_path" for f in result["audit"]["findings"])


def test_compilation_inventory_and_duplicate_unknown_decision_guardrails():
    first, second = row(), row()
    second["example_id"] = "run-b/exp-01"
    results, audit = compile_steps([first, second])
    assert audit["steps"] == 2
    assert audit["counts"] == {"draft_clear": 2}
    assert all(r["agent_payload"] is None for r in results)
    with pytest.raises(ValueError, match="unique"):
        compile_steps([first, first])
    with pytest.raises(ValueError, match="unknown step"):
        compile_steps([first], decisions_by_step={"missing": []})
