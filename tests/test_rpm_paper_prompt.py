"""Prompt fidelity and strict hard-choice decoding; never invoke a model."""

import copy
import json
from pathlib import Path

import pytest

from tools.outcome_prediction import rpm_paper_prompt as prompt


@pytest.fixture
def payload():
    return {
        "task": "Rank immediate official GSM8K accuracy.",
        "candidate_A": {
            "earliest_plan": {"hypothesis": "Retain 0.05 warmup and parent score 0.483333."},
            "code": "lr = 0.0001\nweight_decay = 0.01\n",
        },
        "candidate_B": {"full_plan": "Use 20% more data.", "code": {"train.py": "epochs = 2"}},
        "parent_checkpoint": {"observed_accuracy": 0.71, "metric_scope": "local 200"},
        "observed_current_run_history": [{"score": 0.709, "plan": "previous experiment"}],
    }


def test_eventual_wording_matches_local_paper_when_available():
    source = Path(__file__).resolve().parents[1] / "data/analysis/rpm/paper/2608.13940.txt"
    if not source.exists():
        pytest.skip("Downloaded paper text is not required for normal repository tests")
    paper = source.read_text()
    beginning = "You are a principal investigator allocating compute budget to one of two branches."
    excerpt = paper[paper.index(beginning) :]
    excerpt = excerpt[: excerpt.index("Figure 7")]
    # The source spans pages 17 and 18; remove only the intervening page number.
    excerpt = excerpt.replace("17\n\f", "")
    assert " ".join(excerpt.split()) == " ".join(prompt.EVENTUAL_TEMPLATE.split())


def test_eventual_render_has_exact_paper_policy_and_no_added_output_constraints(payload):
    rendered = prompt.render_prompt(payload, target="eventual")
    assert "best eventual test score after several iterations" in rendered
    assert "Bugs are acceptable if the approach is sound and fixes are clear" in rendered
    assert "higher expected long-term best test score" in rendered
    assert rendered.endswith(r"Provide your answer inside a \boxed{A} or \boxed{B}.")
    assert "200 words" not in rendered
    context = json.dumps(
        {
            key: value
            for key, value in payload.items()
            if key not in {"task", "candidate_A", "candidate_B"}
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    assert rendered.count(context) == 2  # Figure 7 intentionally repeats context.


def test_immediate_is_default_and_does_not_reward_future_fixes(payload):
    rendered = prompt.render_prompt(payload)
    assert rendered == prompt.render_prompt(payload, target="immediate")
    assert "better immediate test score" in rendered
    assert "penalize bugs that affect this execution, even if fixes are clear" in rendered
    assert "do not assume future repairs" in rendered
    for obsolete in (
        "best eventual test score",
        "higher expected long-term best test score",
        "in 1 to 3 iterations",
        "over the next few iterations",
        "Bugs are acceptable",
        "penalize only for hard-to-remedy",
    ):
        assert obsolete not in rendered
    assert rendered.endswith(r"Provide your answer inside a \boxed{A} or \boxed{B}.")


def test_immediate_changes_are_only_documented_substitutions():
    expected = prompt.EVENTUAL_TEMPLATE
    for old, new in prompt.IMMEDIATE_SUBSTITUTIONS:
        assert expected.count(old) == 1
        expected = expected.replace(old, new, 1)
    assert prompt.IMMEDIATE_TEMPLATE == expected


@pytest.mark.parametrize(
    "plan_key", ["earliest_plan", "full_plan", "plan", "plan_time_setup", "setup"]
)
@pytest.mark.parametrize(
    "code_key", ["earliest_code", "code", "code_files", "launch_scripts", "source_code"]
)
def test_flexible_plan_and_code_aliases_preserve_structured_contents(payload, plan_key, code_key):
    plan = {"hypothesis": "parent measured 0.483333; lr 0.0001", "epochs": 2}
    code = {"train.py": "warmup=0.05\ntop_p=0.95"}
    payload["candidate_A"] = {plan_key: plan, code_key: code}
    rendered = prompt.render_prompt(payload)
    assert json.dumps(plan, sort_keys=True, indent=2) in rendered
    assert json.dumps(code, sort_keys=True, indent=2) in rendered


def test_no_redaction_truncation_recursive_replacement_or_input_mutation(payload):
    prose = r"Full plan includes {context_text}, {plan_B}, \boxed{B}, 0.01, 0.483333, and 48.33%."
    code = "# " + "long_code " * 3000 + "\nmodel = {'lr': 0.0001, 'warmup': 0.05}\n"
    payload["candidate_A"] = {"earliest_plan": prose, "code": code}
    before = copy.deepcopy(payload)
    rendered = prompt.render_prompt(payload)
    assert prose in rendered
    assert code in rendered
    assert payload == before
    assert "[n]" not in rendered


def test_multiple_representations_and_extra_candidate_fields_are_not_dropped(payload):
    payload["candidate_A"] = {
        "earliest_plan": "first plan",
        "setup": {"batch_size": 8},
        "code": "first code",
        "code_files": {"train.py": "second code"},
        "recipe_sequence": [{"parent_kind": "base_model"}],
    }
    rendered = prompt.render_prompt(payload)
    plan_section = rendered.split("Candidate A -- Plan:\n", 1)[1].split("Candidate A -- Code:", 1)[
        0
    ]
    code_section = rendered.split("Candidate A -- Code:\n", 1)[1].split("Candidate B -- Plan:", 1)[
        0
    ]
    assert json.loads(plan_section) == {
        "plan": {"earliest_plan": "first plan", "setup": {"batch_size": 8}},
        "additional_candidate_information": {"recipe_sequence": [{"parent_kind": "base_model"}]},
    }
    assert json.loads(code_section) == {
        "code": "first code",
        "code_files": {"train.py": "second code"},
    }


def test_other_top_level_context_preserved_and_candidate_metadata_not_put_in_history(payload):
    payload["unknown_context"] = {"measured": 0.631578, "complete": "keep me"}
    payload["candidate_A"]["metadata"] = "candidate detail, not a historical experiment"
    rendered = prompt.render_prompt(payload)
    context_section = rendered.split("avoid redundant directions:\n", 1)[1].split(
        "\n\nCandidate A", 1
    )[0]
    context = json.loads(context_section)
    assert context == {
        key: value
        for key, value in payload.items()
        if key not in {"task", "candidate_A", "candidate_B"}
    }
    assert "candidate detail" not in context_section


@pytest.mark.parametrize("task_key", ["task", "task_desc", "task_description"])
def test_task_aliases(payload, task_key):
    payload[task_key] = payload.pop("task")
    assert "Task description:\nRank immediate official GSM8K accuracy." in prompt.render_prompt(
        payload
    )


def test_unselected_task_alias_remains_context(payload):
    payload["task_desc"] = "preferred description"
    rendered = prompt.render_prompt(payload)
    assert "Task description:\npreferred description" in rendered
    assert '"task": "Rank immediate official GSM8K accuracy."' in rendered


def test_plan_string_and_missing_code_supported(payload):
    payload["candidate_A"] = "An entire plan with 0.05 and 0.483333."
    rendered = prompt.render_prompt(payload)
    assert "Candidate A -- Plan:\nAn entire plan with 0.05 and 0.483333." in rendered
    assert "Candidate A -- Code:\nCode not supplied." in rendered


def test_missing_candidates_and_unknown_target_rejected(payload):
    with pytest.raises(ValueError, match="Both candidate"):
        prompt.render_prompt({"candidate_A": "A"})
    with pytest.raises(ValueError, match="target"):
        prompt.render_prompt(payload, target="best_score_whichever")
    with pytest.raises(TypeError, match="mapping"):
        prompt.render_prompt([])


@pytest.mark.parametrize("choice", ["A", "B"])
@pytest.mark.parametrize("swapped", [False, True])
def test_decode_maps_displayed_choice_to_canonical_orientation(choice, swapped):
    text = "My full reasoning.\n" + rf"\boxed{{{choice}}}"
    result = prompt.decode_boxed_response({"subtype": "success", "result": text}, swapped=swapped)
    assert result == {
        "choice_a": (choice == "A") != swapped,
        "displayed_choice": choice,
        "rationale": text,
    }
    assert "p_a" not in result


@pytest.mark.parametrize(
    "text", [r"\boxed{A}", r"$\boxed{A}$.", r"\(\boxed{ A }\)", r"\[\boxed{A}\]"]
)
def test_decode_accepts_normal_latex_final_wrappers(text):
    assert prompt.decode_boxed_response({"subtype": "success", "result": text})["choice_a"]


@pytest.mark.parametrize(
    "example",
    [
        r"The code removes `\boxed{}` before grading.",
        r"The examples `\boxed{A}` and `\boxed{B}` are literal syntax.",
        r"Use ``a ` character and \boxed{} here`` as a literal.",
        "```python\nexample = r'\\boxed{A}'\n```\n",
        "~~~text\n\\boxed{}\n\\boxed{B}\n~~~\n",
        "   ````text\n```\n\\boxed{A}\n   `````\n",
        "An inline `code span with\n\\boxed{A}` spans two lines.",
    ],
)
@pytest.mark.parametrize("swapped", [False, True])
def test_literal_code_boxes_ignored_without_losing_reasoning(example, swapped):
    answer = example + "\nMy decision:\n" + r"\boxed{B}"
    decoded = prompt.decode_boxed_response(
        {"subtype": "success", "result": answer}, swapped=swapped
    )
    assert decoded == {
        "choice_a": swapped,
        "displayed_choice": "B",
        "rationale": answer,
    }


@pytest.mark.parametrize(
    "answer",
    [
        r"Example `\boxed{}`. Actual choices: \boxed{A}, then \boxed{B}",
        r"Example `\boxed{B}`. Actual choices: \boxed{A}, then \boxed{A}",
        r"Only a code example: `\boxed{A}`",
        "```text\nOnly a code example: \\boxed{A}\n```",
        "An unclosed fence:\n```text\n\\boxed{A}",
        r"Unmatched ` is not code; \boxed{B} then \boxed{A}",
        r"Escaped \` is not code; \boxed{B} \` then \boxed{A}",
        r"\boxed{A} followed by code `\boxed{B}`",
    ],
)
def test_code_masking_does_not_relax_missing_multiple_or_nonfinal_answer_rules(answer):
    with pytest.raises(ValueError):
        prompt.decode_boxed_response({"subtype": "success", "result": answer})


def test_code_mask_preserves_offsets_and_newlines():
    answer = "Example `\\boxed{}`.\n```py\n'\\boxed{A}'\n```\nFinal: \\boxed{B}"
    masked = prompt._mask_markdown_code(answer)
    assert len(masked) == len(answer)
    assert [i for i, c in enumerate(masked) if c == "\n"] == [
        i for i, c in enumerate(answer) if c == "\n"
    ]
    assert masked.index(r"\boxed{B}") == answer.index(r"\boxed{B}")
    assert r"\boxed{}" not in masked
    assert r"\boxed{A}" not in masked


@pytest.mark.parametrize(
    "text",
    [
        "I choose A.",
        r"\boxed{A} or \boxed{B}",
        r"\boxed{A}, final answer \boxed{A}",
        r"\boxed{A or B}",
        r"\boxed{C}",
        r"\boxed{a}",
        r"\boxed{\text{A}}",
        r"\boxed{B} Actually, I choose A.",
        r"\boxed{A} \boxed{unfinished",
        r"\boxed{unfinished, later \boxed{A}",
        "",
        None,
        {"choice": "A"},
    ],
)
def test_missing_ambiguous_malformed_and_nonfinal_choices_rejected(text):
    with pytest.raises(ValueError):
        prompt.decode_boxed_response({"subtype": "success", "result": text})


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"subtype": "error"},
        {"subtype": "success", "is_error": True, "result": r"\boxed{A}"},
        None,
    ],
)
def test_failed_cli_results_rejected(raw):
    with pytest.raises(ValueError, match="successful"):
        prompt.decode_boxed_response(raw)


def test_swap_flag_must_be_boolean():
    with pytest.raises(TypeError, match="bool"):
        prompt.decode_boxed_response(
            {"subtype": "success", "result": r"\boxed{A}"}, swapped="false"
        )
