import copy

import pytest

from tools.outcome_prediction import wm_recipe_input as ri


def step(step_id="one", parents=None):
    return {
        "step_id": step_id,
        "role": "training",
        "parents": parents or [],
        "plan": {
            "hypothesis": {"claim": "A smaller learning rate may preserve useful features."},
            "setup": {"lr": 1e-5, "epochs": 2, "temperature": 0.6, "top_p": 0.95},
            "evaluation": {"protocol": {"metric": "accuracy", "n": 30}},
        },
        "code": [
            {
                "role": "training",
                "script_path": "train.py",
                "status": "reconstructed",
                "content": "lr = 1e-5\nn = 8604\n",
            }
        ],
    }


def payload():
    return {
        "schema": ri.SCHEMA,
        "task": {"benchmark": "aime2025", "base_model": "Qwen/Qwen3-4B-Base", "evaluation_n": 30},
        "recipe": {
            "steps": [
                step(),
                step("two", [{"step_id": "one", "kind": "weights", "artifact": "ckpts/one/final"}]),
            ],
            "final_step_id": "two",
        },
    }


def reviews(p):
    return {
        s["step_id"]: {
            "payload_sha256": ri.digest(s),
            "reviewer": "synthetic test fixture",
            "evidence_sha256": "a" * 64,
            "outcome_free": True,
            "executable_recipe_preserved": True,
        }
        for s in p["recipe"]["steps"]
    }


def build(p, review=None):
    return ri.build_reviewed_input(
        task=p["task"],
        steps=p["recipe"]["steps"],
        final_step_id=p["recipe"]["final_step_id"],
        reviews=reviews(p) if review is None else review,
    )


def test_complete_recipe_preserves_scientific_parameters_and_code():
    p = payload()
    result = build(p)
    assert result == p
    assert result is not p
    assert result["recipe"]["steps"][0]["plan"]["setup"]["lr"] == 1e-5
    assert "n = 8604" in result["recipe"]["steps"][0]["code"][0]["content"]
    assert "reviewer" not in str(result)


@pytest.mark.parametrize("field", sorted(ri.OUTCOME_FIELDS))
def test_outcome_fields_rejected_even_when_nested_or_zero(field):
    p = payload()
    p["recipe"]["steps"][0]["plan"]["problem"] = {"nested": [{field: 0}]}
    with pytest.raises(ValueError, match="Outcome field"):
        build(p)


def test_comparator_reference_allowed_but_no_ancestor_score():
    p = payload()
    evaluation = p["recipe"]["steps"][1]["plan"]["evaluation"]
    evaluation["comparator"] = {"ref": "one"}
    build(p)
    evaluation["comparator"]["value"] = 0.267
    with pytest.raises(ValueError, match="measured outcome"):
        build(p)


def test_legacy_payload_does_not_pass_complete_recipe_boundary():
    with pytest.raises(ValueError, match="requires schema"):
        ri.validate_full_recipe(
            {
                "task": payload()["task"],
                "plan": {},
                "known_previous_checkpoints": [{"official_accuracy": 0.5}],
            }
        )


def test_disconnected_missing_and_cyclic_steps_are_rejected():
    p = payload()
    p["recipe"]["steps"].insert(1, step("unrelated"))
    with pytest.raises(ValueError, match="Extraneous"):
        build(p)
    p = payload()
    p["recipe"]["steps"][1]["parents"][0]["step_id"] = "missing"
    with pytest.raises(ValueError, match="Missing, cyclic"):
        build(p)
    p = payload()
    p["recipe"]["steps"][0]["parents"] = [
        {
            "step_id": "two",
            "kind": "weights",
            "artifact": "ckpts/two",
        }
    ]
    with pytest.raises(ValueError, match="Missing, cyclic"):
        build(p)


def test_multiple_artifacts_from_same_producer_not_collapsed():
    p = payload()
    p["recipe"]["steps"][1]["role"] = "merge"
    p["recipe"]["steps"][1]["parents"].append(
        {
            "step_id": "one",
            "kind": "weights",
            "artifact": "ckpts/one/checkpoint-1000",
        }
    )
    assert len(build(p)["recipe"]["steps"][1]["parents"]) == 2


def test_generated_data_dependency_included_in_closure():
    p = payload()
    p["recipe"]["steps"][1]["parents"][0]["kind"] = "generated_data"
    p["recipe"]["steps"][1]["parents"][0]["artifact"] = "generated.jsonl"
    assert build(p) == p


def test_unavailable_code_is_not_replaced_with_final_snapshot():
    p = payload()
    code = p["recipe"]["steps"][0]["code"][0]
    code.update(status="unavailable", content=None)
    build(p)
    code["content"] = "later code"
    with pytest.raises(ValueError, match="later snapshot"):
        build(p)


def test_review_absence_staleness_and_explicit_rejection():
    p = payload()
    review = reviews(p)
    review.pop("one")
    with pytest.raises(ValueError, match="Every retained step"):
        build(p, review)
    review = reviews(p)
    p["recipe"]["steps"][0]["plan"]["setup"]["lr"] = 2e-5
    with pytest.raises(ValueError, match="changed after"):
        build(p, review)
    review = reviews(p)
    review["one"]["outcome_free"] = False
    with pytest.raises(ValueError, match="Unapproved"):
        build(p, review)


def test_prose_and_code_need_review_not_just_key_filtering():
    p = payload()
    old_review = reviews(p)
    p["recipe"]["steps"][0]["code"][0]["content"] += "# ancestor achieved 8/30\n"
    # Structural validation cannot establish semantic safety. A changed string
    # invalidates the review; a reviewer must not approve this outcome-bearing code.
    ri.validate_full_recipe(p)
    with pytest.raises(ValueError, match="changed after"):
        build(p, old_review)


def test_final_score_is_not_an_input_or_a_reviewed_step_field():
    p = payload()
    a = {"label": {"accuracy": 0.1}, "model_input": build(p)}
    b = {"label": {"accuracy": 0.9}, "model_input": build(p)}
    assert a["model_input"] == b["model_input"]
    p["recipe"]["steps"][-1]["label"] = {"accuracy": 0.1}
    with pytest.raises(ValueError, match="Unexpected recipe-step fields"):
        build(p)


def row(p, target="exp-02"):
    return {
        "example_id": f"r0-01/{target}",
        "cell_id": "r0-01",
        "model_input": p,
        "step_sources": {"one": "r0-01/exp-01", "two": "r0-01/exp-02"},
    }


def test_optional_strict_terminal_policy_rejects_prefix_target_supervision():
    p = payload()
    terminal = row(p)
    prefix = copy.deepcopy(terminal)
    prefix["example_id"] = "r0-01/exp-01"
    prefix["model_input"]["recipe"] = {"steps": [step()], "final_step_id": "one"}
    prefix["step_sources"] = {"one": "r0-01/exp-01"}
    assert ri.assert_terminal_target_supervision([terminal]) == {
        "target_ids": ["r0-01/exp-02"],
        "unlabeled_ancestor_ids": ["r0-01/exp-01"],
    }
    with pytest.raises(ValueError, match="internal checkpoint"):
        ri.assert_terminal_target_supervision([terminal, prefix])


def test_shared_ancestor_allowed_but_cross_run_lineage_rejected():
    a = row(payload())
    b = copy.deepcopy(a)
    b["example_id"] = "r0-01/exp-03"
    b["step_sources"]["two"] = b["example_id"]
    assert len(ri.assert_terminal_target_supervision([a, b])["target_ids"]) == 2
    b["step_sources"]["one"] = "r0-02/exp-01"
    with pytest.raises(ValueError, match="scientist run"):
        ri.assert_terminal_target_supervision([a, b])
