import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.outcome_prediction import wm_compile as compiler
from tools.outcome_prediction import wm_final_prepare as prepare
from tools.outcome_prediction import wm_lineage as lineage
from tools.outcome_prediction import wm_recipe_input as recipe_input
from tools.outcome_prediction import wm_review as source_review


def row(number, *, cell="r0-90", parent="org/base", script="/task/train.py"):
    key = f"{cell}/exp-{number:02d}"
    cutoff = f"2026-01-01T{number:02d}:00:00Z"
    content = "epochs = 2\nlr = 0.00001\n"
    code = {
        "role": "training",
        "script_path": script,
        "status": "reconstructed",
        "content": content,
    }
    proof = {
        **{k: code[k] for k in ("role", "script_path", "status")},
        "sha256": compiler.text_digest(content),
        "cutoff": cutoff,
        "trace_sha256": "a" * 64,
        "blockers": [],
        "evidence": {"at": "2026-01-01T00:00:00Z"},
    }
    return {
        "example_id": key,
        "cell_id": cell,
        "card_id": key.split("/")[-1],
        "first_submitted_at": cutoff,
        "model_input": {
            "task": {"benchmark": "gsm8k", "base_model": "org/base", "evaluation_n": 1319},
            "plan": {
                "setup": {
                    "method": {"family": "sft"},
                    "base_model": "org/base",
                    "output_dir": f"/task/ckpts/exp-{number:02d}",
                    "parent_checkpoint": {"path": parent},
                    "command": {
                        "cwd": "/task",
                        "script": script,
                        "argv": ["python", script, "--model", parent, "--lr", "1e-5"],
                    },
                    "data": [],
                }
            },
            "code": [code],
        },
        "audit": {"code_provenance": [proof]},
        "label": {"accuracy": 0.5, "official_metric": {"accuracy": 0.5}},
    }


def split():
    return {"train_cell_ids": ["r0-90"], "test_cell_ids": ["r0-91"]}


def setup(record):
    return record["model_input"]["plan"]["setup"]


def assemble(rows, **kwargs):
    return prepare.assemble(rows, lineage.build_graph(rows), split(), **kwargs)


def target(result, number):
    return next(r for r in result["drafts"] if r["example_id"] == f"r0-90/exp-{number:02d}")


def review_map(result):
    return {
        item["payload_sha256"]: {
            "payload_sha256": item["payload_sha256"],
            "reviewer": "synthetic fixture",
            "evidence_sha256": "b" * 64,
            "outcome_free": True,
            "executable_recipe_preserved": True,
        }
        for item in result["review_queue"]
    }


def add_builder(record, *, command=None, source=None):
    setup(record)["data"].append(
        {
            "path": "data/built.jsonl",
            "built_by": "/task/build.py",
            "build_command": command or ["python", "build.py", "--out", "data/built.jsonl"],
            **({"source": source} if source else {}),
        }
    )
    code = {
        "role": "data_builder_0",
        "script_path": "/task/build.py",
        "status": "reconstructed",
        "content": "n = 7473\nseed = 42\n",
    }
    record["model_input"]["code"].append(code)
    proof = copy.deepcopy(record["audit"]["code_provenance"][0])
    proof.update({k: code[k] for k in ("role", "script_path", "status")})
    proof["sha256"] = compiler.text_digest(code["content"])
    record["audit"]["code_provenance"].append(proof)


def test_complete_chain_is_topological_and_preserves_exact_artifact_variants():
    rows = [
        row(1),
        row(2, parent="/task/ckpts/exp-01/checkpoint-606"),
        row(3, parent="/task/ckpts/exp-02/final"),
    ]
    result = assemble(rows)
    recipe = target(result, 3)["model_input"]["recipe"]
    assert [s["step_id"] for s in recipe["steps"]] == [r["example_id"] for r in rows]
    assert recipe["steps"][1]["parents"] == [
        {
            "step_id": rows[0]["example_id"],
            "kind": "weights",
            "artifact": "/task/ckpts/exp-01/checkpoint-606",
        }
    ]
    assert target(result, 3)["step_sources"] == {r["example_id"]: r["example_id"] for r in rows}
    recipe_input.validate_full_recipe(target(result, 3)["model_input"])
    assert "1e-5" in json.dumps(recipe)
    assert all(r["status"] == "draft_needs_review" for r in result["drafts"])
    assert result["audit"]["inventory_cards"] == 3


def test_no_scanner_finding_never_implies_whole_step_approval():
    rows = [row(1)]
    result = assemble(rows)
    assert result["audit"]["issue_counts"] == {"whole_step_content_review_required": 1}
    assert target(result, 1)["content_review"]["status"] == "needs_review"
    reviewed = assemble(rows, whole_step_reviews=review_map(result))
    assert target(reviewed, 1)["status"] == "approved"
    assert not target(reviewed, 1)["audit"]["execution_certified"]


def compiled_with_manual_hold(rows, held_number):
    compiled, _ = compiler.compile_steps([prepare._project(r) for r in rows])
    held = next(s for s in compiled if s["step_id"] == f"r0-90/exp-{held_number:02d}")
    source = held["draft_input"]["plan"]["setup"]["method"]["family"]
    document = {
        "schema": source_review.MANUAL_SCHEMA,
        "scope": source_review.MANUAL_SCOPE,
        "reviewer": "synthetic source reviewer",
        "source_inventory_sha256": "a" * 64,
        "compiler_sha256": hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest(),
        "source_drafts_sha256": source_review.review_drafts_digest(compiled),
        "reviews": [
            {
                "review_id": "unresolved-target-binding",
                "review_status": "needs_review",
                "full_source_read": True,
                "source_sha256": compiler.digest(source),
                "raw_text_sha256": compiler.text_digest(source),
                "evidence_sha256": "e" * 64,
                "rationale": "The target binding requires an explicit source-review decision.",
                "occurrences": [
                    {"step_id": held["step_id"], "path": "/plan/setup/method/family"}
                ],
            }
        ],
    }
    reviewed, summary = source_review.apply_manual_source_reviews(
        compiled, [document], source_inventory_sha256="a" * 64
    )
    assert summary["manual_holds"] == 1
    assert all(a["draft_input"] == b["draft_input"] for a, b in zip(compiled, reviewed))
    return reviewed


@pytest.mark.parametrize("held_number", [1, 2])
@pytest.mark.parametrize("status", ["needs_review", "draft_clear"])
def test_manual_source_hold_blocks_own_and_ancestor_approval(held_number, status):
    rows = [
        row(1),
        row(2, parent="/task/ckpts/exp-01"),
        row(3, parent="/task/ckpts/exp-02"),
        row(4),
    ]
    initial = assemble(rows)
    reviews = review_map(initial)
    clear = assemble(rows, whole_step_reviews=reviews)
    assert all(r["status"] == "approved" for r in clear["drafts"])
    compiled = compiled_with_manual_hold(rows, held_number)
    held = compiled[held_number - 1]
    held["status"] = status  # Holds remain authoritative even with stale status metadata.
    assert not held["audit"]["unresolved_findings"]
    hold = held["audit"]["manual_source_holds"][0]
    hold["private_text"] = "PRIVATE REVIEW NARRATIVE MUST NOT BE COPIED"
    snapshot = copy.deepcopy(compiled)
    result = assemble(rows, compiled_steps=compiled, whole_step_reviews=reviews)
    assert compiled == snapshot
    for number in (1, 2, 3, 4):
        selected = target(result, number)
        blocked = held_number <= number <= 3
        assert selected["status"] == ("draft_needs_review" if blocked else "approved")
        assert selected["model_input"] == target(clear, number)["model_input"]
        issues = selected["audit"]["issues"]
        assert issues == (
            [
                {
                    "kind": "compiler_manual_source_hold",
                    "step_id": held["step_id"],
                    "source_example_id": held["step_id"],
                    **{k: hold[k] for k in ("path", "review_id", "source_sha256", "evidence_sha256")},
                }
            ]
            if blocked
            else []
        )
    assert "PRIVATE REVIEW NARRATIVE" not in json.dumps(result)


def test_absent_or_empty_manual_holds_leave_clear_drafts_unchanged():
    rows = [row(1)]
    compiled, _ = compiler.compile_steps([prepare._project(r) for r in rows])
    reviews = review_map(assemble(rows))
    expected = assemble(rows, compiled_steps=compiled, whole_step_reviews=reviews)
    compiled[0]["audit"]["manual_source_holds"] = []
    assert assemble(rows, compiled_steps=compiled, whole_step_reviews=reviews) == expected


def test_manual_hold_cannot_be_bypassed_by_builder_isolation_approval():
    producer, consumer = row(1), row(2)
    add_builder(producer)
    setup(consumer)["command"]["argv"] += ["--data", "data/built.jsonl"]
    rows = [producer, consumer]
    initial = assemble(rows)
    isolation = next(
        i
        for i in target(initial, 2)["audit"]["issues"]
        if i["kind"] == "operation_isolation_review_required"
    )
    proof = {k: v for k, v in isolation.items() if k not in {"kind", "step_id"}}
    proof.update(reviewer="synthetic isolation reviewer", evidence_sha256="c" * 64)
    options = {
        "whole_step_reviews": review_map(initial),
        "operation_reviews": {proof["payload_sha256"]: proof},
    }
    assert target(assemble(rows, **options), 2)["status"] == "approved"
    compiled = compiled_with_manual_hold(rows, 1)
    result = assemble(rows, compiled_steps=compiled, **options)
    selected = target(result, 2)
    assert selected["status"] == "draft_needs_review"
    assert selected["audit"]["issues"][0]["kind"] == "compiler_manual_source_hold"
    assert selected["audit"]["issues"][0]["step_id"] == "r0-90/exp-01#data-0"
    assert selected["audit"]["issues"][0]["source_example_id"] == producer["example_id"]


@pytest.mark.parametrize("holds", [None, {}, "unresolved", [None], [{}]])
def test_malformed_manual_holds_fail_closed(holds):
    rows = [row(1)]
    compiled, _ = compiler.compile_steps([prepare._project(r) for r in rows])
    compiled[0]["audit"]["manual_source_holds"] = holds
    with pytest.raises((TypeError, ValueError), match="manual source hold"):
        assemble(rows, compiled_steps=compiled, whole_step_reviews=review_map(assemble(rows)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "not-a-pointer"),
        ("review_id", ""),
        ("source_sha256", None),
        ("evidence_sha256", "not-a-hash"),
    ],
)
def test_malformed_manual_hold_binding_is_not_ignored(field, value):
    rows = [row(1)]
    compiled = compiled_with_manual_hold(rows, 1)
    compiled[0]["audit"]["manual_source_holds"][0][field] = value
    with pytest.raises(ValueError, match="manual source hold"):
        assemble(rows, compiled_steps=compiled, whole_step_reviews=review_map(assemble(rows)))


class IdentityOnly(dict):
    def __getitem__(self, key):
        if key != "cell_id":
            raise AssertionError(f"Heldout field accessed: {key}")
        return super().__getitem__(key)


class AllowedKeys(dict):
    def __init__(self, values, allowed):
        super().__init__(values)
        self.allowed = allowed

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError(f"Forbidden field accessed: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self.allowed:
            raise AssertionError(f"Forbidden field read: {key}")
        return super().get(key, default)

    def __deepcopy__(self, memo):
        raise AssertionError("Broad copy would inspect forbidden fields")


def test_assembly_positive_projection_never_reads_labels_or_prior_outcomes():
    ordinary = row(1)
    graph = lineage.build_graph([ordinary])
    guarded = AllowedKeys(
        ordinary,
        {
            "example_id",
            "cell_id",
            "card_id",
            "first_submitted_at",
            "model_input",
            "audit",
        },
    )
    guarded["model_input"] = AllowedKeys(ordinary["model_input"], {"task", "plan", "code"})
    guarded["audit"] = AllowedKeys(ordinary["audit"], {"code_provenance"})
    assert prepare.assemble([guarded], graph, split()) == prepare.assemble(
        [ordinary], graph, split()
    )
    changed = copy.deepcopy(ordinary)
    changed["label"] = {"accuracy": 999}
    changed["prior_observations"] = "DO NOT READ"
    changed["model_input"]["known_previous_checkpoints"] = "DO NOT READ"
    assert assemble([changed]) == assemble([ordinary])


@pytest.mark.parametrize(
    "change",
    [
        lambda graph: graph["nodes"]["r0-90/exp-02"].update(topological_closure=["r0-90/exp-02"]),
        lambda graph: graph["nodes"]["r0-90/exp-02"]["parents"][0].update(artifact="/other"),
        lambda graph: graph["nodes"].pop("r0-90/exp-01"),
    ],
)
def test_stale_or_tampered_graph_and_closure_fail_closed(change):
    rows = [row(1), row(2, parent="/task/ckpts/exp-01")]
    graph = lineage.build_graph(rows)
    change(graph)
    with pytest.raises(ValueError, match="Lineage graph"):
        prepare.assemble(rows, graph, split())


def test_exact_multiple_merge_ingredients_from_one_source_preserved():
    a, b = row(1), row(2, script="/task/soup.py")
    setup(b)["method"]["family"] = "merge"
    setup(b)["command"]["argv"] = [
        "python",
        "soup.py",
        "--models",
        "ckpts/exp-01/final,ckpts/exp-01/checkpoint-12",
    ]
    parents = target(assemble([a, b]), 2)["model_input"]["recipe"]["steps"][-1]["parents"]
    assert len(parents) == 2
    assert {p["artifact"] for p in parents} == {
        "/task/ckpts/exp-01/final",
        "/task/ckpts/exp-01/checkpoint-12",
    }
    assert all(p["kind"] == "weights" for p in parents)


def test_data_builder_only_never_prepends_unrelated_training_or_training_ancestors():
    unrelated = row(1)
    producer = row(2, parent="/task/ckpts/exp-01")
    add_builder(producer)
    consumer = row(3)
    setup(consumer)["command"]["argv"] += ["--data", "data/built.jsonl"]
    result = assemble([unrelated, producer, consumer])
    selected = target(result, 3)
    steps = selected["model_input"]["recipe"]["steps"]
    assert [s["step_id"] for s in steps] == ["r0-90/exp-02#data-0", "r0-90/exp-03"]
    assert steps[0]["role"] == "data_generation"
    assert "train.py" not in json.dumps(steps[0])
    assert "ckpts/exp-01" not in json.dumps(steps[0])
    assert "7473" in json.dumps(steps[0])
    assert steps[1]["parents"][0]["kind"] == "generated_data"
    assert any(
        i["kind"] == "operation_isolation_review_required" for i in selected["audit"]["issues"]
    )
    # A whole-step semantic review alone is not an operation-isolation certificate.
    reviewed = assemble([unrelated, producer, consumer], whole_step_reviews=review_map(result))
    assert target(reviewed, 3)["status"] != "approved"
    isolation = next(
        i for i in selected["audit"]["issues"] if i["kind"] == "operation_isolation_review_required"
    )
    proof = {k: v for k, v in isolation.items() if k not in {"kind", "step_id"}}
    proof.update(reviewer="synthetic isolation reviewer", evidence_sha256="c" * 64)
    certified = assemble(
        [unrelated, producer, consumer],
        whole_step_reviews=review_map(result),
        operation_reviews={proof["payload_sha256"]: proof},
    )
    assert target(certified, 3)["status"] == "approved"


def test_builder_generator_weights_remain_typed_dependencies():
    generator, producer, consumer = row(1), row(2), row(3)
    add_builder(
        producer,
        command=[
            "python",
            "build.py",
            "--model",
            "ckpts/exp-01/checkpoint-9",
            "--out",
            "data/built.jsonl",
        ],
    )
    setup(consumer)["command"]["argv"] += ["--data", "data/built.jsonl"]
    selected = target(assemble([generator, producer, consumer]), 3)
    steps = selected["model_input"]["recipe"]["steps"]
    assert [s["step_id"] for s in steps] == [
        "r0-90/exp-01",
        "r0-90/exp-02#data-0",
        "r0-90/exp-03",
    ]
    assert steps[1]["parents"] == [
        {
            "step_id": "r0-90/exp-01",
            "kind": "weights",
            "artifact": "/task/ckpts/exp-01/checkpoint-9",
        }
    ]


def test_supplemental_diagnostic_catches_rft_flags_missing_from_frozen_graph():
    record = row(1)
    add_builder(
        record,
        source="synthetic:self from sampler_a and sampler_b",
        command=[
            "python",
            "build.py",
            "--rft",
            "data/rft1.jsonl,data/rft2.jsonl",
            "--rft_per_q",
            "6",
            "--out",
            "data/built.jsonl",
        ],
    )
    graph = lineage.build_graph([record])
    assert not graph["nodes"][record["example_id"]]["closure_has_unresolved_dependencies"]
    snapshot = copy.deepcopy(graph)
    result = prepare.assemble([record], graph, split())
    assert graph == snapshot
    issues = [
        i
        for i in target(result, 1)["audit"]["issues"]
        if i["kind"] == "self_generated_input_provenance_unresolved"
    ]
    assert {i["artifact"] for i in issues} == {"/task/data/rft1.jsonl", "/task/data/rft2.jsonl"}
    assert all("6" != i["artifact"] for i in issues)
    reviewed = prepare.assemble([record], graph, split(), whole_step_reviews=review_map(result))
    assert target(reviewed, 1)["status"] != "approved"


def test_self_generation_requires_generator_or_data_edge_not_unrelated_training_parent():
    initial, child = row(1), row(2, parent="/task/ckpts/exp-01")
    add_builder(child, source="synthetic:self")
    result = assemble([initial, child])
    assert "self_generation_operation_provenance_unresolved" in result["audit"]["issue_counts"]


def test_explicit_generator_checkpoint_satisfies_supplemental_diagnostic():
    initial, child = row(1), row(2)
    add_builder(
        child,
        source="synthetic:self",
        command=[
            "python",
            "build.py",
            "--model",
            "ckpts/exp-01",
            "--out",
            "data/built.jsonl",
        ],
    )
    result = assemble([initial, child])
    assert not any(kind.startswith("self_generat") for kind in result["audit"]["issue_counts"])


def test_external_synthetic_corpus_is_not_mistaken_for_ancestor_self_generation():
    record = row(1)
    add_builder(record, source="nvidia/OpenMathReasoning synthetic teacher corpus")
    result = assemble([record])
    assert result["audit"]["issue_counts"] == {"whole_step_content_review_required": 1}


def test_missing_builder_command_stays_unresolved_without_producer_training():
    producer, consumer = row(1), row(2)
    add_builder(producer)
    del setup(producer)["data"][0]["build_command"]
    setup(consumer)["command"]["argv"] += ["--data", "data/built.jsonl"]
    selected = target(assemble([producer, consumer]), 2)
    assert len(selected["model_input"]["recipe"]["steps"]) == 1
    assert "data_builder_operation_not_isolatable" in {
        i["kind"] for i in selected["audit"]["issues"]
    }


def test_card_only_and_unknown_artifact_cannot_be_certified_by_semantic_review():
    rows = [row(1), row(2, parent="/task/unknown")]
    setup(rows[1])["data"] = [{"source": "derived:exp-01"}]
    result = assemble(rows)
    selected = target(assemble(rows, whole_step_reviews=review_map(result)), 2)
    assert selected["status"] != "approved"
    issues = [i for i in selected["audit"]["issues"] if i["kind"] == "unresolved_dependency"]
    assert {i["status"] for i in issues} == {"declared_card_reference_only", "unresolved"}


def test_setup_launch_works_but_never_invents_missing_code():
    record = row(1)
    setup(record)["launch"] = setup(record).pop("command")
    result = assemble([record])
    assert target(result, 1)["model_input"]["recipe"]["steps"][0]["role"] == "training"
    assert result["audit"]["issue_counts"] == {"whole_step_content_review_required": 1}
    record["model_input"]["code"][0].update(status="not_declared", content=None, script_path=None)
    result = assemble([record])
    assert "declared_script_or_config_not_covered" in result["audit"]["coverage_warning_counts"]
    assert result["audit"]["issue_counts"] == {"whole_step_content_review_required": 1}
    codes = target(result, 1)["model_input"]["recipe"]["steps"][0]["code"]
    assert any(c["script_path"] == "/task/train.py" and c["status"] == "unavailable" for c in codes)
    reviewed = assemble([record], whole_step_reviews=review_map(result))
    assert target(reviewed, 1)["status"] == "approved"


@pytest.mark.parametrize("status", ["unavailable", "not_requested", "blocked"])
def test_honest_missing_source_is_disclosed_not_a_whole_plan_blocker(status):
    record = row(1)
    record["model_input"]["code"][0].update(status=status, content=None)
    record["audit"]["code_provenance"] = []
    result = assemble([record])
    assert result["audit"]["issue_counts"] == {"whole_step_content_review_required": 1}
    assert result["audit"]["coverage_warning_counts"]["code_not_reconstructed"] == 1
    reviewed = assemble([record], whole_step_reviews=review_map(result))
    selected = target(reviewed, 1)
    assert selected["status"] == "approved"
    assert selected["model_input"]["recipe"]["steps"][0]["code"][0]["status"] == "unavailable"
    assert selected["audit"]["coverage_warnings"]


def test_claimed_reconstructed_source_cannot_silently_be_downgraded():
    rows = [row(1)]
    compiled, _ = compiler.compile_steps([prepare._project(r) for r in rows])
    compiled[0]["draft_input"]["code"] = []
    compiled[0]["audit"]["draft_input_sha256"] = compiler.digest(compiled[0]["draft_input"])
    result = assemble(rows, compiled_steps=compiled)
    assert "reconstructed_source_identity_changed" in result["audit"]["issue_counts"]
    reviewed = assemble(rows, compiled_steps=compiled, whole_step_reviews=review_map(result))
    assert target(reviewed, 1)["status"] != "approved"


def test_unsupported_command_parsing_is_disclosed_without_dropping_plan():
    record = row(1)
    setup(record)["command"]["argv"] = ["bash", "-lc", "python train.py <<'EOF'\nEOF"]
    result = assemble([record])
    assert result["audit"]["issue_counts"] == {"whole_step_content_review_required": 1}
    assert result["audit"]["coverage_warning_counts"]["command_not_statically_covered"] == 1
    assert (
        target(result, 1)["model_input"]["recipe"]["steps"][0]["plan"]["setup"]["command"]["argv"]
        == setup(record)["command"]["argv"]
    )


def test_archive_target_size_is_not_replaced_by_local_protocol_n_or_shipping_rule():
    record = row(1)
    record["model_input"]["plan"]["evaluation"] = {
        "protocol": {"n": 150},
        "ship_rule": "Use the exported checkpoint after the screen",
    }
    selected = target(assemble([record]), 1)
    assert selected["model_input"]["task"]["evaluation_n"] == 1319
    assert selected["audit"]["target_evaluation_n"] == 1319
    assert (
        selected["model_input"]["recipe"]["steps"][0]["plan"]["evaluation"]["protocol"]["n"] == 150
    )


def test_realized_data_yield_is_queued_for_semantic_review_not_auto_approved():
    record = row(1)
    setup(record)["data"] = [{"n_examples": 18909, "selection": "Correct self-generated responses"}]
    result = assemble([record])
    assert target(result, 1)["status"] == "draft_needs_review"
    assert any(
        "realized correctness-filtered yields" in note
        for note in result["review_queue"][0]["semantic_review_focus"]
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("sha256", "b" * 64, "source_code_hash_mismatch"),
        ("cutoff", "2026-01-02T00:00:00Z", "code_not_proven_before_first_submission"),
        ("trace_sha256", None, "incomplete_reconstruction_evidence"),
        ("blockers", ["unobserved write"], "incomplete_reconstruction_evidence"),
        ("evidence", {"at": "2026-01-01T02:00:00Z"}, "code_not_proven_before_first_submission"),
    ],
)
def test_first_submit_provenance_is_required(field, value, expected):
    record = row(1)
    record["audit"]["code_provenance"][0][field] = value
    result = assemble([record])
    assert expected in result["audit"]["issue_counts"]
    reviewed = assemble([record], whole_step_reviews=review_map(result))
    assert target(reviewed, 1)["status"] != "approved"


def test_reviewed_compiler_projection_and_draft_hashes_are_bound():
    rows = [row(1)]
    projected = [prepare._project(r) for r in rows]
    compiled, _ = compiler.compile_steps(projected)
    compiled[0]["draft_input"]["code"][0]["content"] += "# reviewed edit\n"
    with pytest.raises(ValueError, match="draft hash"):
        assemble(rows, compiled_steps=compiled)
    compiled[0]["audit"]["draft_input_sha256"] = compiler.digest(compiled[0]["draft_input"])
    compiled[0]["audit"]["source_recipe_projection_sha256"] = compiler.digest(
        projected[0]["model_input"]
    )
    compiled[0]["audit"]["source_input_sha256"] = "irrelevant legacy broad hash"
    result = assemble(rows, compiled_steps=compiled)
    proof = target(result, 1)["step_provenance"][rows[0]["example_id"]]["code_coverage"][
        "recorded_code"
    ][0]
    assert proof["source_content_sha256"] != proof["draft_content_sha256"]


def test_reviewed_setup_cannot_silently_change_lineage_declarations():
    rows = [row(1)]
    compiled, _ = compiler.compile_steps([prepare._project(r) for r in rows])
    compiled[0]["draft_input"]["plan"]["setup"]["command"]["argv"][-3] = "other/model"
    compiled[0]["audit"]["draft_input_sha256"] = compiler.digest(compiled[0]["draft_input"])
    assert (
        "review_changed_lineage_declarations"
        in assemble(rows, compiled_steps=compiled)["audit"]["issue_counts"]
    )


def test_split_gaps_reported_without_silent_easy_cohort_success():
    result = assemble([row(1)])
    assert result["audit"]["runs_without_approved_targets"] == ["r0-90", "r0-91"]
    assert result["audit"]["run_coverage"][1]["inventory_targets"] == 0
    with pytest.raises(ValueError, match="overlap"):
        prepare.assemble(
            [row(1)],
            lineage.build_graph([row(1)]),
            {
                "train_cell_ids": ["r0-90"],
                "test_cell_ids": ["r0-90"],
            },
        )
    with pytest.raises(ValueError, match="outside"):
        assemble([row(1, cell="r0-92")])


def test_train_join_filters_heldout_before_reading_identity_features_or_labels():
    rows = [row(1)]
    drafts = assemble(rows)
    approved = assemble(rows, whole_step_reviews=review_map(drafts))
    records, audit = prepare.join_train_labels(
        approved, [*rows, IdentityOnly(cell_id="r0-91")], split()
    )
    assert len(records) == 1 and records[0]["label"] == {"accuracy": 0.5}
    assert "official_metric" not in json.dumps(records[0]["model_input"])
    assert not audit["test_labels_read"] and not audit["empty_train_cell_ids"]


def test_train_join_does_not_read_unapproved_labels_and_reports_empty_runs():
    record = row(1)
    result = assemble([record])
    minimal = AllowedKeys(record, {"cell_id", "example_id"})
    joined, audit = prepare.join_train_labels(result, [minimal], split())
    assert not joined and audit["empty_train_cell_ids"] == ["r0-90"]


def test_final_label_cannot_be_joined_to_different_recipe_target():
    rows = [row(1)]
    drafts = assemble(rows)
    approved = assemble(rows, whole_step_reviews=review_map(drafts))
    approved["drafts"][0]["step_sources"]["r0-90/exp-01"] = "r0-90/exp-02"
    with pytest.raises(ValueError, match="source identity mismatch"):
        prepare.join_train_labels(approved, rows, split())


@pytest.mark.parametrize(
    "label",
    [
        {"accuracy": 0.5},
        {"accuracy": float("nan"), "official_metric": {"accuracy": 0.5}},
        {"accuracy": 0.5, "official_metric": {"accuracy": 0.6}},
    ],
)
def test_only_valid_consistent_official_final_targets_join(label):
    record = row(1)
    drafts = assemble([record])
    approved = assemble([record], whole_step_reviews=review_map(drafts))
    record["label"] = label
    joined, audit = prepare.join_train_labels(approved, [record], split())
    assert not joined and len(audit["label_gaps"]) == 1


def test_artifact_is_immutable_and_has_no_label_fields(tmp_path):
    record = row(1)
    inventory, graph_file, split_file = [
        tmp_path / name for name in ("inventory", "graph", "split")
    ]
    inventory.write_text(json.dumps(record) + "\n")
    graph_file.write_text(json.dumps(lineage.build_graph([record])))
    split_file.write_text(json.dumps(split()))
    output = tmp_path / "drafts"
    result = prepare.build_artifact(inventory, graph_file, split_file, output)
    assert result["audit"]["inventory_cards"] == 1
    text = (output / "drafts.json").read_text()
    assert '"label"' not in text and '"accuracy"' not in text
    proof = json.loads((output / "provenance.json").read_text())
    assert proof["inventory_sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        prepare.build_artifact(inventory, graph_file, split_file, output)


def review_document(result, *, batch_schema=False):
    mapping = review_map(result)
    entries = []
    for key, review in mapping.items():
        entry = {"payload_sha256": key, "findings": []}
        if batch_schema:
            entry.update(
                decision="approved",
                full_step_read=True,
                full_plan_read=True,
                all_supplied_code_read=True,
                declared_parents_inspected=True,
            )
        else:
            entry.update(
                status="approved", complete_step_read=True, complete_supplied_code_read=True
            )
        review["evidence_sha256"] = compiler.digest(entry)
        entries.append(
            entry
            if batch_schema
            else {"evidence": entry, "evidence_sha256": compiler.digest(entry)}
        )
    return {
        "schema": "wm-whole-step-review-batch-v1"
        if batch_schema
        else "wm-whole-step-manual-review-v1",
        "reviews_map": mapping,
        "evidence_entries": entries,
    }


@pytest.mark.parametrize("batch_schema", [False, True])
def test_explicit_whole_step_review_evidence_is_loaded(tmp_path, batch_schema):
    document = review_document(assemble([row(1)]), batch_schema=batch_schema)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(document))
    reviews, provenance = prepare.load_whole_step_reviews([path])
    assert reviews == document["reviews_map"]
    assert provenance[0]["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("failure", ["pending", "unread", "findings", "stale", "wrong_step"])
def test_pending_unread_or_unbound_evidence_cannot_approve(tmp_path, failure):
    document = review_document(assemble([row(1)]))
    wrapper = document["evidence_entries"][0]
    entry = wrapper["evidence"]
    if failure == "pending":
        entry["status"] = "needs_review"
    elif failure == "unread":
        entry["complete_supplied_code_read"] = False
    elif failure == "findings":
        entry["findings"] = ["actual unresolved observation"]
    elif failure == "stale":
        entry["extra"] = "changed after signing"
    else:
        entry["payload_sha256"] = "e" * 64
    if failure != "stale":
        wrapper["evidence_sha256"] = compiler.digest(entry)
        next(iter(document["reviews_map"].values()))["evidence_sha256"] = compiler.digest(
            entry
        )
    path = tmp_path / "review.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        prepare.load_whole_step_reviews([path])


def test_artifact_integrates_explicit_approvals_and_records_provenance(tmp_path):
    record = row(1)
    inventory, graph_file, split_file, review_file = [
        tmp_path / name for name in ("inventory", "graph", "split", "review")
    ]
    inventory.write_text(json.dumps(record) + "\n")
    graph_file.write_text(json.dumps(lineage.build_graph([record])))
    split_file.write_text(json.dumps(split()))
    review_file.write_text(json.dumps(review_document(assemble([record]))))
    output = tmp_path / "reviewed"
    result = prepare.build_artifact(
        inventory, graph_file, split_file, output, whole_step_review_paths=[review_file]
    )
    assert result["drafts"][0]["status"] == "approved"
    assert not result["drafts"][0]["audit"]["execution_certified"]
    proof = json.loads((output / "provenance.json").read_text())
    assert proof["whole_step_reviews"][0]["file_sha256"] == hashlib.sha256(
        review_file.read_bytes()
    ).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "drafts.json").stat().st_mode & 0o777 == 0o600


def test_approval_for_noncurrent_step_fails_before_artifact_write(tmp_path):
    record = row(1)
    inventory, graph_file, split_file, review_file = [
        tmp_path / name for name in ("inventory", "graph", "split", "review")
    ]
    inventory.write_text(json.dumps(record) + "\n")
    graph_file.write_text(json.dumps(lineage.build_graph([record])))
    split_file.write_text(json.dumps(split()))
    review_file.write_text(json.dumps(review_document(assemble([row(2)]))))
    output = tmp_path / "reviewed"
    with pytest.raises(ValueError, match="current assembled step"):
        prepare.build_artifact(
            inventory, graph_file, split_file, output, whole_step_review_paths=[review_file]
        )
    assert not output.exists()
