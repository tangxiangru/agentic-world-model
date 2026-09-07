"""Private source-review edits remain exact, scoped, and non-certifying."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.outcome_prediction.wm_compile import compile_steps, digest, text_digest
from tools.outcome_prediction.wm_review import (
    apply_manual_source_reviews,
    apply_observed_numeric_reviews,
    apply_source_reviews,
    build_artifact,
    build_manual_artifact,
    build_observed_numeric_artifact,
    compile_reviewed_steps,
    load_review_documents,
    observed_numeric_source_digest,
    review_drafts_digest,
)

INVENTORY = text_digest("fixture inventory")
COMPILER = hashlib.sha256(
    (Path(__file__).resolve().parents[1] / "tools/outcome_prediction/wm_compile.py").read_bytes()
).hexdigest()


def test_private_bundle_is_explicit_hash_bound_and_never_overwritten(tmp_path):
    source = row()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(source) + "\n")
    document = artifact(source)
    document["source_inventory_sha256"] = hashlib.sha256(inventory.read_bytes()).hexdigest()
    review_file = tmp_path / "active.json"
    review_file.write_text(json.dumps(document))
    # An adjacent held review must not be discovered or read.
    (tmp_path / "held.json").write_text("invalid held material")
    output = tmp_path / "reviewed"
    summary = build_artifact(inventory, [review_file], output)
    assert summary["steps"] == 1
    stored = json.loads((output / "steps.json").read_text())
    assert stored[0]["agent_payload"] is None
    assert stored[0]["audit"]["semantic_certificate"] is False
    assert "PRIVATE_" not in (output / "steps.json").read_text()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["approved_recipe_count"] == 0
    assert len(provenance["active_review_artifacts"]) == 1
    for name, expected in provenance["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
        assert (output / name).stat().st_mode & 0o777 == 0o600
    before = (output / "steps.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_artifact(inventory, [review_file], output)
    assert (output / "steps.json").read_bytes() == before


def test_invalid_or_empty_review_bundle_leaves_no_output(tmp_path):
    source = row()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(source) + "\n")
    output = tmp_path / "reviewed"
    with pytest.raises(ValueError, match="Explicit"):
        build_artifact(inventory, [], output)
    review_file = tmp_path / "stale.json"
    review_file.write_text(json.dumps(artifact(source)))
    with pytest.raises(ValueError, match="inventory"):
        build_artifact(inventory, [review_file], output)
    assert not output.exists()


def row(code=None):
    return {
        "example_id": "run/exp-02",
        "model_input": {
            "task": {"benchmark": "fixture", "evaluation_n": 30},
            "plan": {"setup": {"notes": "Use fixed length, lr=1e-5, top_p=0.95."}},
            "code": [
                {
                    "role": "training",
                    "script_path": "train.py",
                    "status": "recovered",
                    "content": code or 'accuracy = 0.0\nprint("processor save failed")\n',
                }
            ],
            "known_previous_checkpoints": {"accuracy": "PRIVATE_SOURCE_HISTORY"},
        },
        "label": "PRIVATE_LABEL",
        "prior_observations": "PRIVATE_PRIOR",
    }


def artifact(source, *, path="/code/0/content", spans=(), status=None):
    compiled = compile_steps([source])[0][0]
    value = compiled["draft_input"]
    for part in path[1:].split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    findings = [f for f in compiled["audit"]["findings"] if f["path"] == path]
    assert findings
    redactions = []
    for text, kind in spans:
        start = value.index(text)
        redactions.append(
            {
                "start": start,
                "end": start + len(text),
                "span_sha256": text_digest(text),
                "reason": "Fixture review",
                "kind": kind,
                "preserve_executable_ast": True,
            }
        )
    return {
        "schema": "wm-source-fragment-review-v1",
        "scope": "source_fragments_only_not_whole_step",
        "reviewer": "fixture reviewer",
        "source_inventory_sha256": INVENTORY,
        "compiler_sha256": COMPILER,
        "reviews": [
            {
                "source_sha256": digest(value),
                "raw_text_sha256": text_digest(value),
                "review_status": status
                or ("redact_nonexecuting_text" if spans else "approve_findings"),
                "full_source_read": True,
                "rationale": "Reviewed full fixture source",
                "script_paths": ["train.py"],
                "redactions": redactions,
                "occurrences": [
                    {
                        "step_id": source["example_id"],
                        "path": path,
                        "finding_ids": [f["finding_id"] for f in findings],
                    }
                ],
            }
        ],
    }


def run(source, documents):
    return compile_reviewed_steps(
        [source], documents, source_inventory_sha256=INVENTORY, compiler_sha256=COMPILER
    )


def test_approval_is_private_and_preserves_source_without_reading_labels():
    source = row()
    original = copy.deepcopy(source)
    doc = artifact(source)
    result, summary = run(source, [doc])
    assert source == original
    reviewed = result[0]
    assert reviewed["draft_input"]["code"] == source["model_input"]["code"]
    assert reviewed["agent_payload"] is None
    assert reviewed["audit"]["semantic_certificate"] is False
    assert reviewed["audit"]["requires_whole_step_content_review"] is True
    assert summary["reviewed_occurrences"] == 1
    assert "PRIVATE_" not in json.dumps(result)
    assert reviewed["audit"]["source_recipe_projection_sha256"] == digest(
        {k: source["model_input"][k] for k in ("task", "plan", "code")}
    )
    source["label"] = object()
    source["prior_observations"] = object()
    source["model_input"]["known_previous_checkpoints"] = object()
    assert run(source, [doc]) == (result, summary)


def test_two_spans_for_one_finding_preserve_python_parameters():
    code = '"""The parent model failed. Fixed length is used. The parent collapsed."""\nlr=1e-5\ntop_p=0.95\n'
    source = row(code)
    spans = [
        ("The parent model failed. ", "docstring_content"),
        (" The parent collapsed.", "docstring_content"),
    ]
    doc = artifact(source, spans=spans)
    result, _ = run(source, [doc])
    edited = result[0]["draft_input"]["code"][0]["content"]
    assert edited == '"""Fixed length is used."""\nlr=1e-5\ntop_p=0.95\n'
    ast.parse(edited)
    assert not [
        f for f in result[0]["audit"]["unresolved_findings"] if f["path"].startswith("/code/")
    ]
    review = result[0]["audit"]["source_fragment_reviews"][0]
    assert len(review["redactions"]) == 2
    assert review["edited_sha256"] == digest(edited)
    assert review["whole_step_approval"] is False


def test_one_scanner_finding_supports_multiple_disjoint_comment_redactions():
    source = row("# parent failed one attempt\n# parent failed another attempt\nlr=1e-5\n")
    spans = [("parent failed one attempt", "comment"), ("parent failed another attempt", "comment")]
    doc = artifact(source, spans=spans)
    assert len(doc["reviews"][0]["occurrences"][0]["finding_ids"]) == 1
    result, _ = run(source, [doc])
    assert result[0]["draft_input"]["code"][0]["content"] == "# \n# \nlr=1e-5\n"


def test_unicode_and_adjacent_spans_are_character_offsets():
    source = row('"""é parent failed; model collapsed; fixed budget."""\nlr=1e-5\n')
    doc = artifact(
        source,
        spans=[
            ("parent failed; ", "docstring_content"),
            ("model collapsed; ", "docstring_content"),
        ],
    )
    result, _ = run(source, [doc])
    assert result[0]["draft_input"]["code"][0]["content"].startswith('"""é fixed budget.')


@pytest.mark.parametrize("field", ["source_sha256", "raw_text_sha256"])
def test_stale_source_rejected(field):
    source = row()
    doc = artifact(source)
    doc["reviews"][0][field] = "0" * 64
    with pytest.raises(ValueError, match="Stale reviewed source"):
        run(source, [doc])


@pytest.mark.parametrize("field", ["source_inventory_sha256", "compiler_sha256"])
def test_stale_artifact_context_rejected(field):
    source = row()
    doc = artifact(source)
    doc[field] = "0" * 64
    with pytest.raises(ValueError, match="Stale review"):
        run(source, [doc])


def test_span_hash_overlap_and_bounds_rejected():
    source = row("# parent failed\nlr=1e-5\n")
    doc = artifact(source, spans=[("parent failed", "comment")])
    broken = copy.deepcopy(doc)
    broken["reviews"][0]["redactions"][0]["span_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="span hash"):
        run(source, [broken])
    broken = copy.deepcopy(doc)
    broken["reviews"][0]["redactions"] *= 2
    with pytest.raises(ValueError, match="Overlapping"):
        run(source, [broken])
    broken = copy.deepcopy(doc)
    broken["reviews"][0]["redactions"][0]["start"] = True
    with pytest.raises(ValueError, match="bounds"):
        run(source, [broken])


@pytest.mark.parametrize(
    "code,text,kind",
    [
        ("# parent failed\nlr=1e-5\n", "lr=1e-5", "comment"),
        ('notes = "parent failed"\nlr=1e-5\n', "parent failed", "docstring_content"),
        ('"""parent failed"""\nlr=1e-5\n', '"""parent failed', "docstring_content"),
        ('"""parent failed"""\nx=1.00\n', "0", "comment"),
    ],
)
def test_code_edits_cannot_touch_executable_values_or_docstring_quotes(code, text, kind):
    source = row(code)
    doc = artifact(source, spans=[(text, kind)])
    with pytest.raises(ValueError, match="Python|comments/docstrings"):
        run(source, [doc])


def test_ast_change_from_comment_newline_join_is_rejected():
    source = row("# parent failed\nlr=1e-5\n")
    doc = artifact(source, spans=[("parent failed\n", "comment")])
    with pytest.raises(ValueError, match="executable Python AST"):
        run(source, [doc])


def test_invalid_existing_python_cannot_be_approved():
    source = row("def broken(:\n")
    with pytest.raises(ValueError, match="Python source is invalid"):
        run(source, [artifact(source)])


@pytest.mark.parametrize(
    "mutation", ["unknown_step", "unknown_path", "unknown_finding", "duplicate"]
)
def test_occurrences_must_be_unambiguous_and_known(mutation):
    source = row()
    doc = artifact(source)
    occ = doc["reviews"][0]["occurrences"][0]
    if mutation == "unknown_step":
        occ["step_id"] = "missing"
    elif mutation == "unknown_path":
        occ["path"] = "/code/9/content"
    elif mutation == "unknown_finding":
        occ["finding_ids"] = ["0" * 64]
    else:
        doc["reviews"][0]["occurrences"].append(copy.deepcopy(occ))
    with pytest.raises(ValueError, match="Unknown|Ambiguous"):
        run(source, [doc])


def test_plan_string_deletions_and_needs_review_leave_operations_intact():
    source = row()
    source["model_input"]["plan"]["setup"]["notes"] = (
        "The parent failed. Use lr=1e-5 and top_p=0.95."
    )
    doc = artifact(
        source, path="/plan/setup/notes", spans=[("The parent failed. ", "plan_narrative_span")]
    )
    result, _ = run(source, [doc])
    assert result[0]["draft_input"]["plan"]["setup"]["notes"] == "Use lr=1e-5 and top_p=0.95."
    assert result[0]["draft_input"]["code"] == source["model_input"]["code"]
    pending = artifact(source, path="/plan/setup/notes", status="needs_review")
    unchanged, _ = run(source, [pending])
    assert unchanged[0]["draft_input"]["plan"] == source["model_input"]["plan"]
    assert unchanged[0]["status"] == "needs_review"


def test_plan_field_removal_cannot_remove_list_items():
    source = row()
    doc = artifact(source, path="/plan/setup/notes", status="remove_field")
    result, _ = run(source, [doc])
    assert "notes" not in result[0]["draft_input"]["plan"]["setup"]
    source["model_input"]["plan"]["setup"]["notes"] = ["fixed length"]
    doc = artifact(source, path="/plan/setup/notes/0", status="remove_field")
    with pytest.raises(ValueError, match="plan dictionary leaf"):
        run(source, [doc])


def test_new_signal_created_by_deletion_stays_unresolved():
    source = row()
    source["model_input"]["plan"]["setup"]["notes"] = "parent fixed fXXailed"
    doc = artifact(source, path="/plan/setup/notes", spans=[("XX", "plan_narrative_span")])
    result, _ = run(source, [doc])
    assert any(
        "failed" in f["snippet"]
        for f in result[0]["audit"]["unresolved_findings"]
        if f["path"] == "/plan/setup/notes"
    )


def test_loader_hash_provenance_and_mutated_loaded_document(tmp_path):
    source = row()
    path = tmp_path / "review.json"
    path.write_text(json.dumps(artifact(source)))
    docs = load_review_documents([path])
    result, _ = run(source, docs)
    assert result[0]["audit"]["source_fragment_reviews"][0]["artifact_sha256"] == text_digest(
        path.read_text()
    )
    docs[0]["review_document"]["reviewer"] = "changed"
    with pytest.raises(ValueError, match="changed after loading"):
        run(source, docs)


def test_draft_hash_and_duplicate_steps_rejected():
    source = row()
    compiled = compile_steps([source])[0]
    with pytest.raises(ValueError, match="duplicate step"):
        apply_source_reviews(
            compiled * 2, [], source_inventory_sha256=INVENTORY, compiler_sha256=COMPILER
        )
    compiled[0]["draft_input"]["code"][0]["content"] += "# changed\n"
    with pytest.raises(ValueError, match="Stale compiler draft"):
        apply_source_reviews(
            compiled, [], source_inventory_sha256=INVENTORY, compiler_sha256=COMPILER
        )


def test_duplicate_artifact_occurrences_without_review_ids_are_rejected():
    source = row()
    doc = artifact(source)
    with pytest.raises(ValueError, match="Ambiguous duplicate source occurrence"):
        run(source, [doc, copy.deepcopy(doc)])


def test_explicit_pending_supersession_is_order_independent_and_bound():
    source = row()
    pending = artifact(source, status="needs_review")
    pending["reviews"][0]["review_id"] = "pending-1"
    replacement = artifact(source)
    replacement["reviews"][0].update(review_id="resolved-1", supersedes_review_id="pending-1")
    for docs in ([pending, replacement], [replacement, pending]):
        results, audit = run(source, docs)
        assert audit["superseded_reviews"] == {"pending-1": "resolved-1"}
        assert audit["reviewed_occurrences"] == 1
        assert (
            results[0]["audit"]["source_fragment_reviews"][0]["supersedes_review_id"] == "pending-1"
        )
        assert not [
            f for f in results[0]["audit"]["unresolved_findings"] if f["path"].startswith("/code/")
        ]
    wrong = copy.deepcopy(replacement)
    wrong["reviews"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="supersession"):
        run(source, [pending, wrong])
    with pytest.raises(ValueError, match="supersession"):
        run(source, [replacement])


def test_partial_approval_leaves_other_existing_findings_open():
    source = row()
    doc = artifact(source)
    occurrence = doc["reviews"][0]["occurrences"][0]
    all_ids = occurrence["finding_ids"][:]
    assert len(all_ids) > 1
    occurrence["finding_ids"] = all_ids[:1]
    results, _ = run(source, [doc])
    remaining = {f["finding_id"] for f in results[0]["audit"]["unresolved_findings"]}
    assert set(all_ids[1:]) <= remaining


def test_failing_late_artifact_does_not_mutate_supplied_drafts():
    source = row()
    compiled = compile_steps([source])[0]
    before = copy.deepcopy(compiled)
    first = artifact(source)
    second = artifact(source, path="/plan/setup/notes")
    second["reviews"][0]["raw_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Stale reviewed source"):
        apply_source_reviews(
            compiled, [first, second], source_inventory_sha256=INVENTORY, compiler_sha256=COMPILER
        )
    assert compiled == before


def manual_fixture(code=None):
    source = row(code or "lr=1e-5\ntop_p=0.95\n")
    source["model_input"]["plan"]["setup"]["notes"] = "Plain forward recipe."
    source["model_input"]["plan"]["hypothesis"] = {
        "claim": "The original system produces no correct answers. Use a new recipe."
    }
    return compile_steps([source])[0]


def manual_artifact(steps, *, path="/plan/hypothesis/claim", spans=(), status=None):
    value = steps[0]["draft_input"]
    for part in path[1:].split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    redactions = []
    for text, kind in spans:
        start = value.index(text)
        redactions.append(
            {
                "start": start,
                "end": start + len(text),
                "span_sha256": text_digest(text),
                "reason": "Manual fixture evidence",
                "kind": kind,
                "preserve_executable_ast": True,
            }
        )
    return {
        "schema": "wm-manual-source-field-review-v1",
        "scope": "manual_source_fields_only_not_whole_step",
        "reviewer": "manual fixture reviewer",
        "source_inventory_sha256": INVENTORY,
        "compiler_sha256": COMPILER,
        "source_drafts_sha256": review_drafts_digest(steps),
        "reviews": [
            {
                "review_id": "manual-fixture-1",
                "source_sha256": digest(value),
                "raw_text_sha256": text_digest(value),
                "full_source_read": True,
                "outcome_free": True,
                "executable_recipe_preserved": True,
                "evidence_sha256": text_digest("independent full-source fixture review"),
                "rationale": "This explicit manual decision applies only to the complete source field.",
                "review_status": status
                or ("redact_nonexecuting_text" if spans else "approve_source_field"),
                "script_paths": ["train.py"],
                "redactions": redactions,
                "occurrences": [{"step_id": steps[0]["step_id"], "path": path}],
            }
        ],
    }


def manual_run(steps, docs):
    return apply_manual_source_reviews(steps, docs, source_inventory_sha256=INVENTORY)


def test_manual_unflagged_plan_source_is_bound_and_never_whole_step_approved():
    steps = manual_fixture()
    before = copy.deepcopy(steps)
    assert steps[0]["audit"]["unresolved_findings"] == []
    doc = manual_artifact(
        steps, spans=[("The original system produces no correct answers. ", "plan_narrative_span")]
    )
    result, audit = manual_run(steps, [doc])
    assert steps == before
    assert result[0]["draft_input"]["plan"]["hypothesis"]["claim"] == "Use a new recipe."
    assert result[0]["agent_payload"] is None
    assert result[0]["audit"]["semantic_certificate"] is False
    assert result[0]["audit"]["requires_whole_step_content_review"] is True
    review = result[0]["audit"]["source_fragment_reviews"][0]
    assert review["finding_ids"] == [] and review["manual_source_review"] is True
    assert review["evidence_sha256"] == doc["reviews"][0]["evidence_sha256"]
    assert audit["source_drafts_sha256"] != audit["edited_drafts_sha256"]


def test_manual_multispan_unflagged_code_preserves_numeric_parameters():
    steps = manual_fixture(
        '"""é Original system emits no answers. It produces no correct solutions."""\nlr=1e-5\ntop_p=0.95\n'
    )
    doc = manual_artifact(
        steps,
        path="/code/0/content",
        spans=[
            ("Original system emits no answers. ", "docstring_content"),
            ("It produces no correct solutions.", "docstring_content"),
        ],
    )
    result, _ = manual_run(steps, [doc])
    assert result[0]["draft_input"]["code"][0]["content"] == '"""é """\nlr=1e-5\ntop_p=0.95\n'
    assert result[0]["audit"]["unresolved_findings"] == []


@pytest.mark.parametrize("key", ["finding_ids", "fake_finding_ids"])
def test_manual_cannot_bypass_current_scanner_review(key):
    steps = manual_fixture("accuracy=0\n")
    doc = manual_artifact(steps, path="/code/0/content")
    if key == "finding_ids":
        doc["reviews"][0]["occurrences"][0][key] = []
        message = "must not pretend"
    else:
        message = "CURRENT unresolved"
    with pytest.raises(ValueError, match=message):
        manual_run(steps, [doc])


@pytest.mark.parametrize(
    "key",
    [
        "source_sha256",
        "raw_text_sha256",
        "evidence_sha256",
        "outcome_free",
        "executable_recipe_preserved",
        "full_source_read",
    ],
)
def test_manual_source_hashes_evidence_and_attestations_are_required(key):
    steps = manual_fixture()
    doc = manual_artifact(steps)
    doc["reviews"][0][key] = "0" * 64 if key in {"source_sha256", "raw_text_sha256"} else None
    with pytest.raises(ValueError):
        manual_run(steps, [doc])


def test_manual_stage_binding_rejects_unrelated_field_change_and_partial_stage():
    steps = manual_fixture()
    doc = manual_artifact(steps)
    changed = copy.deepcopy(steps)
    changed[0]["draft_input"]["task"]["evaluation_n"] = 50
    changed[0]["audit"]["draft_input_sha256"] = digest(changed[0]["draft_input"])
    with pytest.raises(ValueError, match="draft-stage"):
        manual_run(changed, [doc])
    other = copy.deepcopy(steps[0])
    other["step_id"] = "another-step"
    larger = steps + [other]
    doc["source_drafts_sha256"] = review_drafts_digest(larger)
    assert review_drafts_digest(larger) == review_drafts_digest(list(reversed(larger)))
    with pytest.raises(ValueError, match="draft-stage"):
        manual_run(steps, [doc])


def test_manual_invalid_spans_and_executable_code_edits_rejected():
    steps = manual_fixture("# Original system emits no answers.\nlr=1e-5\n")
    doc = manual_artifact(
        steps, path="/code/0/content", spans=[("Original system emits no answers.", "comment")]
    )
    wrong = copy.deepcopy(doc)
    wrong["reviews"][0]["redactions"][0]["span_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="span hash"):
        manual_run(steps, [wrong])
    overlap = copy.deepcopy(doc)
    overlap["reviews"][0]["redactions"] *= 2
    with pytest.raises(ValueError, match="Overlapping"):
        manual_run(steps, [overlap])
    executable = manual_artifact(steps, path="/code/0/content", spans=[("lr=1e-5", "comment")])
    with pytest.raises(ValueError, match="comments/docstrings"):
        manual_run(steps, [executable])


def test_manual_new_concatenation_match_stays_open_and_holds_are_explicit():
    steps = manual_fixture()
    steps[0]["draft_input"]["plan"]["hypothesis"]["claim"] = "fXXailed"
    steps[0]["audit"]["draft_input_sha256"] = digest(steps[0]["draft_input"])
    doc = manual_artifact(steps, spans=[("XX", "plan_narrative_span")])
    result, audit = manual_run(steps, [doc])
    assert result[0]["status"] == "needs_review" and audit["residual_findings"] == 1
    pending = manual_artifact(steps, status="needs_review")
    result, audit = manual_run(steps, [pending])
    assert result[0]["status"] == "needs_review" and audit["manual_holds"] == 1
    assert audit["residual_findings"] == 0


def test_manual_carries_proven_previously_closed_false_positive_signals():
    source = row(
        '# Original system emits no answers.\nprint("processor save failed")\nmodel=None\nlr=1e-5\n'
    )
    source["model_input"]["plan"]["setup"]["notes"] = "Plain recipe."
    first, _ = run(source, [artifact(source)])
    assert first[0]["audit"]["unresolved_findings"] == []
    doc = manual_artifact(
        first, path="/code/0/content", spans=[("Original system emits no answers.", "comment")]
    )
    result, _ = manual_run(first, [doc])
    assert result[0]["audit"]["unresolved_findings"] == []
    assert 'print("processor save failed")' in result[0]["draft_input"]["code"][0]["content"]


def test_manual_bundle_is_new_private_stage_without_inventory_label_read(tmp_path):
    steps = manual_fixture()
    source = tmp_path / "source"
    source.mkdir()
    raw = json.dumps(steps).encode()
    (source / "steps.json").write_bytes(raw)
    (source / "provenance.json").write_text(
        json.dumps(
            {
                "files": {"steps.json": hashlib.sha256(raw).hexdigest()},
                "inventory_sha256": INVENTORY,
                "inventory_path": "MUST_NOT_READ.jsonl",
            }
        )
    )
    review = tmp_path / "manual.json"
    review.write_text(json.dumps(manual_artifact(steps)))
    output = tmp_path / "manual-stage"
    audit = build_manual_artifact(source, [review], output)
    assert audit["reviewed_occurrences"] == 1
    assert (output / "steps.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((output / "provenance.json").read_text())["approved_recipe_count"] == 0
    with pytest.raises(FileExistsError):
        build_manual_artifact(source, [review], output)


NUMERIC_DATA = "/plan/setup/data/0/n_examples"
NUMERIC_PROGRESS = "/plan/setup/progress/total"


def numeric_fixture(value=17):
    steps = manual_fixture("rows = load_dataset()\ntrain(rows, epochs=3, batch_size=8)\n")
    steps[0]["draft_input"]["plan"]["setup"].update(
        {
            "data": [{"n_examples": value, "configured_quota": 100}],
            "progress": {"total": 6, "epochs": 3},
            "command": "train --epochs 3 --batch-size 8",
            "learning_rate": 1e-5,
        }
    )
    steps[0]["audit"]["draft_input_sha256"] = digest(steps[0]["draft_input"])
    return steps


def numeric_artifact(steps, path=NUMERIC_DATA):
    step = steps[0]
    value = step["draft_input"]
    for part in path[1:].split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    content = step["draft_input"]["code"][0]["content"]
    evidence = {
        "step_id": step["step_id"],
        "source_step_sha256": digest(step["draft_input"]),
        "source_projection_sha256": step["audit"].get("source_recipe_projection_sha256")
        or step["audit"]["source_input_sha256"],
        "classification": "observed_data_yield"
        if path.endswith("n_examples")
        else "observed_dependent_schedule",
        "observed_not_configured_budget": True,
        "supporting_sources": [
            {
                "kind": "draft_text",
                "path": "/code/0/content",
                "source_sha256": digest(content),
                "raw_text_sha256": text_digest(content),
                "start": 0,
                "end": len(content),
                "text": content,
                "span_sha256": text_digest(content),
            },
            {
                "kind": "private_text",
                "source_locator": "/MUST_NOT_OPEN/private-source#selection",
                "source_artifact_sha256": text_digest("fixture private artifact"),
                "content": "PRIVATE_EVIDENCE: the retained yield was observed, not a quota.",
                "content_sha256": text_digest(
                    "PRIVATE_EVIDENCE: the retained yield was observed, not a quota."
                ),
            },
        ],
    }
    return {
        "schema": "wm-observed-numeric-metadata-review-v1",
        "scope": "observed_numeric_metadata_deletion_only_not_whole_step",
        "reviewer": "numeric fixture reviewer",
        "source_inventory_sha256": INVENTORY,
        "compiler_sha256": COMPILER,
        "source_drafts_sha256": review_drafts_digest(steps),
        "private_selection_notes": "Explicit source review, not automated classification.",
        "reviews": [
            {
                "review_id": "numeric-" + path.rsplit("/", 1)[-1],
                "step_id": step["step_id"],
                "path": path,
                "review_status": "delete_observed_numeric_metadata",
                "source_step_sha256": digest(step["draft_input"]),
                "source_type": type(value).__name__,
                "source_sha256": digest({"type": type(value).__name__, "value": value}),
                "full_source_read": True,
                "outcome_free": True,
                "executable_recipe_preserved": True,
                "rationale": "Explicit fixture decision: observed yield-derived metadata only.",
                "evidence": evidence,
                "evidence_sha256": digest(evidence),
            }
        ],
    }


def numeric_run(steps, docs):
    return apply_observed_numeric_reviews(
        steps, docs, source_inventory_sha256=INVENTORY, compiler_sha256=COMPILER
    )


@pytest.mark.parametrize("value", [0, 17, 0.0, 17.5, -(10**1000), 10**1000])
def test_numeric_deletion_is_typed_exact_and_preserves_every_other_input(value):
    steps = numeric_fixture(value)
    before = copy.deepcopy(steps)
    result, audit = numeric_run(steps, [numeric_artifact(steps)])
    expected = copy.deepcopy(before[0]["draft_input"])
    del expected["plan"]["setup"]["data"][0]["n_examples"]
    assert steps == before
    assert result[0]["draft_input"] == expected
    assert result[0]["agent_payload"] is None
    assert result[0]["audit"]["semantic_certificate"] is False
    assert result[0]["audit"]["requires_whole_step_content_review"] is True
    assert "PRIVATE_EVIDENCE" not in json.dumps(result[0]["draft_input"])
    recorded = result[0]["audit"]["observed_numeric_metadata_reviews"][0]
    assert type(recorded["removed_value"]) is type(value)
    assert recorded["source_sha256"] == observed_numeric_source_digest(value)
    assert audit["deleted_numeric_leaves"] == 1 and audit["approved_recipe_count"] == 0
    assert audit["automatic_numeric_classification"] is False


def test_numeric_progress_deletion_keeps_all_old_holds_findings_and_prior_audits():
    steps = numeric_fixture()
    audit = steps[0]["audit"]
    audit["manual_source_holds"] = [{"path": NUMERIC_PROGRESS, "private_reason": "keep me"}]
    audit["unresolved_findings"] = [{"path": NUMERIC_PROGRESS, "kind": "observed"}]
    audit["source_fragment_reviews"] = [{"prior": "private approval evidence"}]
    before = copy.deepcopy(audit)
    result, summary = numeric_run(steps, [numeric_artifact(steps, NUMERIC_PROGRESS)])
    assert result[0]["draft_input"]["plan"]["setup"]["progress"] == {"epochs": 3}
    for key in ("manual_source_holds", "unresolved_findings", "source_fragment_reviews"):
        assert result[0]["audit"][key] == before[key]
    assert summary["manual_holds"] == 1 and result[0]["status"] == "needs_review"
    steps[0]["status"] = "rejected"
    result, _ = numeric_run(steps, [numeric_artifact(steps)])
    assert result[0]["status"] == "rejected"


@pytest.mark.parametrize(
    "value", [True, False, None, "17", [], {}, float("nan"), float("inf"), -float("inf")]
)
def test_numeric_non_numbers_and_nonfinite_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        observed_numeric_source_digest(value)
    if type(value) is float:
        steps = numeric_fixture()
        doc = numeric_artifact(steps)
        steps[0]["draft_input"]["plan"]["setup"]["data"][0]["n_examples"] = value
        with pytest.raises(ValueError):
            numeric_run(steps, [doc])
        return
    steps = numeric_fixture(value)
    with pytest.raises(ValueError, match="finite"):
        numeric_run(steps, [numeric_artifact(steps)])


@pytest.mark.parametrize(
    "path",
    [
        "/plan/setup/data/00/n_examples",
        "/plan/setup/data/-1/n_examples",
        "/plan/setup/data/1e0/n_examples",
        "/plan/setup/data/9/n_examples",
        "/plan/setup/data/0/configured_quota",
        "/plan/setup/progress/epochs",
        "/plan/setup/learning_rate",
        "/plan/evaluation/n",
        "/task/evaluation_n",
        "/code/0/content",
        "/plan/setup/../progress/total",
        "/plan/setup/data/~00/n_examples",
        "/plan/setup/data/0/n_examples/",
        "/plan/setup/data/0/n_examples\n",
    ],
)
def test_numeric_path_allowlist_excludes_hyperparameters_budgets_and_ambiguous_indices(path):
    steps = numeric_fixture()
    doc = numeric_artifact(steps)
    doc["reviews"][0]["path"] = path
    with pytest.raises(ValueError):
        numeric_run(steps, [doc])


def test_numeric_list_indices_cannot_alias_dictionary_keys():
    steps = numeric_fixture()
    steps[0]["draft_input"]["plan"]["setup"]["data"] = {"0": {"n_examples": 17}}
    steps[0]["audit"]["draft_input_sha256"] = digest(steps[0]["draft_input"])
    with pytest.raises(ValueError, match="requires a list"):
        numeric_run(steps, [numeric_artifact(steps)])


@pytest.mark.parametrize(
    "key",
    [
        "source_inventory_sha256",
        "compiler_sha256",
        "source_drafts_sha256",
        "schema",
        "scope",
        "reviewer",
    ],
)
def test_numeric_document_bindings_are_required(key):
    steps = numeric_fixture()
    doc = numeric_artifact(steps)
    doc[key] = ""
    with pytest.raises(ValueError):
        numeric_run(steps, [doc])


@pytest.mark.parametrize(
    "key",
    [
        "source_step_sha256",
        "source_type",
        "source_sha256",
        "full_source_read",
        "outcome_free",
        "executable_recipe_preserved",
        "evidence_sha256",
        "rationale",
        "step_id",
        "review_status",
    ],
)
def test_numeric_source_bindings_and_attestations_are_required(key):
    steps = numeric_fixture()
    doc = numeric_artifact(steps)
    doc["reviews"][0][key] = ""
    with pytest.raises(ValueError):
        numeric_run(steps, [doc])


def test_numeric_int_and_float_digests_are_not_interchangeable():
    steps = numeric_fixture(17)
    doc = numeric_artifact(steps)
    doc["reviews"][0]["source_sha256"] = observed_numeric_source_digest(17.0)
    with pytest.raises(ValueError, match="typed"):
        numeric_run(steps, [doc])


@pytest.mark.parametrize(
    "key,value",
    [
        ("classification", "configured_budget"),
        ("classification", "observed_achieved_progress"),
        ("observed_not_configured_budget", False),
        ("step_id", "wrong-step"),
        ("source_step_sha256", "0" * 64),
        ("source_projection_sha256", "0" * 64),
        ("supporting_sources", []),
    ],
)
def test_numeric_bound_evidence_requires_explicit_observed_classification(key, value):
    steps = numeric_fixture()
    doc = numeric_artifact(steps)
    review = doc["reviews"][0]
    review["evidence"][key] = value
    review["evidence_sha256"] = digest(review["evidence"])
    with pytest.raises(ValueError):
        numeric_run(steps, [doc])


@pytest.mark.parametrize(
    "index,key,value",
    [
        (0, "source_sha256", "0" * 64),
        (0, "raw_text_sha256", "0" * 64),
        (0, "span_sha256", "0" * 64),
        (0, "text", "altered"),
        (0, "start", True),
        (0, "end", 9999),
        (0, "path", "/task/description"),
        (1, "content", "altered"),
        (1, "content_sha256", "0" * 64),
        (1, "source_artifact_sha256", ""),
        (1, "source_locator", ""),
        (1, "kind", "execute"),
    ],
)
def test_numeric_source_evidence_is_checked_even_when_outer_evidence_hash_is_recomputed(
    index, key, value
):
    steps = numeric_fixture()
    doc = numeric_artifact(steps)
    review = doc["reviews"][0]
    review["evidence"]["supporting_sources"][index][key] = value
    review["evidence_sha256"] = digest(review["evidence"])
    with pytest.raises(ValueError):
        numeric_run(steps, [doc])


@pytest.mark.parametrize("container", ["review", "evidence", "source"])
def test_numeric_replacement_or_unknown_fields_are_not_silently_ignored(container):
    steps = numeric_fixture()
    doc = numeric_artifact(steps)
    review = doc["reviews"][0]
    target = {
        "review": review,
        "evidence": review["evidence"],
        "source": review["evidence"]["supporting_sources"][0],
    }[container]
    target["replacement"] = 0
    review["evidence_sha256"] = digest(review["evidence"])
    with pytest.raises(ValueError, match="fields"):
        numeric_run(steps, [doc])


def test_numeric_duplicate_stale_partial_or_mutated_documents_fail_atomically():
    steps = numeric_fixture()
    before = copy.deepcopy(steps)
    doc = numeric_artifact(steps)
    other = numeric_artifact(steps, NUMERIC_PROGRESS)
    other["reviews"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        numeric_run(steps, [doc, other])
    assert steps == before
    for change_id in (False, True):
        duplicate = copy.deepcopy(doc)
        if change_id:
            duplicate["reviews"][0]["review_id"] += "-duplicate-path"
        with pytest.raises(ValueError, match="duplicate"):
            numeric_run(steps, [doc, duplicate])
    larger = steps + [copy.deepcopy(steps[0])]
    with pytest.raises(ValueError, match="duplicate step"):
        numeric_run(larger, [doc])
    larger[1]["step_id"] = "another-step"
    with pytest.raises(ValueError, match="source_drafts"):
        numeric_run(larger, [doc])
    changed = copy.deepcopy(steps)
    changed[0]["draft_input"]["task"]["evaluation_n"] = 50
    with pytest.raises(ValueError, match="Stale compiler"):
        numeric_run(changed, [doc])
    changed[0]["audit"]["draft_input_sha256"] = digest(changed[0]["draft_input"])
    with pytest.raises(ValueError, match="source_drafts"):
        numeric_run(changed, [doc])
    wrapped = {"review_document": doc, "document_sha256": "0" * 64}
    with pytest.raises(ValueError, match="after loading"):
        numeric_run(steps, [wrapped])


@pytest.mark.parametrize("alias_location", ["another_data_path", "another_step", "private_audit"])
def test_numeric_in_memory_mutable_alias_cannot_expand_a_single_deletion(alias_location):
    steps = numeric_fixture()
    shared = steps[0]["draft_input"]["plan"]["setup"]["data"][0]
    if alias_location == "another_data_path":
        steps[0]["draft_input"]["plan"]["setup"]["data"].append(shared)
    elif alias_location == "another_step":
        other = copy.deepcopy(steps[0])
        other["step_id"] = "another-step"
        other["draft_input"]["plan"]["setup"]["data"][0] = shared
        steps.append(other)
    else:
        steps[0]["audit"]["shared_original_source"] = shared
    for step in steps:
        step["audit"]["draft_input_sha256"] = digest(step["draft_input"])
    before = copy.deepcopy(steps)
    with pytest.raises(ValueError, match="Aliased"):
        numeric_run(steps, [numeric_artifact(steps)])
    assert steps == before and shared["n_examples"] == 17


def test_numeric_api_does_not_expand_legacy_string_review_permissions():
    steps = numeric_fixture()
    legacy = manual_artifact(steps)
    legacy["reviews"][0]["occurrences"][0]["path"] = NUMERIC_DATA
    with pytest.raises(TypeError, match="string source"):
        manual_run(steps, [legacy])
    with pytest.raises(ValueError, match="numeric"):
        numeric_run(steps, [legacy])
    steps[0]["agent_payload"] = {"not": "private"}
    with pytest.raises(ValueError, match="private"):
        numeric_run(steps, [numeric_artifact(steps)])


def numeric_bundle_fixture(tmp_path):
    steps = numeric_fixture()
    source = tmp_path / "source"
    source.mkdir()
    for name, value in (("steps.json", steps), ("audit.json", {"private_audit": "retained"})):
        (source / name).write_text(json.dumps(value))
    provenance = {
        "files": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("steps.json", "audit.json")
        },
        "inventory_sha256": INVENTORY,
        "inventory_path": "/MUST_NOT_READ/private-labels.jsonl",
        "compiler_sha256": COMPILER,
        "integration_source_sha256": text_digest("previous integrator, not current source"),
    }
    (source / "provenance.json").write_text(json.dumps(provenance))
    doc = numeric_artifact(steps)
    doc["source_drafts_file_sha256"] = provenance["files"]["steps.json"]
    doc["source_bundle_provenance_sha256"] = hashlib.sha256(
        (source / "provenance.json").read_bytes()
    ).hexdigest()
    review = tmp_path / "numeric.json"
    review.write_text(json.dumps(doc))
    return source, review, tmp_path / "numeric-stage"


def test_numeric_builder_is_new_private_hash_bound_stage_and_never_reads_labels(tmp_path):
    source, review, output = numeric_bundle_fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    summary = build_observed_numeric_artifact(source, [review], output)
    assert summary["deleted_numeric_leaves"] == 1
    assert output.stat().st_mode & 0o777 == 0o700
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["approved_recipe_count"] == 0
    for name, expected in provenance["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        build_observed_numeric_artifact(source, [review], output)


@pytest.mark.parametrize("target", ["steps.json", "audit.json", "provenance.json", "review"])
def test_numeric_builder_rejects_altered_source_or_review_files_before_writing(tmp_path, target):
    source, review, output = numeric_bundle_fixture(tmp_path)
    path = review if target == "review" else source / target
    path.write_text(path.read_text() + " ")
    if target == "review":
        doc = json.loads(path.read_text())
        doc["source_drafts_file_sha256"] = "0" * 64
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        build_observed_numeric_artifact(source, [review], output)
    assert not output.exists()


@pytest.mark.parametrize("target", ["source", "steps.json", "review", "output_parent"])
def test_numeric_builder_rejects_symlink_files_and_parent_directories(tmp_path, target):
    source, review, output = numeric_bundle_fixture(tmp_path)
    link = tmp_path / "alias"
    if target == "source":
        link.symlink_to(source, target_is_directory=True)
        source = link
    elif target == "steps.json":
        original = source / "steps.json"
        original.rename(source / "saved-steps.json")
        original.symlink_to(source / "saved-steps.json")
    elif target == "review":
        link.symlink_to(review)
        review = link
    else:
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "numeric-stage"
    with pytest.raises(ValueError, match="Symlinks"):
        build_observed_numeric_artifact(source, [review], output)
    assert not output.exists()


@pytest.mark.parametrize(
    "injection", ['"duplicate":1,"duplicate":2,', '"bad":NaN,', '"bad":Infinity,', '"bad":1e999,']
)
def test_numeric_builder_strict_json_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path, injection
):
    source, review, output = numeric_bundle_fixture(tmp_path)
    review.write_text("{" + injection + review.read_text()[1:])
    with pytest.raises(ValueError):
        build_observed_numeric_artifact(source, [review], output)
    assert not output.exists()


def test_numeric_builder_requires_explicit_docs_and_both_source_file_pins(tmp_path):
    source, review, output = numeric_bundle_fixture(tmp_path)
    with pytest.raises(ValueError, match="Explicit"):
        build_observed_numeric_artifact(source, [], output)
    doc = json.loads(review.read_text())
    del doc["source_bundle_provenance_sha256"]
    review.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="provenance file"):
        build_observed_numeric_artifact(source, [review], output)
    assert not output.exists()
