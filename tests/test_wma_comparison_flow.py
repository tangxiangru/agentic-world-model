"""Public candidate selection through the real client, sidecar and immutable records."""

from __future__ import annotations

import json
import sys
from itertools import count
from pathlib import Path

import pytest

from awm import wma_client
from awm.cli import main
from awm.exp_protocol import decisions
from awm.wma import backends, sidecar


def prepared_proposal(session: Path) -> tuple[Path, dict]:
    path = decisions.create_proposal(session)
    proposal = json.loads(path.read_text())
    proposal["situation"].update(remaining_h=3, incumbent="models/current",
                                  evidence=["earlier permitted evaluation: 70%"])
    for c in proposal["candidates"]:
        c.update(hypothesis="new examples may improve generalization",
                 parent_checkpoint="models/current", change=f"mixture {c['candidate_id']}",
                 train_h=1, eval_h=0.1, cost_basis="previous observed throughput",
                 evidence=["training composition summary"], uncertainty="new mixture unobserved",
                 decision_test="compare using the same permitted evaluator")
    path.write_text(json.dumps(proposal))
    assert decisions.load_proposal(session, proposal["decision_id"]) == proposal
    return path, proposal


def output_for(proposal):
    return {
        "schema_version": decisions.COMPARISON_SCHEMA,
        "decision_id": proposal["decision_id"],
        "proposal_sha256": decisions.proposal_sha(proposal),
        "ranking": ["B", "A"],
        "comparisons": [{"preferred": "B", "alternative": "A", "reason": "B tests a new mixture",
                         "evidence": ["shared training summary"], "uncertainty": "full-run scores unknown"}],
        "candidate_assessments": [
            {"candidate_id": cid, "feasibility": "ready", "expected_effect": "unknown endpoint",
             "opportunity_cost": "one hour unavailable to the other mixture", "uncertainty": "unobserved"}
            for cid in ("B", "A")
        ],
        "suggestions": [],
    }


class JointBackend(backends.Backend):
    name = "joint-stub"

    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def run(self, brief):
        self.calls.append(brief)
        proposal = json.loads(brief.card_path.read_text())
        output = output_for(proposal)
        if self.mutate:
            self.mutate(brief, output)
        brief.extra["measured"] = {"cost": {"usd": 0.02}, "access": {"files": 1, "outside": []}}
        brief.verdict_path.write_text(json.dumps(output))


def attach_worker(monkeypatch, session, private, backend):
    skill = private / "skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("Compare evidence in context.\n")
    config = sidecar.Config(
        session_dir=session, skill_dir=skill, history_dir=None, backend="claude",
        model="test-model", effort="high", budget=backends.Budget(wall_min=1, max_turns=3),
        jobs=3, private_output_dir=private / "output",
    )
    # Process the real queue at each client poll. No background timing race or model call.
    sidecar.run(config, once=True)
    monkeypatch.setattr(sidecar, "get_backend", lambda *args: backend)
    monkeypatch.setattr(wma_client.time, "sleep", lambda _seconds: sidecar.run(config, once=True))
    return config


def test_multi_self_freezes_choice_without_calling_attached_joint_backend(tmp_path, monkeypatch):
    from awm import sandbox
    session = tmp_path / "session"
    sandbox.setup(session, sha="test", exp_protocol=True, decision_mode="multi-self")
    _, proposal = prepared_proposal(session)
    backend = JointBackend()
    config = attach_worker(monkeypatch, session, tmp_path / "private", backend)
    result = wma_client.compare_and_wait(session, proposal["decision_id"], out=lambda _: None)
    sidecar.run(config, once=True)
    assert result["state"] == "not_requested" and result["sidecar_attached"] is True
    assert backend.calls == []
    assert list((session / ".wma/requests").iterdir()) == []
    choice = wma_client.record_choice(session, proposal["decision_id"], "A", "my own choice")
    assert json.loads(choice.read_text())["comparison_state"] == "not_requested"
    # Disabling joint comparison must leave the selected card's blocking review active.
    from exp_protocol_cards import plan_card

    from awm.exp_protocol import lock, schema
    card = plan_card()
    card_path = session / "memory/cards/exp-01.yaml"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    schema.dump_card(card_path, card)
    lock.write_lock(card_path, card, {})
    formal_calls = []

    def formal_backend(*_args):
        formal_calls.append("formal")
        return backends.HeuristicBackend()

    monkeypatch.setattr(sidecar, "get_backend", formal_backend)
    reviewed = wma_client.review_and_wait(session, "exp-01", timeout_min=0.1,
                                         out=lambda _: None, sleep=lambda _: sidecar.run(config, once=True))
    assert reviewed["state"] == "delivered"
    assert formal_calls == ["formal"] and backend.calls == []


def test_single_mode_refuses_to_create_a_fake_candidate_pool(tmp_path):
    from awm import sandbox
    sandbox.setup(tmp_path, sha="test", exp_protocol=True, decision_mode="single")
    with pytest.raises(ValueError, match="candidate pool"):
        decisions.create_proposal(tmp_path)
    with pytest.raises(ValueError, match="not configured"):
        wma_client.compare_and_wait(tmp_path, "decision-01")
    assert not (tmp_path / ".wma/comparisons").exists()


def test_self_completion_cannot_be_reused_after_switching_mode(tmp_path):
    from awm import sandbox
    from awm.exp_protocol import treatment
    sandbox.setup(tmp_path, sha="test", exp_protocol=True, decision_mode="multi-self")
    _, proposal = prepared_proposal(tmp_path)
    wma_client.compare_and_wait(tmp_path, proposal["decision_id"], out=lambda _: None)
    path = tmp_path / "awm_sandbox.json"
    value = json.loads(path.read_text())
    value.update(decision_mode="multi-joint",
                 decision_mode_sha256=treatment.describe("multi-joint", explicit=True)["sha256"])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="mode changed"):
        wma_client.record_choice(tmp_path, proposal["decision_id"], "A", "cannot reuse old request")


def test_joint_selection_flows_through_public_request_sidecar_and_advisory_choice(tmp_path, monkeypatch):
    session = tmp_path / "session"
    _, proposal = prepared_proposal(session)
    backend = JointBackend()
    attach_worker(monkeypatch, session, tmp_path / "private", backend)
    messages = []
    result = wma_client.compare_and_wait(session, "decision-01", timeout_min=0.1, out=messages.append)
    assert result["state"] == "completed" and len(backend.calls) == 1
    archive = session / ".wma/comparisons" / result["request_id"]
    request = json.loads((archive / "request.json").read_text())
    comparison = json.loads((archive / "comparison.json").read_text())
    assert request["proposal"] == proposal
    assert comparison["ranking"] == ["B", "A"]
    assert comparison["request_id"] == result["request_id"]
    assert comparison["cost"]["usd"] == 0.02
    assert (session / ".wma/processed" / f"{result['request_id']}.json").exists()
    assert any("B > A" in line for line in messages)
    # The scientist can disagree; comparison and choice confer no launch permission.
    choice_path = wma_client.record_choice(session, "decision-01", "A", "A has lower preparation risk", "exp-01")
    choice = json.loads(choice_path.read_text())
    assert choice["scientist_preference"] == "A" and choice["selected"] == "A"
    assert choice["comparison_state"] == "completed"
    assert choice["request_id"] == request["request_id"]
    assert not (session / "memory/cards").exists()


def test_control_prepares_same_candidates_and_archives_its_unassisted_choice(tmp_path):
    session = tmp_path / "control"
    _, proposal = prepared_proposal(session)
    result = wma_client.compare_and_wait(session, "decision-01", out=lambda _: None)
    assert result["state"] == "not_attached"
    assert not wma_client.sidecar_attached(session)
    choice = json.loads(wma_client.record_choice(session, "decision-01", "B", "B tests new examples").read_text())
    assert choice["comparison_state"] == "not_attached"
    assert choice["scientist_preference"] == proposal["scientist_preference"] == "A"
    archive = session / ".wma/comparisons" / result["request_id"]
    assert json.loads((archive / "request.json").read_text())["proposal"] == proposal
    assert not (archive / "comparison.json").exists()


def test_live_proposal_edits_require_a_new_comparison_before_choice(tmp_path, monkeypatch):
    session = tmp_path / "session"
    path, original = prepared_proposal(session)

    def edit_draft(_brief, _output):
        changed = json.loads(path.read_text())
        changed["candidates"][0]["train_h"] = 2
        path.write_text(json.dumps(changed))

    backend = JointBackend(edit_draft)
    attach_worker(monkeypatch, session, tmp_path / "private", backend)
    result = wma_client.compare_and_wait(session, "decision-01", timeout_min=0.1, out=lambda _: None)
    assert result["state"] == "completed"
    comparison = json.loads(Path(result["comparison_path"]).read_text())
    assert comparison["proposal_sha256"] == decisions.proposal_sha(original)
    with pytest.raises(ValueError, match="proposal changed"):
        wma_client.record_choice(session, "decision-01", "B", "use the old recommendation")


@pytest.mark.parametrize("failure", ["bad_ranking", "backend_error"])
def test_failed_comparison_preserves_attempt_and_allows_an_explicit_choice(tmp_path, monkeypatch, failure):
    session = tmp_path / "session"
    prepared_proposal(session)

    def fail(_brief, output):
        if failure == "backend_error":
            raise backends.BackendError("model request failed")
        output["ranking"] = ["B", "B"]

    backend = JointBackend(fail)
    private = tmp_path / "private"
    attach_worker(monkeypatch, session, private, backend)
    result = wma_client.compare_and_wait(session, "decision-01", timeout_min=0.1, out=lambda _: None)
    assert result["state"] == "failed" and len(backend.calls) == 1
    archive = session / ".wma/comparisons" / result["request_id"]
    assert json.loads((archive / "completion.json").read_text())["state"] == "failed"
    assert not (archive / "comparison.json").exists()
    assert len(list(private.rglob("failure.json"))) == 1
    choice_path = wma_client.record_choice(session, "decision-01", "A", "WMA unavailable; retain original preference")
    assert json.loads(choice_path.read_text())["comparison_state"] == "failed"


def test_timeout_keeps_choice_bound_to_the_returned_timeout_even_after_late_delivery(tmp_path, monkeypatch):
    session = tmp_path / "session"
    prepared_proposal(session)
    backend = JointBackend()
    config = attach_worker(monkeypatch, session, tmp_path / "private", backend)
    clock = count(100.0)
    monkeypatch.setattr(wma_client.time, "monotonic", lambda: next(clock))
    result = wma_client.compare_and_wait(session, "decision-01", timeout_min=0.001, out=lambda _: None)
    assert result["state"] == "timeout" and not backend.calls
    choice_path = wma_client.record_choice(session, "decision-01", "A", "budget does not permit waiting")
    before = choice_path.read_bytes()
    sidecar.run(config, once=True)
    archive = session / ".wma/comparisons" / result["request_id"]
    assert (archive / "comparison.json").exists()
    assert json.loads((archive / "completion.json").read_text())["state"] == "timeout"
    assert choice_path.read_bytes() == before
    assert json.loads(before)["comparison_state"] == "timeout"


def test_choose_requires_a_comparison_attempt_and_a_real_candidate(tmp_path):
    session = tmp_path / "session"
    prepared_proposal(session)
    with pytest.raises(ValueError, match="compare the proposal set first"):
        wma_client.record_choice(session, "decision-01", "A", "choose without an attempt")
    wma_client.compare_and_wait(session, "decision-01", out=lambda _: None)
    with pytest.raises(ValueError, match="existing candidate"):
        wma_client.record_choice(session, "decision-01", "missing", "not proposed")


def test_public_decision_commands_register_with_private_package_present(tmp_path, capsys):
    session = tmp_path / "session"
    assert main(["wma", "propose", "--dir", str(session)]) == 0
    path = decisions.proposal_path(session, "decision-01")
    assert path.exists()
    _, proposal = prepared_proposal(tmp_path / "seed")
    path.write_text(json.dumps(proposal))
    assert main(["wma", "compare", "--dir", str(session), "decision-01"]) == 0
    assert main(["wma", "choose", "--dir", str(session), "decision-01", "--candidate", "B",
                 "--reason", "B is worth testing"]) == 0
    choices = list((session / "memory/decisions/decision-01/choices").glob("*.json"))
    assert len(choices) == 1 and json.loads(choices[0].read_text())["selected"] == "B"
    assert "no world-model agent" in capsys.readouterr().out


def test_command_backend_accepts_standalone_comparison_and_reports_its_measured_cost(tmp_path, monkeypatch):
    session = tmp_path / "session"
    _, proposal = prepared_proposal(session)
    output = output_for(proposal)
    output["cost"] = {"usd": 999}
    script = tmp_path / "stub_cli.py"
    marker = tmp_path / "calls.txt"
    script.write_text(
        "import json, re, sys\nfrom pathlib import Path\n"
        "prompt = sys.stdin.read()\n"
        "target = re.search(r'Write ONE JSON object to exactly (.+?) and no other file', prompt).group(1)\n"
        f"Path(target).write_text(json.dumps({output!r}))\n"
        f"with Path({str(marker)!r}).open('a') as f: f.write('called\\n')\n"
        "print(json.dumps({'type': 'result', 'total_cost_usd': 0.025, 'num_turns': 2}))\n"
    )
    backend = backends.CommandBackend("stub-cli", [sys.executable, str(script)],
                                      model="actual-stub-model", effort="high", transcript="stream-json")
    private = tmp_path / "private"
    attach_worker(monkeypatch, session, private, backend)
    result = wma_client.compare_and_wait(session, "decision-01", timeout_min=0.1, out=lambda _: None)
    assert result["state"] == "completed"
    comparison = json.loads(Path(result["comparison_path"]).read_text())
    assert comparison["schema_version"] == decisions.COMPARISON_SCHEMA
    assert "levels" not in comparison
    assert comparison["model"] == "actual-stub-model"
    assert comparison["cost"]["usd"] == 0.025 and comparison["cost"]["turns"] == 2
    assert marker.read_text() == "called\n"
    assert len(list(private.rglob("comparison.transcript.jsonl"))) == 1
