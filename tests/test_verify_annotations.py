"""The floor under agent judgement: a pointer that does not resolve is rejected."""

from __future__ import annotations

import gzip
import json

import pytest

from awm.traj import verify_annotations as va


@pytest.fixture
def stream(tmp_path):
    """A two-event run, written where the checker expects to find it."""
    events = [
        {"run_id": "r", "agent_id": "main", "i": 5, "type": "tool_use", "role": "assistant",
         "tool": "Bash", "args": {"command": "nohup python train_sft.py --out ckpt/sft1 &"}},
        {"run_id": "r", "agent_id": "main", "i": 9, "type": "tool_result", "role": "user",
         "text": '{"accuracy": 0.93, "stderr": 0.02}'},
    ]
    path = tmp_path / "r.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return tmp_path


def annotation(**tables) -> dict:
    return {"run_id": "r", **tables}


class TestAcceptance:
    def test_a_fragment_present_in_the_named_event_passes(self, stream) -> None:
        a = annotation(changes=[
            {"change_id": "c1", "i": 5, "category": "C4", "summary": "launch SFT",
             "evidence": [[5, "train_sft.py --out ckpt/sft1"]]},
        ])
        r = va.check_annotation(a, stream)
        assert r.ok, r.problems
        assert r.checked == 1

    def test_a_fragment_quoted_from_a_result_passes(self, stream) -> None:
        a = annotation(changes=[
            {"change_id": "c1", "i": 9, "category": "C1", "summary": "read the score",
             "evidence": [[9, '"accuracy": 0.93']]},
        ])
        assert va.check_annotation(a, stream).ok

    def test_the_pipe_escaping_the_brief_adds_is_tolerated(self, stream) -> None:
        """Agents quote from the rendered table, where `|` is written `\\|`."""
        a = annotation(changes=[
            {"change_id": "c1", "i": 5, "category": "C4", "summary": "x",
             "evidence": [[5, "python train_sft.py"]]},
        ])
        assert va.check_annotation(a, stream).ok


class TestRejection:
    def test_an_event_that_does_not_exist(self, stream) -> None:
        a = annotation(changes=[
            {"change_id": "c1", "i": 999, "category": "C4", "summary": "x",
             "evidence": [[999, "train_sft.py --out ckpt/sft1"]]},
        ])
        r = va.check_annotation(a, stream)
        assert [p.reason for p in r.problems] == ["no_such_event"]

    def test_a_fragment_that_is_not_in_that_event(self, stream) -> None:
        """The classic fabrication: a real event, a quote from nowhere."""
        a = annotation(changes=[
            {"change_id": "c1", "i": 5, "category": "C4", "summary": "x",
             "evidence": [[5, "--learning-rate 3e-5 --warmup 100"]]},
        ])
        r = va.check_annotation(a, stream)
        assert [p.reason for p in r.problems] == ["fragment_not_found"]

    def test_a_fragment_from_the_wrong_event(self, stream) -> None:
        a = annotation(changes=[
            {"change_id": "c1", "i": 5, "category": "C1", "summary": "x",
             "evidence": [[5, '"accuracy": 0.93']]},
        ])
        assert [p.reason for p in va.check_annotation(a, stream).problems] == [
            "fragment_not_found"
        ]

    def test_no_evidence_at_all(self, stream) -> None:
        a = annotation(changes=[
            {"change_id": "c1", "i": 5, "category": "C4", "summary": "x", "evidence": []},
        ])
        assert [p.reason for p in va.check_annotation(a, stream).problems] == ["no_evidence"]

    def test_a_fragment_too_short_to_mean_anything(self, stream) -> None:
        a = annotation(changes=[
            {"change_id": "c1", "i": 5, "category": "C4", "summary": "x",
             "evidence": [[5, "py"]]},
        ])
        assert [p.reason for p in va.check_annotation(a, stream).problems] == [
            "fragment_too_short"
        ]

    def test_a_verification_pointing_at_no_event(self, stream) -> None:
        a = annotation(verifications=[{"i": 404, "judges_changes": [], "outcome_read": "x"}])
        assert [p.reason for p in va.check_annotation(a, stream).problems] == ["no_such_event"]


class TestScope:
    def test_trainings_and_proposals_are_evidenced_too(self, stream) -> None:
        a = annotation(
            trainings=[{"i": 5, "tested_variable": "C4", "vs_previous": "baseline",
                        "evidence": [[5, "train_sft.py"]]}],
            proposed_category=[{"slug": "x", "why": "y", "evidence": [[5, "nohup python"]]}],
        )
        r = va.check_annotation(a, stream)
        assert r.ok, r.problems
        assert r.checked == 2

    def test_an_empty_annotation_is_vacuously_fine(self, stream) -> None:
        r = va.check_annotation(annotation(), stream)
        assert r.ok
        assert r.checked == 0
