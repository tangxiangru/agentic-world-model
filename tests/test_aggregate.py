"""Agreement decides what may be quoted, so the arithmetic has to be right."""

from __future__ import annotations

import gzip
import json

import pytest

from awm.traj import aggregate as agg


class TestKappa:
    def test_perfect_agreement(self) -> None:
        k = agg.cohens_kappa([("C3", "C3"), ("C4", "C4"), ("both", "both")])
        assert k.kappa == pytest.approx(1.0)

    def test_agreement_no_better_than_chance_is_zero(self) -> None:
        """Two annotators using the same label distribution independently."""
        pairs = [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
        assert agg.cohens_kappa(pairs).kappa == pytest.approx(0.0)

    def test_one_label_throughout_is_not_agreement(self) -> None:
        """The lopsided case kappa exists for: both say `both` every time."""
        k = agg.cohens_kappa([("both", "both")] * 20)
        assert k.observed == 1.0
        assert k.kappa != k.kappa, "degenerate agreement must not read as kappa=1"

    def test_high_raw_agreement_on_a_skewed_field_scores_lower(self) -> None:
        pairs = [("both", "both")] * 18 + [("C3", "C4"), ("C4", "C3")]
        k = agg.cohens_kappa(pairs)
        assert k.observed == pytest.approx(0.9)
        assert k.kappa < 0.7, "chance correction must bite on a skewed field"

    def test_empty_is_none(self) -> None:
        assert agg.cohens_kappa([]) is None


class TestVerdict:
    @pytest.mark.parametrize(
        "kappa,expected",
        [(0.95, "quotable"), (0.8, "quotable"), (0.7, "findings_only"),
         (0.6, "findings_only"), (0.4, "discard")],
    )
    def test_thresholds(self, kappa: float, expected: str) -> None:
        assert agg.Agreement("f", 10, 0.9, 0.5, kappa).verdict == expected


class TestJaccard:
    def test_identical_link_sets(self) -> None:
        assert agg.jaccard([({"c1", "c2"}, {"c1", "c2"})]) == pytest.approx(1.0)

    def test_two_empties_agree(self) -> None:
        """Both annotators saying "this judged nothing" is agreement, not a hole."""
        assert agg.jaccard([(set(), set())]) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        assert agg.jaccard([({"c1", "c2"}, {"c2", "c3"})]) == pytest.approx(1 / 3)


@pytest.fixture
def batch(tmp_path):
    """Two annotators over one run, with one unresolvable pointer planted."""
    events = [
        {"run_id": "r", "agent_id": "main", "i": 5, "type": "tool_use", "role": "assistant",
         "tool": "Bash", "args": {"command": "python train_sft.py --out ckpt/a --lr 1e-5"}},
        {"run_id": "r", "agent_id": "main", "i": 7, "type": "tool_use", "role": "assistant",
         "tool": "Bash", "args": {"command": "python evaluate.py --model-path ckpt/a --limit 150"}},
    ]
    root = tmp_path / "events"
    root.mkdir()
    with gzip.open(root / "r.jsonl.gz", "wt") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")

    first = tmp_path / "first"
    second = tmp_path / "second"
    for d in (first, second):
        d.mkdir()

    (first / "r.json").write_text(json.dumps({
        "run_id": "r",
        "changes": [
            {"change_id": "c1", "i": 5, "category": "C4", "summary": "sft",
             "evidence": [[5, "train_sft.py --out ckpt/a"]]},
            {"change_id": "c2", "i": 7, "category": "C1", "summary": "invented",
             "evidence": [[7, "--temperature 0.0 --top-p 0.9"]]},
        ],
        "trainings": [{"i": 5, "tested_variable": "C4", "vs_previous": "baseline",
                       "evidence": [[5, "--lr 1e-5"]]}],
        "verifications": [{"i": 7, "judges_changes": ["c1"], "outcome_read": "0.9"}],
    }), encoding="utf-8")

    (second / "r.json").write_text(json.dumps({
        "run_id": "r",
        "changes": [{"change_id": "x1", "i": 5, "category": "C4", "summary": "sft",
                     "evidence": [[5, "python train_sft.py"]]}],
        "trainings": [{"i": 5, "tested_variable": "both", "vs_previous": "baseline",
                       "evidence": [[5, "--out ckpt/a"]]}],
        "verifications": [{"i": 7, "judges_changes": ["x1"], "outcome_read": "0.9"}],
    }), encoding="utf-8")
    return root, first, second


class TestBatch:
    def test_a_row_with_an_unresolvable_pointer_is_dropped(self, batch) -> None:
        root, first, _ = batch
        ann, rep = agg.load_batch(first, root)
        t = agg.tables(ann, rep)
        assert list(t["changes"]["change_id"]) == ["c1"], "the invented change must not survive"

    def test_summary_reports_the_rejection(self, batch) -> None:
        root, first, _ = batch
        s = agg.summarise(*agg.load_batch(first, root))
        assert s["runs"] == 1
        assert s["judgements_rejected"] == 1
        assert s["changes"] == 1

    def test_agreement_across_two_annotators(self, batch) -> None:
        root, first, second = batch
        a1, _ = agg.load_batch(first, root)
        a2, _ = agg.load_batch(second, root)
        out = agg.agreement(a1, a2)
        assert out["runs_compared"] == 1.0
        # One training, labelled C4 by one and both by the other: they disagree.
        assert out["tested_variable"].observed == 0.0
        # The change at i=5 was categorised the same way by both.
        assert out["category"].observed == 1.0
        # Both said the evaluation judged the change anchored at i=5. They spell
        # its id differently (c1 / x1), which must not read as disagreement.
        assert out["judges_changes_jaccard"] == pytest.approx(1.0)
