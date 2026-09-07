"""Rich plan leakage boundary, data lineage, and known-history checks."""

import copy
from pathlib import Path

from tools.outcome_prediction import rpm_rich_context as rich


def fixtures(monkeypatch, plans, parents=None):
    rows, records = [], {}
    for i, plan in enumerate(plans, 1):
        cid = f"exp-{i:02d}"
        source = f"cells/test/wm/cards/{cid}/card.json"
        row = {
            "card_id": cid,
            "cell_id": "test",
            "example_id": f"test/{cid}",
            "source_card": source,
            "parent_ids": (parents or {}).get(cid, []),
            "data_dependency_ids": [],
            "recipe": {"method": "sft", "base_model": "base"},
            "y": i / 10,
        }
        card = {
            "setup": {"output_dir": f"runs/sft{i}", "method": {"hyperparams": {"lr": 0.00002}}},
            **copy.deepcopy(plan),
        }
        records[source] = ({"at": f"2026-01-01T00:{i:02d}:00Z", "card": card, "source": source},)
        rows.append(row)
    monkeypatch.setattr(rich, "read_records", lambda root, source: records[source])
    return rows, records


def test_checkpoint_alias_data_source_and_model_arg_reject_siblings(monkeypatch):
    for data in (
        {"source": "synthetic:self (sft1 sampling on GSM8K-train)"},
        {
            "source": "synthetic:self",
            "build_command": ["python", "generate.py", "--model", "runs/sft1"],
        },
    ):
        rows, _ = fixtures(monkeypatch, [{}, {"setup": {"data": [data]}}])
        payload, audit = rich.build_pair(rows[0], rows[1], rows, Path("raw"))
        assert payload is None
        assert "depends on the other candidate" in audit["reasons"][0]


def test_whole_section_redaction_preserves_numeric_hyperparameters(monkeypatch):
    rows, _ = fixtures(
        monkeypatch,
        [
            {},
            {
                "problem": {
                    "statement": "exp-01 collapsed",
                    "evidence": [{"observation": "0.48 again"}],
                },
                "hypothesis": {"claim": "Train on external data"},
                "result": {"measurements": [{"metric": "accuracy", "value": 0.99}]},
                "conclusion": {"summary": "candidate is best"},
            },
        ],
    )
    payload, audit = rich.build_pair(rows[0], rows[1], rows, Path("raw"))
    plan = payload["candidate_B"]["earliest_plan"]
    assert audit["accepted"] and not audit["strict_unredacted"]
    assert "problem" not in plan
    assert "result" not in plan and "conclusion" not in plan
    assert plan["setup"]["method"]["hyperparams"]["lr"] == 0.00002
    assert plan["hypothesis"]["claim"] == "Train on external data"


def test_early_baselines_and_safe_section_attribution_are_retained(monkeypatch):
    rows, _ = fixtures(
        monkeypatch,
        [
            {"result": {"measurements": [{"metric": "accuracy", "value": 0.42}]}},
            {
                "problem": {
                    "statement": "Previous model failed",
                    "evidence": [{"observation": "accuracy 0.42"}],
                }
            },
            {
                "problem": {
                    "statement": "exp-01 is the baseline",
                    "evidence": [{"observation": "accuracy 0.42"}],
                }
            },
        ],
    )
    payload, audit = rich.build_pair(rows[1], rows[2], rows, Path("raw"))
    assert audit["strict_unredacted"]
    assert "problem" in payload["candidate_A"]["earliest_plan"]
    assert "problem" in payload["candidate_B"]["earliest_plan"]
    assert payload["scored_predecessors"][0]["card_ref"] == "exp-01"
    assert payload["scored_predecessors"][0]["observed_local_accuracy"] == 0.42


def test_candidate_and_future_outcomes_never_enter_payload(monkeypatch):
    rows, _ = fixtures(monkeypatch, [{}, {}, {}, {}])
    payload, audit = rich.build_pair(rows[1], rows[2], rows, Path("raw"))
    assert audit["accepted"]
    rows[1]["y"] = rows[2]["y"] = rows[3]["y"] = 0.999
    changed, _ = rich.build_pair(rows[1], rows[2], rows, Path("raw"))
    assert changed == payload
    assert payload["scored_predecessors"] == []


def test_compact_card_refs_expand_and_generic_result_dirs_do_not_alias(monkeypatch):
    assert rich.card_references("derived:exp-01/02") == {"exp-01", "exp-02"}
    rows, _ = fixtures(
        monkeypatch,
        [
            {"problem": {"evidence": [{"path": "/work/results/base.json"}]}},
            {"setup": {"output_dir": "/work/results"}},
        ],
    )
    payload, audit = rich.build_pair(rows[0], rows[1], rows, Path("raw"))
    assert audit["strict_unredacted"]
    assert "problem" in payload["candidate_A"]["earliest_plan"]
