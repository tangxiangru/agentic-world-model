"""The ledger: every verdict, scored at read time against the card beside it (or the replay's truth card)."""

from __future__ import annotations

import copy

from awm.exp_protocol import schema as cards
from awm.wma import ledger, schema
from exp_protocol_cards import closed_card, plan_card
from test_wma_schema import verdict


def write(dir_, card_id, skill, l3="yes", decision="adopt", wall_h=1.0, execution="completed", closed=True,
          interval=None):
    v = copy.deepcopy(verdict())
    v.update({"card_id": card_id, "wma_skill": skill})
    v["levels"]["L3_worth_now"]["answer"] = l3
    if interval is not None:
        v["levels"]["L2_effect"]["interval"] = interval
    card_path = dir_ / "memory" / "cards" / f"{card_id}.yaml"
    card = closed_card() if closed else plan_card()
    card["card_id"] = card_id
    if closed:
        card["conclusion"]["decision"] = decision
        card["conclusion"]["verdict"] = "inconclusive" if decision != "adopt" else "supported"
        card["result"].update({"execution": execution, "wall_h": wall_h})
        card["result"]["measurements"][0]["delta_vs_comparator"] = 0.01
        if execution != "completed":
            card["result"]["output_checkpoint"] = None
            card["result"]["measurements"] = []
    cards.dump_card(card_path, card)
    schema.dump_verdict(schema.verdict_path(card_path), v)


def test_rows_score_against_the_card_beside_the_verdict_and_group_by_skill(tmp_path) -> None:
    a, b = tmp_path / "s1", tmp_path / "s2"
    write(a, "exp-01", "AAAA")
    write(a, "exp-02", "AAAA", execution="failed", decision="reject", interval=[0.05, 0.06])
    write(b, "exp-01", "BBBB", l3="no", decision="reject", wall_h=2.5)          # saved 2.5 h
    write(b, "exp-02", "BBBB", l3="no", decision="adopt", wall_h=1.5)           # wrongly killed 1.5 h
    write(b, "exp-03", "BBBB", closed=False)                                    # no outcome yet
    rows = ledger.rows([tmp_path])
    assert len(rows) == 5
    summary = {s["wma_skill"]: s for s in ledger.summarize(rows)}
    assert summary["AAAA"]["n"] == 2 and summary["AAAA"]["L0_hit"] == 0.5 and summary["AAAA"]["L2_coverage"] == 1.0
    assert summary["BBBB"]["n"] == 3 and summary["BBBB"]["n_reconciled"] == 2
    assert summary["BBBB"]["gpu_h_saved"] == 2.5 and summary["BBBB"]["gpu_h_wrongly_killed"] == 1.5
    assert summary["BBBB"]["L3_hit"] == 0.5


def test_the_verdict_file_is_never_modified_by_scoring(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA")
    vp = schema.verdict_path(tmp_path / "memory" / "cards" / "exp-01.yaml")
    before = vp.read_text()
    ledger.summarize(ledger.rows([tmp_path]))
    assert vp.read_text() == before


def test_scoring_rules_apply_at_read_time(tmp_path) -> None:
    """Change the card after the verdict: the ledger sees the card as it is now, no re-reconcile step."""
    write(tmp_path, "exp-01", "AAAA", decision="adopt")
    assert ledger.rows([tmp_path])[0]["scored"]["L3"] == "hit"
    p = tmp_path / "memory" / "cards" / "exp-01.yaml"
    card = cards.load_card(p)
    card["conclusion"]["decision"] = "reject"
    cards.dump_card(p, card)
    assert ledger.rows([tmp_path])[0]["scored"]["L3"] == "miss"


def test_replay_layout_resolves_truth_outside_the_session(tmp_path) -> None:
    out = tmp_path / "replay"
    session = out / "r-aaaa" / "exp-02"
    card_path = session / "memory" / "cards" / "exp-02.yaml"
    open_card = plan_card()
    open_card["card_id"] = "exp-02"
    cards.dump_card(card_path, open_card)                      # the session's card stays open
    v = copy.deepcopy(verdict())
    v["card_id"] = "exp-02"
    schema.dump_verdict(schema.verdict_path(card_path), v)
    truth = closed_card()
    truth["card_id"] = "exp-02"
    cards.dump_card(out / "_truth" / "r-aaaa" / "exp-02.yaml", truth)
    tp, t = ledger.truth_for(schema.verdict_path(card_path))
    assert tp == out / "_truth" / "r-aaaa" / "exp-02.yaml" and t["decision"] == "adopt"
    row = ledger.rows([out])[0]
    assert row["reconciled"] and row["scored"]["L3"] == "hit"


def test_unscorable_levels_are_excluded_from_rates_and_width_is_reported(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA", execution="killed")     # ran, no valid candidate, no delta
    s = ledger.summarize(ledger.rows([tmp_path]))[0]
    assert s["L0_hit"] == 1.0 and s["L1_hit"] == 0.0 and s["L2_coverage"] == ""
    assert s["L2_width_mean"] == 0.05


def test_csv_and_render(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA")
    summary = ledger.summarize(ledger.rows([tmp_path]))
    assert ledger.to_csv(summary).splitlines()[0].startswith("wma_skill,backend,mode,n,")
    assert "AAAA" in ledger.render(summary)
