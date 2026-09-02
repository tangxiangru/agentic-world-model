"""The ledger: every verdict, scored at read time against the card beside it (or the replay's truth card)."""

from __future__ import annotations

import copy

from awm.exp_protocol import schema as cards
from awm.wma import ledger, schema
from exp_protocol_cards import closed_card, plan_card
from test_wma_schema import verdict


def write(dir_, card_id, skill, l3="yes", decision="adopt", wall_h=1.0, execution="completed", closed=True,
          interval=None, l0="yes", l1="yes", model=None, effort=None, tag=None, valid=True, change_types=None,
          family=None):
    v = copy.deepcopy(verdict())
    v.update({"card_id": card_id, "wma_skill": skill})
    if model:
        v["model"] = model
    if effort:
        v["effort"] = effort
    if change_types is not None:
        v["change_types"] = change_types
    v["levels"]["L0_runs"]["answer"] = l0
    v["levels"]["L1_valid"]["answer"] = l1
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
        if execution != "completed" or not valid:
            card["result"]["output_checkpoint"] = None
            card["result"]["measurements"] = []
    if family:
        card["setup"]["method"]["family"] = family
    cards.dump_card(card_path, card)
    schema.dump_verdict(schema.verdict_path(card_path, tag=tag), v)


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
    assert summary["BBBB"]["n"] == 3 and summary["BBBB"]["n_scored"] == 2
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
    assert row["has_truth"] and row["scored"]["L3"] == "hit"


def test_unscorable_levels_are_excluded_from_rates_and_width_is_reported(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA", execution="killed")     # ran; L1 not scored on a killed run; no delta
    s = ledger.summarize(ledger.rows([tmp_path]))[0]
    assert s["L0_hit"] == 1.0 and s["L1_hit"] == "" and s["L2_coverage"] == ""
    assert s["L2_width_mean"] == 0.05


def test_csv_and_render(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA")
    summary = ledger.summarize(ledger.rows([tmp_path]))
    assert ledger.to_csv(summary).splitlines()[0].startswith("wma_skill,backend,model,effort,mode,slice,n,n_scored,")
    assert "AAAA" in ledger.render(summary)


def test_tagged_verdicts_resolve_to_the_same_card(tmp_path) -> None:
    card_path = tmp_path / "memory" / "cards" / "exp-01.yaml"
    cards.dump_card(card_path, closed_card())
    for tag in ("opus", "codex"):
        v = copy.deepcopy(verdict())
        v["backend"] = tag
        schema.dump_verdict(schema.verdict_path(card_path, tag=tag), v)
    rows = ledger.rows([tmp_path])
    assert len(rows) == 2 and all(r["has_truth"] and r["scored"]["L0"] == "hit" for r in rows)
    assert {r["backend"] for r in rows} == {"opus", "codex"}


def test_measured_cost_is_summed_and_suspected_leaks_are_kept_out_of_the_rates(tmp_path) -> None:
    for cid, usd, leak in (("exp-01", 0.5, False), ("exp-02", 0.7, False), ("exp-03", 0.9, True)):
        write(tmp_path, cid, "AAAA", execution="failed", decision="reject")   # every verdict says yes → L0 miss
        vp = schema.verdict_path(tmp_path / "memory" / "cards" / f"{cid}.yaml")
        v = schema.load_verdict(vp)
        v["cost"]["usd"] = usd
        if leak:
            v["leak_suspected"] = True
            v["access"] = {"files": 4, "outside": ["/somewhere/_truth/x.yaml"]}
        schema.dump_verdict(vp, v)
    s = ledger.summarize(ledger.rows([tmp_path]))[0]
    assert s["n"] == 3 and s["n_leak_suspected"] == 1 and s["n_scored"] == 2
    assert s["cost_usd_sum"] == 2.1 and s["cost_usd_mean"] == 0.7
    assert s["L0_hit"] == 0.0          # computed over the two clean rows only


# ---- model and effort are part of the group key; recall on the failed cards is reported (2026-09-02) ----

def test_the_same_skill_on_two_models_or_efforts_is_two_ledger_rows(tmp_path) -> None:
    write(tmp_path / "a", "exp-01", "AAAA", model="m-1", effort="high")
    write(tmp_path / "a", "exp-01", "AAAA", model="m-2", effort="high", tag="m2")
    write(tmp_path / "a", "exp-01", "AAAA", model="m-1", effort="xhigh", tag="xh")
    summary = ledger.summarize(ledger.rows([tmp_path]))
    assert [(s["model"], s["effort"]) for s in summary] == [("m-1", "high"), ("m-1", "xhigh"), ("m-2", "high")]
    assert all(s["n"] == 1 for s in summary)
    for col in ("model", "effort", "n_scored", "L0_recall_failed", "L1_recall_invalid", "n_L2_scorable"):
        assert col in ledger.SUMMARY_COLUMNS


def test_recall_is_measured_on_the_cards_that_failed_or_yielded_no_candidate(tmp_path) -> None:
    d = tmp_path / "s"
    write(d, "exp-01", "AAAA", execution="failed", decision="reject", l0="no", l1="no")   # L0 caught; L1 not scored
    write(d, "exp-02", "AAAA", execution="failed", decision="reject", l0="yes", l1="yes")  # L0 missed
    write(d, "exp-03", "AAAA", execution="killed", decision="reject", l0="yes", l1="no")   # ran; L1 not scored
    write(d, "exp-04", "AAAA", l0="yes", l1="yes")                                          # completed, valid
    write(d, "exp-05", "AAAA", l0="no", l1="no")                                            # completed, valid: false alarms
    write(d, "exp-06", "AAAA", decision="reject", l1="no", valid=False)                     # completed, no candidate: caught
    write(d, "exp-07", "AAAA", decision="reject", l1="yes", valid=False)                    # completed, no candidate: missed
    s = ledger.summarize(ledger.rows([d]))[0]
    assert s["L0_recall_failed"] == 0.5                      # exp-01 caught, exp-02 missed; exp-03 ran
    assert s["L1_recall_invalid"] == 0.5                     # exp-06 caught, exp-07 missed; failed/killed not scored
    assert s["L0_hit"] == round(5 / 7, 3) and s["L1_hit"] == 0.5   # L1 over the four completed cards
    assert s["n_scored"] == 7 and s["n_L2_scorable"] == 2    # only the two completed, valid cards carry a delta


def test_rejected_files_are_not_verdicts_but_their_spend_is_counted(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA")
    card_path = tmp_path / "memory" / "cards" / "exp-02.yaml"
    cards.dump_card(card_path, plan_card() | {"card_id": "exp-02"})
    bad = schema.verdict_path(card_path)
    schema.dump_verdict(bad, {**schema.empty_verdict("exp-02"), "levels": {}})
    schema.reject_verdict(bad, "levels: required", cost={"usd": 0.7, "turns": 3}, wall_min=2.0)
    schema.reject_verdict(schema.dump_verdict(bad, {"x": 1}) or bad, "invalid verdict JSON", cost={"usd": 0.2})
    assert len(ledger.rows([tmp_path])) == 1
    r = ledger.rejected([tmp_path])
    assert r == {"n": 2, "cost_usd_sum": 0.9}


def test_width_is_reported_against_the_noise_floor_of_the_evaluation(tmp_path) -> None:
    """closed_card measures at n=150 → floor 0.03; an interval of width 0.06 is twice the floor, not 'wide'."""
    write(tmp_path, "exp-01", "AAAA", interval=[0.0, 0.06])
    s = ledger.summarize(ledger.rows([tmp_path]))[0]
    assert s["L2_width_mean"] == 0.06 and s["L2_width_over_noise"] == 2.0
    assert "L2_width_over_noise" in ledger.SUMMARY_COLUMNS


def test_the_ledger_slices_by_change_type_and_by_family(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA", change_types=["C2", "C3"], family="sft")
    write(tmp_path, "exp-02", "AAAA", change_types=["C1b"], family="decode-config", l3="no", decision="reject")
    write(tmp_path, "exp-03", "AAAA", family="sft", l3="no", decision="adopt")          # no change_types on this one
    rows = ledger.rows([tmp_path])
    by_type = {s["slice"]: s for s in ledger.summarize(rows, by="type")}
    assert set(by_type) == {"C2", "C3", "C1b", "(untyped)"}
    assert by_type["C2"]["n"] == 1 and by_type["C3"]["n"] == 1 and by_type["C1b"]["L3_hit"] == 1.0
    assert by_type["(untyped)"]["n"] == 1 and by_type["(untyped)"]["L3_hit"] == 0.0
    by_family = {s["slice"]: s for s in ledger.summarize(rows, by="family")}
    assert by_family["sft"]["n"] == 2 and by_family["decode-config"]["n"] == 1
    plain = ledger.summarize(rows)
    assert len(plain) == 1 and plain[0]["slice"] == "" and plain[0]["n"] == 3
