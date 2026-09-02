"""The ledger: every verdict, grouped by skill, held to its outcome."""

from __future__ import annotations

import copy

from awm.wma import ledger, schema
from test_wma_schema import verdict


def write(dir_, card_id, skill, l3="yes", scored=None, wall_h=1.0, decision="adopt", reconciled=True):
    v = copy.deepcopy(verdict())
    v.update({"card_id": card_id, "wma_skill": skill})
    v["levels"]["L3_worth_now"]["answer"] = l3
    if reconciled:
        v["reconciled_at"] = "t"
        v["actual"] = {"execution": "completed", "decision": decision, "wall_h": wall_h, "delta": 0.01}
        v["scored"] = scored or {"L0": "hit", "L1": "hit", "L2": "in_interval", "L3": "hit"}
    schema.dump_verdict(dir_ / "memory" / "cards" / f"{card_id}.verdict.json", v)


def test_rows_walk_nested_dirs_and_summary_groups_by_skill(tmp_path) -> None:
    a, b = tmp_path / "s1", tmp_path / "s2"
    write(a, "exp-01", "AAAA")
    write(a, "exp-02", "AAAA", scored={"L0": "miss", "L1": "miss", "L2": "below", "L3": "miss"})
    write(b, "exp-01", "BBBB", l3="no", decision="reject", wall_h=2.5)          # saved 2.5 h
    write(b, "exp-02", "BBBB", l3="no", decision="adopt", wall_h=1.5,
          scored={"L0": "hit", "L1": "hit", "L2": "above", "L3": "miss"})         # wrongly killed 1.5 h
    write(b, "exp-03", "BBBB", reconciled=False)
    rows = ledger.rows([tmp_path])
    assert len(rows) == 5
    summary = {s["wma_skill"]: s for s in ledger.summarize(rows)}
    assert summary["AAAA"]["n"] == 2 and summary["AAAA"]["L0_hit"] == 0.5 and summary["AAAA"]["L2_coverage"] == 0.5
    assert summary["BBBB"]["n"] == 3 and summary["BBBB"]["n_reconciled"] == 2
    assert summary["BBBB"]["gpu_h_saved"] == 2.5 and summary["BBBB"]["gpu_h_wrongly_killed"] == 1.5
    assert summary["BBBB"]["L3_hit"] == 0.5


def test_unscorable_levels_are_excluded_from_rates(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA", scored={"L0": "hit", "L1": "unscorable", "L2": "unscorable", "L3": "hit"})
    s = ledger.summarize(ledger.rows([tmp_path]))[0]
    assert s["L0_hit"] == 1.0 and s["L1_hit"] == "" and s["L2_coverage"] == ""


def test_interval_width_is_reported_so_coverage_cannot_be_bought_by_widening(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA")
    s = ledger.summarize(ledger.rows([tmp_path]))[0]
    assert s["L2_width_mean"] == 0.05


def test_csv_and_render(tmp_path) -> None:
    write(tmp_path, "exp-01", "AAAA")
    summary = ledger.summarize(ledger.rows([tmp_path]))
    assert ledger.to_csv(summary).splitlines()[0].startswith("wma_skill,backend,mode,n,")
    assert "AAAA" in ledger.render(summary)
