"""The meta loop compares protocol variants on numbers the cards themselves carry."""

from __future__ import annotations

import json

from awm.exp_protocol import collect, lock, schema
from exp_protocol_cards import closed_card, plan_card


def test_collect_reads_cards_locks_preflights_and_metrics(tmp_path) -> None:
    s = tmp_path / "s1"
    cards = s / "memory" / "cards"
    c1 = closed_card()
    c1["situation"]["pitfalls_hit"] = [{"what": "OOM at bs 8", "cost_h": 0.5, "fix": "bs 4"},
                                       {"what": "eos", "cost_h": 1.0, "fix": "template"}]
    schema.dump_card(cards / "exp-01.yaml", c1)
    lock.write_lock(cards / "exp-01.yaml", c1, {"pass": 5, "warn": 1, "fail": 0})
    c2 = plan_card()
    c2["card_id"] = "exp-02"
    schema.dump_card(cards / "exp-02.yaml", c2)
    lock.write_lock(cards / "exp-02.yaml", c2, {"pass": 3, "warn": 0, "fail": 2})
    (s / "metrics.json").write_text(json.dumps({"accuracy": 0.4123, "stderr": 0.01}))

    rows = collect.collect([s])
    assert len(rows) == 1
    r = rows[0]
    assert r["session"] == "s1"
    assert r["accuracy"] == 0.4123
    assert r["n_cards"] == 2 and r["n_closed"] == 1 and r["n_locked"] == 2 and r["n_locked_open"] == 1
    assert r["preflight_fail"] == 2
    assert r["pitfalls_hit"] == 2 and r["pitfalls_cost_h"] == 1.5
    assert r["adopted"] == 1
    assert r["fields_filled"] == 1.0


def test_metrics_json_in_the_parent_dir_is_found(tmp_path) -> None:
    s = tmp_path / "result" / "task"
    schema.dump_card(s / "memory" / "cards" / "exp-01.yaml", plan_card())
    (tmp_path / "result" / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))
    assert collect.collect([s])[0]["accuracy"] == 0.5


def test_a_session_without_cards_still_yields_a_row(tmp_path) -> None:
    s = tmp_path / "empty"
    s.mkdir()
    r = collect.collect([s])[0]
    assert r["n_cards"] == 0 and r["accuracy"] == "" and r["fields_filled"] == ""


def test_csv_has_the_columns_in_order(tmp_path) -> None:
    s = tmp_path / "s"
    s.mkdir()
    text = collect.to_csv(collect.collect([s]))
    assert text.splitlines()[0] == ",".join(collect.COLUMNS)
