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


# ---- review findings (2026-09-01) --------------------------------------------

def test_a_posttrainbench_task_dir_is_labelled_by_its_cell(tmp_path) -> None:
    """I3: every PTB session dir is named `task`; the row must still say which cell it is."""
    s = tmp_path / "gsm8k_gemma_17138223" / "task"
    schema.dump_card(s / "memory" / "cards" / "exp-01.yaml", plan_card())
    assert collect.collect([s])[0]["session"] == "gsm8k_gemma_17138223/task"


def test_relocks_overrides_and_unreadable_cards_are_counted(tmp_path) -> None:
    s = tmp_path / "s"
    cards = s / "memory" / "cards"
    c = plan_card()
    schema.dump_card(cards / "exp-01.yaml", c)
    lock.write_lock(cards / "exp-01.yaml", c, {"fail": 1}, overrides={"max_seq_len_headroom": "code rows"})
    c["hypothesis"]["claim"] = "changed"
    lock.write_lock(cards / "exp-01.yaml", c, {}, relock_reason="typo")
    (cards / "exp-02.yaml").write_text("- not a card\n")
    r = collect.collect([s])[0]
    assert r["n_cards"] == 1 and r["n_unreadable"] == 1
    assert r["n_relocked"] == 1 and r["n_overrides"] == 1
