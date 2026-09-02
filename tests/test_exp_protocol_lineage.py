"""Follow parent_checkpoint.origin back and you have the recipe; look at what still exists and you have a start."""

from __future__ import annotations

import pytest

from awm.exp_protocol import lineage, lock, schema
from exp_protocol_cards import closed_card, plan_card


def card(card_id: str, origin: str, decision: str | None = None, ckpt: str | None = None,
         value: float | None = None):
    c = closed_card() if decision else plan_card()
    c["card_id"] = card_id
    c["setup"]["parent_checkpoint"]["origin"] = origin
    if decision:
        c["conclusion"]["decision"] = decision
        c["conclusion"]["verdict"] = "supported" if value else "inconclusive"
        c["result"]["output_checkpoint"] = ckpt
        c["result"]["measurements"] = ([{"metric": "accuracy", "value": value, "n": 150, "path": "/t/e.json"}]
                                       if value is not None else [])
    return c


@pytest.fixture
def session(tmp_path):
    cards = tmp_path / "memory" / "cards"
    ck = tmp_path / "ckpts"
    (ck / "exp-01").mkdir(parents=True)
    schema.dump_card(cards / "exp-01.yaml", card("exp-01", "base_model", "adopt", str(ck / "exp-01"), 0.41))
    schema.dump_card(cards / "exp-02.yaml", card("exp-02", "exp-01", "reject", str(ck / "exp-02"), 0.39))
    schema.dump_card(cards / "exp-03.yaml", card("exp-03", "exp-01", "adopt", str(ck / "exp-03-gone"), 0.44))
    schema.dump_card(cards / "exp-04.yaml", card("exp-04", "exp-03"))       # open, not locked
    lock.write_lock(cards / "exp-01.yaml", schema.load_card(cards / "exp-01.yaml"), {})
    return tmp_path


def test_chain_walks_back_to_base_model(session) -> None:
    cards = lineage.load_cards(lineage.cards_dir(session))
    assert lineage.chain(cards, "exp-04") == ["exp-04", "exp-03", "exp-01", "base_model"]


def test_chain_stops_on_a_cycle_instead_of_looping(session) -> None:
    cards = lineage.load_cards(lineage.cards_dir(session))
    cards["exp-01"]["setup"]["parent_checkpoint"]["origin"] = "exp-04"
    out = lineage.chain(cards, "exp-04")
    assert out[0] == "exp-04" and out[-1].startswith("cycle:")


def test_index_rows_carry_status_decision_and_best_measurement(session) -> None:
    rows = lineage.index_rows(lineage.load_cards(lineage.cards_dir(session)), lineage.cards_dir(session))
    by_id = {r["card_id"]: r for r in rows}
    assert by_id["exp-01"]["status"] == "closed" and by_id["exp-01"]["locked"] is True
    assert by_id["exp-04"]["status"] == "open" and by_id["exp-04"]["locked"] is False
    assert by_id["exp-03"]["decision"] == "adopt" and by_id["exp-03"]["best"] == "accuracy=0.44"
    assert by_id["exp-02"]["parent"] == "exp-01"


def test_rendered_index_is_one_line_per_card_in_id_order(session) -> None:
    text = lineage.write_index(session).read_text()
    lines = [ln for ln in text.splitlines() if ln.startswith("| exp-")]
    assert [ln.split("|")[1].strip() for ln in lines] == ["exp-01", "exp-02", "exp-03", "exp-04"]


def test_starting_points_distinguish_checkpoint_level_from_recipe_level(session) -> None:
    cards = lineage.load_cards(lineage.cards_dir(session))
    points = {p["card_id"]: p for p in lineage.starting_points(cards)}
    assert set(points) == {"exp-01", "exp-03"}            # adopted only
    assert points["exp-01"]["level"] == "checkpoint"      # dir exists
    assert points["exp-03"]["level"] == "recipe"          # dir gone: rerun the chain


# ---- review findings (2026-09-01) --------------------------------------------

def test_a_missing_origin_is_unknown_not_base_model(session) -> None:
    """I7: a v1 corpus card without origin must not be given a base_model parent it never claimed."""
    cards = lineage.load_cards(lineage.cards_dir(session))
    del cards["exp-04"]["setup"]["parent_checkpoint"]["origin"]
    assert lineage.chain(cards, "exp-04") == ["exp-04", "unknown"]


def test_a_non_string_origin_does_not_crash(session) -> None:
    cards = lineage.load_cards(lineage.cards_dir(session))
    cards["exp-04"]["setup"]["parent_checkpoint"]["origin"] = {"path": "x"}
    out = lineage.chain(cards, "exp-04")
    assert out[0] == "exp-04" and out[-1].startswith("invalid:")


def test_an_unreadable_card_is_skipped_and_reported(session) -> None:
    """I8: one truncated file must not take index/close/collect down with it."""
    bad = lineage.cards_dir(session) / "exp-05.yaml"
    bad.write_text("just a string\n")
    problems: list = []
    cards = lineage.load_cards(lineage.cards_dir(session), problems=problems)
    assert set(cards) == {"exp-01", "exp-02", "exp-03", "exp-04"}
    assert len(problems) == 1 and problems[0][0] == bad
    text = lineage.write_index(session).read_text()
    assert "exp-05.yaml" in text and "unreadable" in text.lower()


def test_index_rows_carry_relock_counts(session) -> None:
    cards_directory = lineage.cards_dir(session)
    card = schema.load_card(cards_directory / "exp-01.yaml")
    card["hypothesis"]["claim"] = "changed"
    lock.write_lock(cards_directory / "exp-01.yaml", card, {}, relock_reason="typo")
    rows = {r["card_id"]: r for r in lineage.index_rows(lineage.load_cards(cards_directory), cards_directory)}
    assert rows["exp-01"]["relocks"] == 1 and rows["exp-02"]["relocks"] == 0
