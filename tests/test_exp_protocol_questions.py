"""Nothing is guessed: a field the card does not settle becomes a question."""

from __future__ import annotations

from awm.exp_protocol import questions, schema
from exp_protocol_cards import plan_card


def test_a_complete_card_has_no_questions() -> None:
    assert questions.missing_fields(plan_card()) == []


def test_a_fresh_card_asks_every_required_question_in_order() -> None:
    asked = [f for f, _ in questions.missing_fields(schema.minimal_card("exp-01"))]
    assert asked == list(questions.REQUIRED)


def test_the_question_text_is_the_one_from_the_table() -> None:
    card = plan_card()
    card["setup"]["output_dir"] = None
    assert questions.missing_fields(card) == [("setup.output_dir", questions.REQUIRED["setup.output_dir"])]


def test_an_empty_data_list_asks_for_data() -> None:
    card = plan_card()
    card["setup"]["data"] = []
    assert [f for f, _ in questions.missing_fields(card)] == ["setup.data"]


def test_a_training_family_adds_the_stop_token_and_max_seq_len_questions() -> None:
    card = schema.minimal_card("exp-01")
    card["setup"]["method"]["family"] = "sft"
    asked = [f for f, _ in questions.missing_fields(card)]
    assert asked[-2:] == ["setup.method.stop_token", "setup.method.hyperparams.max_seq_len"]
    card["setup"]["method"]["family"] = "merge"
    asked = [f for f, _ in questions.missing_fields(card)]
    assert "setup.method.stop_token" not in asked


def test_a_card_that_trains_nothing_is_not_asked_for_data() -> None:
    card = plan_card()
    card["setup"]["method"]["family"] = "decode-config"
    card["setup"]["data"] = []
    assert "setup.data" not in [f for f, _ in questions.missing_fields(card)]


def test_a_card_without_a_family_yet_is_still_asked_for_data() -> None:
    card = plan_card()
    card["setup"]["method"]["family"] = None
    card["setup"]["data"] = []
    assert "setup.data" in [f for f, _ in questions.missing_fields(card)]
