"""Independent accuracy/regret accounting does not call the production scorer."""

import pytest

from tools.outcome_prediction.rpm_faithful_results_verify import independent_summary


def row(cell, probability, choice=None):
    result = {
        "cell_id": cell,
        "y_a": 0.8,
        "y_b": 0.4,
        "probabilities": {"method": probability},
    }
    if choice is not None:
        result["choices"] = {"method": choice}
    return result


def test_macro_weights_cells_and_micro_weights_pairs():
    result = independent_summary(
        [row("one", 0.2), row("one", 0.2), row("two", 0.8)], "method"
    )
    assert result["pairs"] == 3
    assert result["cells"] == 2
    assert result["micro_accuracy"] == pytest.approx(1 / 3)
    assert result["macro_accuracy"] == pytest.approx(0.5)
    assert result["micro_regret"] == pytest.approx(0.8 / 3)
    assert result["macro_regret"] == pytest.approx(0.2)


def test_exact_chance_has_expected_half_credit_and_regret():
    result = independent_summary([row("one", 0.5)], "method")
    assert result["micro_accuracy"] == 0.5
    assert result["micro_regret"] == pytest.approx(0.2)


def test_forced_choice_overrides_tied_probability():
    result = independent_summary([row("one", 0.5, False)], "method")
    assert result["micro_accuracy"] == 0
    assert result["micro_regret"] == pytest.approx(0.4)


def test_regret_recomputes_gap_from_labels():
    record = row("one", 0.2)
    record["gap"] = 0.99
    result = independent_summary([record], "method")
    assert result["micro_regret"] == pytest.approx(0.4)
