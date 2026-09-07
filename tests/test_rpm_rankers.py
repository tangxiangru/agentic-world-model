"""Pair grouping, antisymmetry, and training-only feature boundary checks."""

import copy

import numpy as np
import pytest
from scipy import sparse

from tools.outcome_prediction.rpm_rankers import (
    FeatureMap,
    canonical_document,
    evaluate_supplied_pairs,
    fit_ranker,
    group_folds,
    make_pairs,
    pair_features,
    score_pairs,
)


def row(cell, number, y, parents=None, complete=True):
    recipe = {"method": "sft", "hyperparameters": {"lr": 10 ** (-number - 2)}}
    return {
        "example_id": f"{cell}/exp-{number}",
        "cell_id": cell,
        "y": y,
        "eligible": True,
        "recipe": recipe,
        "lineage": [{"recipe": recipe}],
        "parent_ids": parents or [],
        "lineage_complete": complete,
        "numeric_features": {"hp_lr": 10 ** (-number - 2)},
    }


def test_pairs_respect_cells_exact_parents_lineage_and_gap():
    rows = [
        row("a", 1, 0.2),
        row("a", 2, 0.21),
        row("a", 3, 0.4, ["p"]),
        row("a", 4, 0.6, ["p"], False),
        row("b", 1, 0.9),
        row("b", 2, 0.7),
    ]
    all_pairs = make_pairs(rows)
    assert len(all_pairs) == 7
    siblings = make_pairs(rows, sibling_only=True)
    assert len(siblings) == 2
    assert all(p["parent_ids"] == [] for p in siblings)
    assert len(make_pairs(rows, 0.02, sibling_only=True)) == 1
    merged = [row("c", 1, 0.1, ["q", "p"]), row("c", 2, 0.8, ["p", "q"])]
    assert len(make_pairs(merged, sibling_only=True)) == 0
    merged[0]["parent_ids"] = merged[1]["parent_ids"] = ["p"]
    merged[0]["recipe"]["method"] = "merge"
    assert len(make_pairs(merged, sibling_only=True)) == 0


def test_sibling_training_uses_only_matching_complete_nonmerge_pairs():
    rows = [
        row("a", 1, 0.2),
        row("a", 2, 0.21),
        row("a", 3, 0.4, ["p"]),
        row("a", 4, 0.6, ["p"], False),
        row("b", 1, 0.9),
        row("b", 2, 0.7),
    ]
    spec = {"view": "recipe", "kind": "logistic", "C": 1}
    assert fit_ranker(rows, spec).training_pairs == 7
    assert fit_ranker(rows, spec, train_siblings_only=True).training_pairs == 2


@pytest.mark.parametrize(
    "spec",
    [
        {"view": "recipe", "kind": "logistic", "C": 1},
        {"view": "recipe", "kind": "ridge", "alpha": 1},
        {"view": "lineage", "kind": "forest", "max_depth": 3, "min_samples_leaf": 1},
        {"view": "lineage", "kind": "contextual_forest", "max_depth": 3, "min_samples_leaf": 1},
        {"view": "recipe", "kind": "pointwise_ridge", "alpha": 1},
    ],
)
def test_ranker_predictions_are_antisymmetric_and_outcome_blind(spec):
    train = [row(c, n, 0.1 + n * 0.2) for c in ("a", "b", "c") for n in (1, 2, 3)]
    fitted = fit_ranker(train, spec)
    a, b = row("heldout", 1, 0.8), row("heldout", 3, 0.2)
    p = fitted.predict_pairs([a], [b])[0]
    q = fitted.predict_pairs([b], [a])[0]
    assert p + q == pytest.approx(1)
    assert fitted.predict_pairs([a], [a])[0] == pytest.approx(0.5)
    changed = copy.deepcopy(a)
    changed.update(
        y=0.001,
        cell_id="changed",
        scientist_model="different",
        plan_text="accuracy 0.99",
        recipe_text="fake leaked text",
        local_accuracy=0.99,
        parent_accuracy=0.01,
    )
    changed["numeric_features"]["local_accuracy"] = 0.99
    assert fitted.predict_pairs([changed], [b])[0] == pytest.approx(p)


def test_training_vocabulary_and_numeric_keys_do_not_read_test_rows():
    train = [row("a", 1, 0.2), row("a", 2, 0.4)]
    fmap = FeatureMap("recipe").fit(train)
    target = row("b", 9, 0.1)
    target["recipe"]["method"] = "heldoutuniquetoken"
    target["numeric_features"]["hp_beta"] = 123
    fmap.transform([target])
    assert "heldoutuniquetoken" not in fmap.text.vocabulary_
    assert "hp_beta" not in fmap.keys
    target["lineage"][0]["card_id"] = "leaky-id"
    assert "leaky-id" not in canonical_document(target, "lineage")


def test_contextual_pair_features_preserve_absolute_level_at_identical_delta():
    a, b = sparse.csr_matrix([[2.0], [12.0]]), sparse.csr_matrix([[1.0], [11.0]])
    ordinary = pair_features(a, b).toarray()
    context = pair_features(a, b, contextual=True).toarray()
    reverse = pair_features(b, a, contextual=True).toarray()
    assert np.array_equal(ordinary[0], ordinary[1])
    assert np.array_equal(context[:, :1], ordinary)
    assert not np.array_equal(context[0], context[1])
    assert np.array_equal(reverse[:, :1], -context[:, :1])
    assert np.array_equal(reverse[:, 1:], context[:, 1:])


def test_outer_group_folds_never_split_cell():
    rows = [row(c, n, 0.1 * n) for c in "abcdefgh" for n in (1, 2)]
    for train, test in group_folds(rows):
        assert {rows[i]["cell_id"] for i in train}.isdisjoint({rows[i]["cell_id"] for i in test})


def test_macro_accuracy_regret_and_ties_are_cell_weighted():
    records = [
        {"cell_id": "a", "a_wins": 1, "gap": 0.2, "probabilities": {"m": 0.9}},
        {"cell_id": "b", "a_wins": 0, "gap": 0.4, "probabilities": {"m": 0.5}},
        {"cell_id": "b", "a_wins": 0, "gap": 0.2, "probabilities": {"m": 0.9}},
    ]
    score = score_pairs(records, "m", 100)
    assert score["macro_cell_accuracy"] == pytest.approx(0.625)
    assert score["micro_pair_accuracy"] == pytest.approx(0.5)
    assert score["macro_cell_regret"] == pytest.approx(0.1)
    assert score["ties"] == 1


def test_supplied_pairs_preserve_orientation_and_exact_frozen_banks(tmp_path):
    rows = [row(c, n, 0.1 * n) for c in "abcd" for n in (1, 2)]
    first = [r["example_id"] for r in rows if r["cell_id"] in "ab"]
    second = [r["example_id"] for r in rows if r["cell_id"] in "cd"]
    assignments = [
        {"train_ids": first, "test_ids": second},
        {"train_ids": second, "test_ids": first},
    ]
    pairs = [{"pair_id": "blind-oriented", "a_id": "a/exp-2", "b_id": "a/exp-1"}]
    methods = {"fixed": [{"view": "recipe", "kind": "logistic", "C": 1.0}]}
    predicted, folds = evaluate_supplied_pairs(
        rows, pairs, methods=methods, outer_assignments=assignments, output_dir=tmp_path
    )
    assert predicted[0]["a_id"] == "a/exp-2"
    assert predicted[0]["a_wins"] == 1
    assert predicted[0]["fold"] == 1
    assert predicted[0]["pair_id"] == "blind-oriented"
    assert folds[1]["train_ids"] == second
    assert (tmp_path / "fold-01-frozen.json").exists()
    bad = copy.deepcopy(assignments)
    bad[0]["train_ids"] += [second[0]]
    with pytest.raises(ValueError, match="partition"):
        evaluate_supplied_pairs(rows, pairs, methods=methods, outer_assignments=bad)
