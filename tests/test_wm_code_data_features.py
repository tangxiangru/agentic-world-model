"""Synthetic extraction tests; no archived outcomes are used."""

import copy
import math

import pytest

from tools.outcome_prediction.wm_code_data_features import extract_data_features


def sample(command=None, code="", *, status="reconstructed", selection="", source=""):
    return {
        "plan": {
            "setup": {
                "command": {"cwd": "/task"},
                "method": {"family": "sft"},
                "data": [
                    {
                        "build_command": command or ["python", "prepare.py"],
                        "built_by": "/task/prepare.py",
                        "source": source,
                        "selection": selection,
                    }
                ],
            }
        },
        "code": [
            {
                "role": "data_builder_0",
                "script_path": "/task/prepare.py",
                "status": status,
                "content": code,
            }
        ],
    }


def test_fixed_schema_and_finite_features():
    empty, _ = extract_data_features({})
    rich, _ = extract_data_features(sample(["python", "prepare.py", "--n-gsm", "123"]))
    assert empty.keys() == rich.keys()
    assert 50 <= len(rich) <= 65
    assert all(v is None or (type(v) is float and math.isfinite(v)) for v in rich.values())


@pytest.mark.parametrize("field", ["result", "label", "progress", "official_accuracy", "score"])
def test_outcome_fields_do_not_change_features(field):
    inp = sample(["python", "prepare.py", "--n-gsm", "12000"])
    before = extract_data_features(inp)
    inp[field] = {"accuracy": 0.999, "text": "fewshot fraction 0.88; max_examples=90000"}
    inp["plan"][field] = inp[field]
    inp["plan"]["setup"][field] = inp[field]
    assert extract_data_features(inp) == before


def test_realized_examples_paths_ids_and_other_prose_excluded():
    inp = sample()
    before = extract_data_features(inp)
    d = inp["plan"]["setup"]["data"][0]
    d.update(
        n_examples=999999,
        path="/secret/exp-21-accuracy99/train.jsonl",
        other="Final accuracy .99, 15000 generated examples.",
        contamination_check="passed",
    )
    inp["example_id"] = "exp-99"
    assert extract_data_features(inp) == before


@pytest.mark.parametrize(
    "flag,field,value",
    [
        ("--n-gsm", "quota.gsm", 15000),
        ("--n_aug_gsm", "quota.aug_gsm", 20000),
        ("--n-math", "quota.math", 3000),
        ("--n-aug-math", "quota.aug_math", 1000),
        ("--max_per_q", "cap.solutions_per_problem", 4),
        ("--max-tokens", "cap.max_tokens", 2048),
        ("--fewshot-frac", "fewshot.fraction", 0.15),
        ("--n", "quota.requested_n", 40000),
        ("--shard-start", "sampling.shard_start", 3),
    ],
)
def test_exact_cli_options(flag, field, value):
    features, audit = extract_data_features(sample(["python", "prepare.py", flag, str(value)]))
    assert features[field] == value
    assert audit["field_sources"][field] == [{"entry": 0, "source": "argv"}]


def test_shell_string_argv_equals_and_unrelated_flags():
    features, _ = extract_data_features(
        sample("python prepare.py --n-gsm=15000 --max_per_q=4 --accuracy .99 --exp-id 198")
    )
    assert features["quota.gsm"] == 15000
    assert features["cap.solutions_per_problem"] == 4
    assert all("accuracy" not in key and "exp-id" not in key for key in features)


@pytest.mark.parametrize(
    "command",
    [
        "python prepare.py --n-gsm 7; python other.py",
        "python -c 'print(9)' --n-gsm 7",
        ["python", "prepare.py", "&&", "python", "other.py", "--n-gsm", "7"],
        "python prepare.py --n-gsm $(cat x)",
        "python prepare.py --n-gsm `cat x`",
        "python prepare.py --n-gsm 'unterminated",
    ],
)
def test_compound_or_dynamic_shell_is_unknown(command):
    features, _ = extract_data_features(sample(command))
    assert features["quota.gsm"] is None


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "True", "${N}", "1e99", "10000 rows"])
def test_invalid_numeric_values_are_not_guessed(value):
    features, _ = extract_data_features(sample(["python", "prepare.py", "--n-gsm", value]))
    assert features["quota.gsm"] is None


def test_invalid_override_blocks_default_and_duplicate_conflicts():
    code = "import argparse\np=argparse.ArgumentParser()\np.add_argument('--n-gsm', default=8)\na=p.parse_args()"
    features, _ = extract_data_features(sample(["python", "prepare.py", "--n-gsm", "${N}"], code))
    assert features["quota.gsm"] is None
    features, audit = extract_data_features(
        sample(["python", "prepare.py", "--n-gsm", "8", "--n-gsm", "9"], code)
    )
    assert features["quota.gsm"] is None
    assert "quota.gsm" in audit["conflicts"]


def test_only_invoked_reconstructed_builder_defaults():
    code = "import argparse\np=argparse.ArgumentParser()\np.add_argument('--n-gsm', default=8)\na=p.parse_args()"
    features, audit = extract_data_features(sample(code=code))
    assert features["quota.gsm"] == 8
    assert audit["field_sources"]["quota.gsm"][0]["source"] == "argparse_default"
    for status in ("snapshot", "unavailable", "missing", "final_snapshot"):
        features, _ = extract_data_features(sample(code=code, status=status))
        assert features["quota.gsm"] is None
    features, _ = extract_data_features(
        sample(["python", "other.py", "--input", "prepare.py"], code)
    )
    assert features["quota.gsm"] is None


def test_training_role_code_is_not_a_data_builder():
    inp = sample(code="MAXLEN=999\nprint(MAXLEN)")
    inp["code"][0]["role"] = "training"
    features, _ = extract_data_features(inp)
    assert features["code_declared.max_seq_len"] is None


def test_cli_overrides_default_without_conflict():
    code = "import argparse\np=argparse.ArgumentParser()\np.add_argument('--fewshot-frac',default=.15)\na=p.parse_args()"
    features, audit = extract_data_features(
        sample(["python", "prepare.py", "--fewshot-frac", ".4"], code)
    )
    assert features["fewshot.fraction"] == 0.4
    assert not audit["conflicts"]


@pytest.mark.parametrize(
    "code",
    [
        "if unknown:\n p.add_argument('--n-gsm', default=8)\na=p.parse_args()",
        "p.add_argument('--n-gsm',default=8)\na=p.parse_args(['--n-gsm','7'])",
        "p.add_argument('--n-gsm',default=8)\np.set_defaults(n_gsm=7)\na=p.parse_args()",
        "p.add_argument('--n-gsm',default=8)\na=p.parse_args()\nb=q.parse_args()",
        "def uncalled():\n p.add_argument('--n-gsm',default=8)\na=p.parse_args()",
        "q.add_argument('--n-gsm',default=8)\na=p.parse_args()",
        "p.add_subparsers()\np.add_argument('--n-gsm',default=8)\na=p.parse_args()",
    ],
)
def test_dynamic_alternative_or_uninvoked_defaults_are_unknown(code):
    features, _ = extract_data_features(sample(code=code))
    assert features["quota.gsm"] is None


def test_explicit_main_invocation_supports_defaults():
    code = "def main():\n p.add_argument('--n-gsm',default=8)\n a=p.parse_args()\nif __name__ == '__main__':\n main()"
    features, _ = extract_data_features(sample(code=code))
    assert features["quota.gsm"] == 8


def test_only_referenced_whitelisted_module_constants():
    code = "MAXLEN=4096\nMAX_KEEP=900\nSCORE=.99\nn_examples=9999\nMAX_CHARS=800\nprint(MAXLEN,MAX_KEEP)\n"
    features, _ = extract_data_features(sample(code=code))
    assert features["code_declared.max_seq_len"] == 4096
    assert features["code_declared.max_keep"] == 900
    assert features["code_declared.max_chars"] is None
    changed = code.replace("SCORE=.99", "SCORE=.01").replace("n_examples=9999", "n_examples=1")
    assert extract_data_features(sample(code=changed))[0] == features


def test_conditionally_defined_constants_are_not_promoted():
    features, _ = extract_data_features(sample(code="if x:\n MAXLEN=4096\nprint(MAXLEN)"))
    assert features["code_declared.max_seq_len"] is None


def test_unsupported_syntax_is_unknown_but_cli_survives():
    features, audit = extract_data_features(
        sample(["python", "prepare.py", "--n-gsm", "8"], "{broken!")
    )
    assert features["quota.gsm"] == 8
    assert audit["entries"][0]["builders"][0]["parsed"] is False


def test_mixture_quota_summaries_require_nonoverlapping_named_buckets():
    features, _ = extract_data_features(
        sample(
            [
                "python",
                "prepare.py",
                "--n-gsm",
                "10",
                "--n-aug-gsm",
                "30",
                "--n-math",
                "20",
                "--n-aug-math",
                "40",
                "--n-omi",
                "99999",
            ]
        )
    )
    assert features["mixture.quota_sum"] == 100
    assert features["mixture.math_fraction"] == 0.6
    assert features["mixture.augmented_fraction"] == 0.7
    assert features["mixture.quota_source_count"] == 4


def test_partial_quota_does_not_assume_missing_sources_zero():
    features, _ = extract_data_features(sample(["python", "prepare.py", "--n-gsm", "10"]))
    assert features["mixture.quota_sum"] == 10
    assert features["mixture.math_fraction"] is None


def test_fresh_replay_quota_fraction_is_not_derived_from_ancestor_mentions():
    features, _ = extract_data_features(sample(source="derived:exp-01 + fresh examples"))
    assert features["mixture.fresh_fraction"] is None
    features, _ = extract_data_features(
        sample(
            [
                "python",
                "prepare.py",
                "--n-fresh",
                "30",
                "--n-replay",
                "10",
            ]
        )
    )
    assert features["mixture.fresh_fraction"] == 0.75


def test_declared_mixture_weights_and_incompatible_entries():
    inp = sample(["python", "prepare.py", "--n-gsm", "10"])
    d = inp["plan"]["setup"]["data"][0]
    d["mixture_weight"] = 1
    d2 = copy.deepcopy(d)
    d2["mixture_weight"] = 3
    d2["build_command"] = ["python", "prepare2.py", "--n-gsm", "20"]
    inp["plan"]["setup"]["data"].append(d2)
    features, audit = extract_data_features(inp)
    assert features["quota.gsm"] is None
    assert features["mixture.max_weight"] == 0.75
    assert features["mixture.weight_count"] == 2
    assert audit["conflicts"] == ["quota.gsm"]


@pytest.mark.parametrize(
    "selection,field,expected",
    [
        ("15% of rows get a k-shot prefix from train questions.", "fewshot.fraction", 0.15),
        ("with p=0.30 a k-shot prefix is prepended.", "fewshot.fraction", 0.30),
        ("at most 4 solutions per distinct problem", "cap.solutions_per_problem", 4),
        ("at most one solution per problem", "cap.solutions_per_problem", 1),
        ("accuracy 15%; 90% of answers correct; 12000 generated rows", "fewshot.fraction", None),
        ("15% final accuracy in few-shot evaluation", "fewshot.fraction", None),
        ("150% of rows get a k-shot prefix", "fewshot.fraction", None),
    ],
)
def test_bounded_numeric_policy_expressions(selection, field, expected):
    features, _ = extract_data_features(sample(selection=selection))
    assert features[field] == expected


def test_policy_markers_and_negation():
    features, _ = extract_data_features(
        sample(
            selection="Filtered to correct final answer; deduped. Numeric expected_answer only. "
            "Require a boxed answer. Append ANSWER: N. Removed calculator annotations.",
            source="HF OpenMathInstruct-2 GSM8K + derived:exp-987",
        )
    )
    for field in (
        "correctness_filter",
        "deduplicate",
        "numeric_answer",
        "boxed_required",
        "answer_line",
        "strip_calculator",
    ):
        assert features["policy.evidence." + field] == 1
    assert features["source.evidence.derived"] == 1
    features, _ = extract_data_features(
        sample(selection="No deduplication is done.", source="No self-generated data")
    )
    assert features["policy.evidence.deduplicate"] == 0
    assert features["source.evidence.synthetic"] == 0


def test_unrelated_code_strings_do_not_supply_policy_or_numerics():
    code = 'NOTE="filtered to correct; 15% of rows get a k-shot prefix; MAXLEN=99000"\nscore=0.99'
    features, _ = extract_data_features(sample(code=code))
    assert features["fewshot.fraction"] is None
    assert features["code_declared.max_seq_len"] is None
    assert features["policy.evidence.correctness_filter"] == 0


@pytest.mark.parametrize(
    "command,cwd,path,expected",
    [
        (["python", "other/prepare.py"], "/task", "/task/prepare.py", None),
        (["python", "prepare.py"], None, "/task/prepare.py", None),
        (["python", "/task/prepare.py"], None, "/task/prepare.py", 8),
        (["python", "./scripts/../prepare.py"], "/task", "/task/prepare.py", 8),
        (["python", "prepare.py"], "/task/other", "/task/prepare.py", None),
    ],
)
def test_builder_routing_uses_exact_normalized_paths(command, cwd, path, expected):
    code = "p.add_argument('--n-gsm',default=8)\na=p.parse_args()"
    inp = sample(command, code)
    inp["plan"]["setup"]["command"]["cwd"] = cwd
    inp["code"][0]["script_path"] = path
    features, _ = extract_data_features(inp)
    assert features["quota.gsm"] == expected


@pytest.mark.parametrize(
    "family",
    [
        "merge",
        "model_merge",
        "model_soup",
        "decode-config",
        "decoding",
        "eval",
        "eval_only",
        "checkpoint_selection",
        "weight_averaging",
    ],
)
def test_nontraining_interventions_do_not_inherit_ancestor_data_dose(family):
    inp = sample(
        ["python", "prepare.py", "--n-gsm", "15000"],
        "MAXLEN=8192\nprint(MAXLEN)",
        source="HF GSM8K + derived:exp-01",
        selection="15% of rows get a k-shot prefix. Filtered to correct final answer.",
    )
    inp["plan"]["setup"]["method"]["family"] = family
    features, audit = extract_data_features(inp)
    assert features["quota.gsm"] is None
    assert features["code_declared.max_seq_len"] is None
    assert features["fewshot.fraction"] is None
    assert features["policy.evidence.correctness_filter"] == 0
    assert features["source.evidence.gsm8k"] == 1
    assert features["source.evidence.derived"] == 1
    assert audit["data_intervention_active"] is False


@pytest.mark.parametrize(
    "selection",
    [
        "Not 15% of rows get a k-shot prefix.",
        "Never with p=0.30 a k-shot prefix is prepended.",
        "Not at most 4 solutions per distinct problem.",
    ],
)
def test_numeric_policy_expressions_honor_local_negation(selection):
    features, _ = extract_data_features(sample(selection=selection))
    assert features["fewshot.fraction"] is None
    assert features["cap.solutions_per_problem"] is None


@pytest.mark.parametrize(
    "code",
    [
        "p.add_argument('--n-gsm',default=8)\ndef never_called():\n a=p.parse_args()",
        "p.add_argument('--n-gsm',default=8)\nif unknown:\n a=p.parse_args()",
        "def main():\n p.add_argument('--n-gsm',default=8)\n a=p.parse_args()\nif __name__ == '__main__':\n if unknown:\n  main()",
        "[p.add_argument('--n-gsm',default=8) for x in unknown]\na=p.parse_args()",
    ],
)
def test_parse_args_must_be_unconditionally_reachable(code):
    features, _ = extract_data_features(sample(code=code))
    assert features["quota.gsm"] is None
