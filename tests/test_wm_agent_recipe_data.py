import copy
import json
import math

import pytest

from tools.outcome_prediction import wm_agent_recipe_data as data


def example(source=None, code=None, *, role="data_builder_0", status="reconstructed"):
    return {
        "model_input": {"plan": {"setup": {"data": [] if source is None else [{"source": source}]}},
                        "code": [] if code is None else [{"role": role, "status": status, "content": code,
                                                           "script_path": "/PRIVATE/CANARY/code.py"}]},
        "history": [], "history_complete_to_base": True,
    }


def families(result, scope="current"):
    return {family for family in data.FAMILIES if result["features"][f"recipe_data.{scope}.family.{family}"]}


def test_schema_is_exact57_finite_scalars():
    result = data.extract(example())
    assert len(data.FEATURE_KEYS) == len(set(data.FEATURE_KEYS)) == 57
    assert tuple(result["features"]) == data.FEATURE_KEYS
    assert all(type(value) is float and math.isfinite(value) for value in result["features"].values())
    assert result["features"]["recipe_data.current.identity_unknown"] == 1.0
    assert result["features"]["recipe_data.current.identity_known"] == 0.0


@pytest.mark.parametrize("family,repo", data.DATASETS.items())
def test_declared_whitelist_identities(family, repo):
    result = data.extract(example("HF: " + repo + " (train split)"))
    assert family in families(result)
    assert result["context"]["current"]["quantities_and_mixture_ratios"] == "not_extracted"


def test_family_alias_does_not_claim_resolved_dataset_version():
    result = data.extract(example("s1K"))
    context = result["context"]["current"]
    assert context["families"] == ["s1k"]
    assert context["canonical_family_representatives"] == ["simplescaling/s1K-1.1"]
    assert context["dataset_versions_and_revisions"] == "not_resolved_family_representatives_only"
    assert "canonical_dataset_names" not in context


def test_declared_mixture_and_unknown_ratios():
    result = data.extract(example("mix: open-r1/OpenR1-Math-220k + simplescaling/s1K-1.1"))
    assert families(result) == {"openr1_math220k", "s1k"}
    assert result["features"]["recipe_data.current.multiple_families"] == 1
    assert result["features"]["recipe_data.current.mixture_declared"] == 1
    assert result["context"]["current"]["quantities_and_mixture_ratios"] == "not_extracted"


def test_multiple_upstream_references_do_not_claim_mixture():
    result = data.extract(example("open-r1/Mixture-of-Thoughts (derived from open-r1/OpenR1-Math-220k)"))
    assert families(result) == {"mixture_of_thoughts", "openr1_math220k"}
    assert result["features"]["recipe_data.current.multiple_families"] == 1
    assert result["features"]["recipe_data.current.mixture_declared"] == 0


def test_synthetic_derived_unknown_identity_not_no_data():
    result = data.extract(example("synthetic:self (exp-99 samples) + derived:exp-98 replay"))
    assert not families(result)
    assert result["features"]["recipe_data.current.identity_unknown"] == 1
    assert result["features"]["recipe_data.current.synthetic_declared"] == 1
    assert result["features"]["recipe_data.current.derived_declared"] == 1
    assert "exp-99" not in json.dumps(result["context"])


def test_unknown_dataset_no_guess_from_benchmark():
    value = example("private-org/unknown-math-corpus")
    value.update(benchmark="gsm8k", parent={"accuracy": 0.5}, accuracy=0.9)
    result = data.extract(value)
    assert not families(result)
    assert result["context"]["current"]["unmapped_declared_source_present"] is True


def test_source_substring_and_general_math_not_canonical_identity():
    for text in ("evil/openai/gsm8k", "openai/gsm8k_extra", "MATH", "AIME 2025", "GSM8K evaluation benchmark"):
        assert not families(data.extract(example(text)))


def test_only_source_field_not_results_selection_prose_or_counts():
    value = example()
    value["model_input"]["plan"]["setup"]["data"] = [{
        "source": "unknown", "selection": "CANARY openai/gsm8k with accuracy 0.999991",
        "n_examples": 918273, "mixture_weight": 0.87,
    }]
    value["model_input"]["plan"]["result"] = {"accuracy": 0.123456789}
    result = data.extract(value)
    assert not families(result)
    context = json.dumps(result["context"])
    assert all(term not in context for term in ("CANARY", "918273", "0.87", "0.999991", "0.123456789"))


@pytest.mark.parametrize("prefix,call", [
    ("from datasets import load_dataset", "load_dataset('openai/gsm8k', 'main', split='train')"),
    ("import datasets as ds", "ds.load_dataset(path='openai/gsm8k', split='train[:50%]')"),
    ("from datasets import load_dataset as ld", "ld('openai/gsm8k')['train']"),
])
def test_training_builder_calls(prefix, call):
    result = data.extract(example(code=prefix + "\ntrain = " + call))
    assert families(result) == {"gsm8k"}
    assert result["features"]["recipe_data.current.code_only_identity"] == 1
    assert "PRIVATE" not in json.dumps(result["context"])


@pytest.mark.parametrize("body", [
    "x = load_dataset('openai/gsm8k', split='test')",
    "x = load_dataset('openai/gsm8k', split='validation[:20]')",
    "x = load_dataset('openai/gsm8k')['test']",
    "test_data = load_dataset('openai/gsm8k', split='train')",
    "eval_data = load_dataset('openai/gsm8k', split='train')",
    "def evaluate():\n    return load_dataset('openai/gsm8k', split='train')",
    "evaluate(load_dataset('openai/gsm8k', split='train'))",
    "if False:\n    x = load_dataset('openai/gsm8k', split='train')",
    "if 0:\n    x = load_dataset('openai/gsm8k', split='train')",
    "# load_dataset('openai/gsm8k', split='train')",
    "text = \"load_dataset('openai/gsm8k', split='train')\"",
    "print('openai/gsm8k')",
])
def test_eval_comments_strings_and_dead_code_excluded(body):
    result = data.extract(example(code="from datasets import load_dataset\n" + body))
    assert not families(result)


@pytest.mark.parametrize("role,status", [("training", "reconstructed"), ("evaluation", "reconstructed"),
                                         ("data_builder_0", "snapshot"), ("data_builder_other", "reconstructed")])
def test_only_reconstructed_builder_code(role, status):
    code = "from datasets import load_dataset\nx=load_dataset('openai/gsm8k',split='train')"
    assert not families(data.extract(example(code=code, role=role, status=status)))


def test_import_shadowing_does_not_fake_a_loader():
    code = "from datasets import load_dataset\ndef load_dataset(x, **kwargs):\n    return []\nx=load_dataset('openai/gsm8k',split='train')"
    assert not families(data.extract(example(code=code)))
    assert not families(data.extract(example(code="x=datasets.load_dataset('openai/gsm8k',split='train')")))


def test_cache_paths_are_only_read_from_known_builder_sinks():
    path = "/SECRET/datasets--nvidia--OpenMathReasoning/snapshots/PRIVATE/data/cot-*.parquet"
    code = "import glob\nROOT=" + repr(path) + "\nfiles=glob.glob(ROOT)"
    result = data.extract(example(code=code))
    assert families(result) == {"openmathreasoning"}
    assert all(token not in json.dumps(result["context"]) for token in ("SECRET", "PRIVATE", "parquet"))
    assert not families(data.extract(example(code="import glob\nprint(" + repr(path) + ")")))
    assert not families(data.extract(example(code="import glob\nfiles=glob.glob(" + repr(path.replace("cot-", "test-")) + ")")))


def test_local_shadowing_of_cache_constant_is_unknown():
    path = "/datasets--nvidia--OpenMathReasoning/snapshots/PRIVATE/data/cot-*.parquet"
    code = "import glob\nROOT=" + repr(path) + "\ndef build(ROOT):\n    return glob.glob(ROOT)"
    assert not families(data.extract(example(code=code)))


def test_hub_download_requires_dataset_repo_type():
    prefix = "from huggingface_hub import snapshot_download\n"
    result = data.extract(example(code=prefix + "snapshot_download('nvidia/OpenMathReasoning',repo_type='dataset')"))
    assert families(result) == {"openmathreasoning"}
    assert not families(data.extract(example(code=prefix + "snapshot_download('nvidia/OpenMathReasoning')")))


def test_history_is_identity_only_current_not_backfilled():
    value = example("derived:exp-01")
    parent = example("nvidia/OpenMathReasoning")["model_input"]
    parent["plan"]["result"] = {"accuracy": 0.999991}
    value["history"] = [{"recipe_status": "screened", "model_input": parent, "accuracy": 0.888887}]
    result = data.extract(value)
    assert not families(result, "current")
    assert families(result, "parent") == {"openmathreasoning"}
    assert result["features"]["recipe_data.history.any_family.openmathreasoning"] == 1
    assert "0.999991" not in json.dumps(result) and "0.888887" not in json.dumps(result)


def test_quarantined_latest_parent_does_not_backfill_older_recipe():
    value = example("openai/gsm8k")
    value["history"] = [
        {"recipe_status": "screened", "model_input": example("nvidia/OpenMathReasoning")["model_input"]},
        {"recipe_status": "quarantined", "model_input": example("open-r1/OpenR1-Math-220k")["model_input"]},
    ]
    result = data.extract(value)
    assert not families(result, "parent")
    assert result["context"]["immediate_parent_recipe"]["status"] == "quarantined"
    assert result["features"]["recipe_data.history.any_family.openmathreasoning"] == 1
    assert result["features"]["recipe_data.history.any_family.openr1_math220k"] == 0
    assert result["features"]["recipe_data.history.unresolved_steps"] == 1


def test_no_mutation_and_only_safe_context_changes_with_source():
    value = example("openai/gsm8k train: CANARY_ID 999999 rows, accuracy 0.923456789")
    original = copy.deepcopy(value)
    first = data.extract(value)
    value["model_input"]["plan"]["setup"]["data"][0]["source"] = "openai/gsm8k train"
    second = data.extract(value)
    assert first["features"] == second["features"] and first["context"] == second["context"]
    assert "CANARY_ID" not in json.dumps(first)
    assert first["provenance"] != second["provenance"]
    assert original["model_input"]["plan"]["setup"]["data"][0]["source"].endswith("0.923456789")
