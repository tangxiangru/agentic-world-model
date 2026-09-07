"""Synthetic checkpoint-input resolution; never execute trajectory Python."""

import copy
import textwrap

import pytest

from tools.outcome_prediction.wm_checkpoint_inputs import resolve_inputs


def row(argv=None, code=None, parent=None, family="sft", cwd="/work", script="/work/train.py"):
    return {
        "model_input": {
            "task": {"base_model": "org/base"},
            "plan": {
                "setup": {
                    "base_model": "org/base",
                    "method": {"family": family},
                    "command": {
                        "argv": argv or ["python", "train.py"],
                        "cwd": cwd,
                        "script": script,
                    },
                    "parent_checkpoint": {"path": parent, "origin": "exp-99"},
                }
            },
            "code": []
            if code is None
            else [
                {"script_path": script, "status": "reconstructed", "content": textwrap.dedent(code)}
            ],
        }
    }


def paths(result):
    return [item["path"] for item in result["inputs"]]


SINGLE = """
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default='org/base')
    p.add_argument('--out', default='result')
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('org/unrelated-tokenizer')
    model = AutoModelForCausalLM.from_pretrained(args.model)
    trainer = Trainer(model=model)
    trainer.train(resume_from_checkpoint=args.resume)
if __name__ == '__main__':
    main()
"""


def test_argparse_model_default_identifies_base_without_invented_accuracy():
    sample = row(code=SINGLE)
    before = copy.deepcopy(sample)
    result = resolve_inputs(sample)
    assert result["status"] == "base"
    assert result["inputs"] == [{"path": "org/base", "base_model": "org/base"}]
    assert not result["reasons"]
    assert sample == before


@pytest.mark.parametrize(
    "option", [["--model", "runs/parent"], ["--model=runs/parent"], ["-m", "runs/parent"]]
)
def test_explicit_argv_overrides_default_and_stale_structured_parent(option):
    result = resolve_inputs(row(["python", "train.py", *option], SINGLE, parent="org/base"))
    assert result["status"] == "single"
    assert paths(result) == ["/work/runs/parent"]
    assert any(
        e["source"] == "structured_parent_disagrees_with_stronger_input" for e in result["evidence"]
    )


def test_same_option_repeated_follows_argparse_last_value():
    result = resolve_inputs(row(["python", "train.py", "--model", "old", "--model", "new"], SINGLE))
    assert result["status"] == "single"
    assert paths(result) == ["/work/new"]


def test_verified_trainer_resume_overrides_initial_model_weights():
    result = resolve_inputs(
        row(
            [
                "python",
                "train.py",
                "--model",
                "org/base",
                "--resume",
                "runs/current/checkpoint-500",
            ],
            SINGLE,
        )
    )
    assert result["status"] == "single"
    assert paths(result) == ["/work/runs/current/checkpoint-500"]
    assert any(
        e["source"] == "verified_trainer_resume_overrides_initial_model" for e in result["evidence"]
    )


def test_unused_resume_flag_is_not_assumed_to_override_model():
    source = SINGLE.replace("trainer.train(resume_from_checkpoint=args.resume)", "trainer.train()")
    result = resolve_inputs(row(["python", "train.py", "--resume", "runs/other"], source))
    assert result["status"] == "base"
    assert paths(result) == ["org/base"]


def test_resume_without_reconstructed_usage_remains_unresolved():
    result = resolve_inputs(
        row(
            ["python", "train.py", "--model", "org/base", "--resume_from_checkpoint", "runs/new"],
            parent="org/base",
        )
    )
    assert result["status"] == "unresolved"
    assert "resume_precedence_not_verified_in_code" in result["reasons"]


@pytest.mark.parametrize("nargs", ["'+'", "2"])
def test_merge_nargs_paths_and_numeric_weights_are_different_arguments(nargs):
    source = f"""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--src', nargs={nargs})
    p.add_argument('--weights', nargs='*', type=float)
    p.add_argument('--out')
    args = p.parse_args()
    """
    result = resolve_inputs(
        row(
            ["python", "train.py", "--src", "a", "b", "--weights", ".3", ".7", "--out", "dest"],
            source,
            family="soup",
        )
    )
    assert result["status"] == "multiple"
    assert paths(result) == ["/work/a", "/work/b"]
    assert all(".3" not in p and ".7" not in p for p in paths(result))


def test_positional_after_single_model_option_is_not_swallowed_as_parent():
    source = """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('output')
    args = p.parse_args()
    """
    result = resolve_inputs(
        row(["python", "train.py", "--model", "runs/parent", "runs/output"], source)
    )
    assert result["status"] == "single"
    assert paths(result) == ["/work/runs/parent"]


def test_literal_shell_cd_env_wrapper_resolves_effective_cwd():
    sample = row(
        ["bash", "-lc", "cd /other && env CUDA_VISIBLE_DEVICES=0 python train.py --model ckpt"],
        script="/other/train.py",
    )
    result = resolve_inputs(sample)
    assert result["status"] == "single"
    assert paths(result) == ["/other/ckpt"]
    assert {e["source"] for e in result["evidence"]} >= {
        "literal_shell_command",
        "literal_cd",
        "env_assignments",
    }


def test_distributed_launcher_flags_do_not_become_model_arguments():
    result = resolve_inputs(
        row(["torchrun", "--nproc_per_node", "2", "train.py", "--model", "parent"], SINGLE)
    )
    assert result["status"] == "single"
    assert paths(result) == ["/work/parent"]


def test_mismatched_declared_entrypoint_is_not_backfilled():
    result = resolve_inputs(row(["python", "different.py", "--model", "parent"], SINGLE))
    assert result["status"] == "unresolved"
    assert "command_script_conflict" in result["reasons"]


def test_mutable_final_snapshot_is_never_inspected():
    sample = row(parent="org/base", code="raise RuntimeError('do not run')")
    sample["model_input"]["code"][0]["status"] = "snapshot_only"
    result = resolve_inputs(sample)
    assert result["status"] == "base"
    assert not any(e["source"] == "reconstructed_entrypoint" for e in result["evidence"])


def test_conflicting_reconstructed_entrypoints_remain_unresolved():
    sample = row(code=SINGLE)
    duplicate = copy.deepcopy(sample["model_input"]["code"][0])
    duplicate["content"] = SINGLE.replace("org/base", "org/other")
    sample["model_input"]["code"].append(duplicate)
    result = resolve_inputs(sample)
    assert result["status"] == "unresolved"
    assert "conflicting_reconstructed_entrypoint" in result["reasons"]


@pytest.mark.parametrize(
    "path", ["$MODEL", "${MODEL}", "$(secret)", "~/ckpt", "runs/*", "runs/<choice>"]
)
def test_dynamic_paths_are_never_expanded_or_guessed(path):
    assert (
        resolve_inputs(row(["python", "train.py", "--model", path], SINGLE))["status"]
        == "unresolved"
    )


def test_unknown_branch_model_choice_remains_unresolved():
    source = """
    import os
    from transformers import AutoModelForCausalLM
    if os.environ.get('WHICH'):
        model = AutoModelForCausalLM.from_pretrained('a')
    else:
        model = AutoModelForCausalLM.from_pretrained('b')
    """
    result = resolve_inputs(row(code=source, parent="a"))
    assert result["status"] == "unresolved"


def test_uncalled_helper_model_is_not_a_consumed_checkpoint():
    source = (
        SINGLE + "\ndef unused():\n    return AutoModelForCausalLM.from_pretrained('unrelated')\n"
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "base"
    assert paths(result) == ["org/base"]


def test_labels_prior_observations_origin_and_result_objects_are_never_accessed():
    class Explodes:
        def __iter__(self):
            raise AssertionError("forbidden outcome object accessed")

        def __str__(self):
            raise AssertionError("forbidden outcome object accessed")

    sample = row(code=SINGLE)
    sample["label"] = sample["prior_observations"] = Explodes()
    sample["model_input"]["plan"]["setup"]["parent_checkpoint"]["origin"] = Explodes()
    sample["model_input"]["plan"]["result"] = Explodes()
    assert resolve_inputs(sample)["status"] == "base"


def test_no_arbitrary_code_execution(tmp_path):
    sentinel = tmp_path / "must_not_exist"
    source = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n" + SINGLE
    result = resolve_inputs(row(code=source))
    assert not sentinel.exists()
    assert result["status"] == "base"


def test_numeric_merge_weight_is_not_a_checkpoint_even_without_code():
    result = resolve_inputs(
        row(["python", "train.py", "--models", "a", "b", "--weights", ".3", ".7"], family="soup")
    )
    assert result["status"] == "multiple"
    assert paths(result) == ["/work/a", "/work/b"]


def test_dynamic_merge_loads_are_not_overridden_by_weaker_argparse_source_declarations():
    source = """
    import argparse
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser()
    p.add_argument('--srcs', nargs='+')
    args = p.parse_args()
    for checkpoint in args.srcs:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
    """
    result = resolve_inputs(row(["python", "train.py", "--srcs", "a", "b"], source, family="soup"))
    assert result["status"] == "unresolved"
    assert "unknown_or_nonliteral_checkpoint" in result["reasons"]


def test_repeated_weighted_checkpoint_arguments_need_explicit_split_code():
    source = """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append')
    args = p.parse_args()
    for source in args.ckpt:
        checkpoint, coefficient = source.rsplit(':', 1)
    """
    result = resolve_inputs(
        row(["python", "train.py", "--ckpt", "a:0.3", "--ckpt", "b:0.7"], source, family="soup")
    )
    assert result["status"] == "multiple"
    assert paths(result) == ["/work/a", "/work/b"]
    assert any(
        e["source"] == "code_verified_checkpoint_weight_separator" for e in result["evidence"]
    )


def test_unknown_kwargs_that_might_override_model_are_not_assumed_safe():
    source = """
    from transformers import AutoModelForCausalLM
    options = read_runtime_configuration()
    model = AutoModelForCausalLM.from_pretrained('org/base', **options)
    """
    assert resolve_inputs(row(code=source, parent="org/base"))["status"] == "unresolved"


def test_true_resume_means_unknown_checkpoint_not_literal_true_path():
    source = SINGLE.replace("default=None", "action='store_true'")
    result = resolve_inputs(row(["python", "train.py", "--resume"], source))
    assert result["status"] == "unresolved"
    assert not result["inputs"]


def test_known_parser_rejects_undeclared_options_instead_of_falling_back():
    result = resolve_inputs(
        row(["python", "train.py", "--checkpoint-unknown", "x"], SINGLE, parent="org/base")
    )
    assert result["status"] == "unresolved"
    assert "undeclared_command_option" in result["reasons"]


def test_differently_named_code_file_does_not_supply_a_default():
    sample = row(code=SINGLE)
    sample["model_input"]["code"][0]["script_path"] = "/other/train.py"
    result = resolve_inputs(sample)
    assert result["status"] == "unresolved"
    assert "no_statically_resolved_checkpoint_input" in result["reasons"]


def test_hf_snapshot_base_requires_full_pinned_snapshot_not_prefix_matching():
    pinned = "/cache/models--org--base/snapshots/" + "a" * 40
    result = resolve_inputs(row(parent=pinned))
    assert result["status"] == "base"
    prefix = resolve_inputs(row(parent="/cache/models--org--base/snapshots"))
    assert prefix["status"] == "single"
    assert "base_model" not in prefix["inputs"][0]


def test_multiple_distinct_model_loads_are_not_averaged_to_a_single_parent():
    source = """
    from transformers import AutoModelForCausalLM
    model_a = AutoModelForCausalLM.from_pretrained('a')
    model_b = AutoModelForCausalLM.from_pretrained('b')
    """
    result = resolve_inputs(row(code=source))
    assert result["status"] == "multiple"
    assert paths(result) == ["/work/a", "/work/b"]


def test_unknown_runtime_assertion_is_a_disclosed_condition_not_a_path_choice():
    source = SINGLE.replace(
        "model = AutoModelForCausalLM.from_pretrained(args.model)",
        "assert 'end' in tokenizer.get_vocab()\n    model = AutoModelForCausalLM.from_pretrained(args.model)",
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "base"
    assert any(
        e["source"] == "input_conditional_on_runtime_assertion_passing" for e in result["evidence"]
    )


@pytest.mark.parametrize("guard", ["False", "0", "''"])
def test_statically_failing_assertion_is_unresolved(guard):
    source = SINGLE.replace(
        "model = AutoModelForCausalLM.from_pretrained(args.model)",
        f"assert {guard}\n    model = AutoModelForCausalLM.from_pretrained(args.model)",
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "unresolved"
    assert "statically_failed_assertion" in result["reasons"]


def test_weighted_checkpoint_rpartition_syntax_is_supported():
    source = """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append')
    args = p.parse_args()
    for source in args.ckpt:
        checkpoint, separator, coefficient = source.rpartition(':')
    """
    result = resolve_inputs(
        row(["python", "train.py", "--ckpt", "a:1", "--ckpt", "b:1"], source, family="soup")
    )
    assert result["status"] == "multiple"
    assert paths(result) == ["/work/a", "/work/b"]


def test_custom_trainer_with_inherited_train_can_verify_resume():
    source = SINGLE.replace(
        "trainer = Trainer(model=model)",
        "class CustomTrainer(Trainer):\n        def compute_loss(self, model, inputs):\n            return unknown_loss\n    trainer = CustomTrainer(model=model)",
    )
    result = resolve_inputs(row(["python", "train.py", "--resume", "ckpt"], source))
    assert result["status"] == "single"
    assert paths(result) == ["/work/ckpt"]


@pytest.mark.parametrize("override", ["train", "__init__", "__getattr__"])
def test_custom_trainer_override_does_not_prove_resume_precedence(override):
    source = SINGLE.replace(
        "trainer = Trainer(model=model)",
        f"class CustomTrainer(Trainer):\n        def {override}(self, *args, **kwargs):\n            pass\n    trainer = CustomTrainer(model=model)",
    )
    result = resolve_inputs(row(["python", "train.py", "--resume", "ckpt"], source))
    assert result["status"] == "unresolved"


def test_loading_adapter_adds_a_distinct_checkpoint_input():
    source = SINGLE.replace(
        "trainer = Trainer(model=model)",
        "model.load_adapter('adapter')\n    trainer = Trainer(model=model)",
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "multiple"
    assert paths(result) == ["org/base", "/work/adapter"]


def test_loading_opaque_state_dict_does_not_leave_a_false_base_reference():
    source = SINGLE.replace(
        "trainer = Trainer(model=model)",
        "model.load_state_dict(unknown_weights())\n    trainer = Trainer(model=model)",
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "unresolved"


def test_weight_method_monkeypatch_is_not_treated_as_verified_library_usage():
    source = SINGLE.replace(
        "trainer.train(resume_from_checkpoint=args.resume)",
        "trainer.train = unknown_method\n    trainer.train(resume_from_checkpoint=args.resume)",
    )
    result = resolve_inputs(row(["python", "train.py", "--resume", "ckpt"], source))
    assert result["status"] == "unresolved"
    assert "dynamic_weight_method_override" in result["reasons"]


def test_conditional_adapter_load_keeps_input_identity_unresolved():
    source = SINGLE.replace(
        "trainer = Trainer(model=model)",
        "if unknown_condition:\n        model.load_adapter('adapter')\n    trainer = Trainer(model=model)",
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "unresolved"


def test_required_option_is_not_satisfied_by_its_default_value():
    source = SINGLE.replace("default='org/base'", "default='org/base', required=True")
    result = resolve_inputs(row(code=source))
    assert result["status"] == "unresolved"
    assert "missing_required_argument" in result["reasons"]
    supplied = resolve_inputs(row(["python", "train.py", "--model", "org/base"], source))
    assert supplied["status"] == "base"


def test_hardcoded_merge_load_is_stronger_than_unused_source_argument():
    source = """
    import argparse
    from transformers import AutoModelForCausalLM
    p=argparse.ArgumentParser()
    p.add_argument('--src')
    args=p.parse_args()
    model=AutoModelForCausalLM.from_pretrained('/actual')
    """
    result = resolve_inputs(
        row(["python", "train.py", "--src", "/declared"], source, family="soup")
    )
    assert result["status"] == "single"
    assert paths(result) == ["/actual"]


@pytest.mark.parametrize(
    "call",
    [
        "trainer.train(**runtime_kwargs())",
        "trainer.train('/parent', resume_from_checkpoint='/other')",
    ],
)
def test_opaque_or_conflicting_resume_arguments_do_not_leave_base_parent(call):
    source = SINGLE.replace("trainer.train(resume_from_checkpoint=args.resume)", call)
    assert resolve_inputs(row(code=source))["status"] == "unresolved"


def test_positional_trainer_resume_is_a_verified_checkpoint():
    source = SINGLE.replace(
        "trainer.train(resume_from_checkpoint=args.resume)", "trainer.train('/parent')"
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "single"
    assert paths(result) == ["/parent"]


def test_literal_chdir_changes_model_path_resolution_at_the_load_call():
    source = """
    import os
    from transformers import AutoModelForCausalLM
    os.chdir('/other')
    model=AutoModelForCausalLM.from_pretrained('parent')
    os.chdir('/later')
    """
    result = resolve_inputs(row(code=source))
    assert result["status"] == "single"
    assert paths(result) == ["/other/parent"]


def test_conditional_chdir_makes_later_relative_model_path_unknown():
    source = """
    import os
    from transformers import AutoModelForCausalLM
    if unknown:
        os.chdir('/other')
    model=AutoModelForCausalLM.from_pretrained('parent')
    """
    assert resolve_inputs(row(code=source))["status"] == "unresolved"


def test_custom_argparse_type_does_not_leave_unconverted_string_default():
    source = SINGLE.replace("default='org/base'", "type=runtime_converter, default='org/base'")
    assert resolve_inputs(row(code=source))["status"] == "unresolved"


def test_merge_and_unload_preserves_identity_for_subsequent_adapter_load():
    source = SINGLE.replace(
        "trainer = Trainer(model=model)",
        "model=model.merge_and_unload()\n    model.load_adapter('other')\n    trainer = Trainer(model=model)",
    )
    result = resolve_inputs(row(code=source))
    assert result["status"] == "multiple"
    assert paths(result) == ["org/base", "/work/other"]


@pytest.mark.parametrize(
    "call", ["load(**runtime_kwargs())", "load('/a',path='/b')", "load(*runtime_values)"]
)
def test_ambiguous_local_helper_binding_cannot_backfill_its_default(call):
    source = f"""
    from transformers import AutoModelForCausalLM
    def load(path='org/base'):
        return AutoModelForCausalLM.from_pretrained(path)
    model={call}
    """
    assert resolve_inputs(row(code=source))["status"] == "unresolved"
