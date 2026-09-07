"""Synthetic checkpoint-path proofs; no training code is executed."""

import copy

import pytest

from tools.outcome_prediction.wm_checkpoint_paths import resolve_target_binding

HEADER = """import os
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, GenerationConfig
from peft import get_peft_model
model = AutoModelForCausalLM.from_pretrained('base')
tokenizer = AutoTokenizer.from_pretrained('base')
"""


def fixture(code, *, planned="/task/exp", target="/task/exp/final", argv=None):
    return {
        "example_id": "cell/exp-01",
        "cell_id": "cell",
        "first_stage": "plan",
        "first_submitted_at": "2026-09-05T13:48:09Z",
        "model_input": {
            "plan": {
                "setup": {
                    "output_dir": planned,
                    "command": {
                        "script": "/task/train.py",
                        "argv": argv or ["python", "train.py"],
                        "cwd": "/task",
                    },
                    "method": {"family": "sft"},
                }
            },
            "code": [
                {
                    "role": "training",
                    "script_path": "/task/train.py",
                    "status": "reconstructed",
                    "content": HEADER + code,
                }
            ],
        },
        "audit": {
            "reasons": [],
            "first_final_setup_changed_fields": [],
            "output_artifact_comparison": {
                "first_declared_output_dir": planned,
                "final_output_checkpoint": target,
            },
        },
    }


def assert_resolved(row, expected="planned_save_path"):
    answer = resolve_target_binding(row)
    assert answer["status"] == expected, answer
    assert answer["reasons"] == []
    assert answer["runtime_certified"] is False
    assert answer["checkpoint_bytes_certified"] is False
    assert answer["score_values_read"] is False
    return answer


def assert_unknown(row):
    answer = resolve_target_binding(row)
    assert answer["status"] == "unresolved", answer
    assert answer["reasons"]
    return answer


@pytest.mark.parametrize(
    "planned,target",
    [
        ("/task/exp", "/task/exp"),
        ("/task/exp/", "/task/exp"),
        ("exp", "/task/exp"),
        ("/task/./exp", "/task/exp"),
        ("/task/a/../exp", "/task/exp"),
    ],
)
def test_exact_lexical_paths_without_suffix_inference(planned, target):
    answer = assert_resolved(fixture("", planned=planned, target=target), "exact")
    assert answer["evidence"][0]["kind"] == "lexical_path_identity"


@pytest.mark.parametrize("path", [None, "", " ", "$OUT", "~/exp", "/task/\x00exp"])
def test_unknown_or_dynamic_declared_path_fails_closed(path):
    assert_unknown(fixture("", planned=path))


@pytest.mark.parametrize(
    "target",
    ["/task/exp/final", "/task/exp/merged", "/task/exp/checkpoint-100", "/task/exp/pkg_greedy"],
)
def test_no_blanket_final_merged_or_checkpoint_prefix_alias(target):
    assert_unknown(fixture("model.save_pretrained('/task/exp')", target=target))


@pytest.mark.parametrize(
    "expression",
    [
        "'/task/exp/final'",
        "os.path.join('/task/exp', 'final')",
        "'/task/exp' + '/final'",
        "Path('/task/exp') / 'final'",
        "Path('/task/exp').joinpath('final')",
        "f'{out}/final'",
        "os.path.abspath('exp/final')",
        "os.path.normpath('/task/exp/./final')",
    ],
)
def test_explicit_supported_path_expressions(expression):
    row = fixture(f"out='/task/exp'\nmodel.save_pretrained({expression})")
    answer = assert_resolved(row)
    assert answer["candidate_save_paths"] == ["/task/exp/final"]
    assert len(answer["evidence"][-1]["script_sha256"]) == 64


def test_cli_path_override_has_precedence_over_default():
    code = "p=argparse.ArgumentParser()\np.add_argument('--out', default='wrong')\na=p.parse_args()\nmodel.save_pretrained(os.path.join(a.out,'final'))"
    assert_resolved(fixture(code, argv=["python", "train.py", "--out", "exp"]))
    assert_unknown(fixture(code))


def test_argparse_default_and_explicitly_called_helpers():
    code = """def args():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='exp')
    return p.parse_args()
def save_final(m, root):
    final = os.path.join(root, 'final')
    m.save_pretrained(final)
def main():
    a=args()
    save_final(model,a.out)
if __name__ == '__main__':
    main()
"""
    assert_resolved(fixture(code))


def test_planned_lora_merge_proves_merged_destination():
    code = """model=get_peft_model(model,None)
trainer=Trainer(model=model,args=TrainingArguments(output_dir='/task/exp'))
trainer.train()
merged=trainer.model.merge_and_unload()
merged.save_pretrained('/task/exp/merged')
tokenizer.save_pretrained('/task/exp/merged')
"""
    answer = assert_resolved(fixture(code, target="/task/exp/merged"))
    assert any(e["receiver_kind"] == "planned_lora_merge" for e in answer["evidence"])


@pytest.mark.parametrize(
    "save",
    [
        "trainer.save_model('/task/exp/final')",
        "trainer.save_model(output_dir='/task/exp/final')",
        "trainer.save_model()",
    ],
)
def test_trainer_save_with_explicit_or_declared_default_directory(save):
    code = (
        "trainer=Trainer(model=model,args=TrainingArguments(output_dir='/task/exp/final'))\ntrainer.train()\n"
        + save
    )
    assert_resolved(fixture(code))


def test_save_pretrained_keyword_destination():
    assert_resolved(fixture("model.save_pretrained(save_directory='/task/exp/final')"))


def test_custom_trainer_inheriting_save_method_is_supported():
    code = """class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        return unknown_runtime_loss
trainer=MyTrainer(model=model,args=TrainingArguments(output_dir='/task/exp'))
trainer.train()
trainer.save_model('/task/exp/final')
"""
    assert_resolved(fixture(code))


@pytest.mark.parametrize("override", ["save_model", "save_pretrained", "__init__", "__getattr__"])
def test_custom_trainer_with_save_or_constructor_override_is_unknown(override):
    code = f"class MyTrainer(Trainer):\n def {override}(self,*args):\n  pass\ntrainer=MyTrainer(model=model)\ntrainer.save_model('/task/exp/final')"
    assert_unknown(fixture(code))


def test_earlier_periodic_checkpoints_are_dominated_by_fixed_terminal_save():
    code = """trainer=Trainer(model=model)
for step in unknown_runtime_steps:
    if step % 100 == 0:
        model.save_pretrained(f'/task/exp/checkpoint-{step}')
trainer.train()
model.save_pretrained('/task/exp/final')
"""
    answer = assert_resolved(fixture(code))
    assert any(not e["unconditional"] for e in answer["evidence"])


def test_earlier_explicit_checkpoint_is_not_confused_with_terminal_model():
    code = "model.save_pretrained('/task/exp/checkpoint-100')\nmodel.save_pretrained('/task/exp/final')"
    assert_resolved(fixture(code))
    assert_unknown(fixture(code, target="/task/exp/checkpoint-100"))


def test_multiple_planned_versions_require_exact_terminal_destination():
    code = "model.save_pretrained('/task/exp/final')\nmodel.save_pretrained('/task/exp/another')"
    assert_unknown(fixture(code))
    assert_resolved(fixture(code, target="/task/exp/another"))


@pytest.mark.parametrize(
    "tail",
    [
        "if unknown:\n model.save_pretrained('/task/exp/other')",
        "model.save_pretrained(runtime_choice())",
        "model.save_pretrained(best_checkpoint_path)",
    ],
)
def test_later_dynamic_or_conditional_save_prevents_terminal_alias_proof(tail):
    assert_unknown(fixture("model.save_pretrained('/task/exp/final')\n" + tail))


def test_weight_training_after_save_is_not_terminal_output():
    code = "trainer=Trainer(model=model)\ntrainer.save_model('/task/exp/final')\ntrainer.train()"
    assert_unknown(fixture(code))


@pytest.mark.parametrize(
    "code",
    [
        "tokenizer.save_pretrained('/task/exp/final')",
        "GenerationConfig().save_pretrained('/task/exp/final')",
        "model.generation_config.save_pretrained('/task/exp/final')",
        "unknown_model.save_pretrained('/task/exp/final')",
        "def unused():\n model.save_pretrained('/task/exp/final')",
    ],
)
def test_tokenizer_config_unknown_receiver_or_uncalled_function_is_not_model_proof(code):
    assert_unknown(fixture(code))


def test_auxiliary_save_after_model_does_not_invalidate_known_model_destination():
    code = "model.save_pretrained('/task/exp/final')\ntokenizer.save_pretrained('/task/exp/final')\nmodel.generation_config.save_pretrained('/task/exp/final')"
    assert_resolved(fixture(code))


def test_unknown_auxiliary_try_block_does_not_invalidate_model_save():
    code = "model.save_pretrained('/task/exp/final')\ntry:\n tokenizer.save_pretrained('/task/exp/final')\nexcept Exception:\n pass"
    assert_resolved(fixture(code))


@pytest.mark.parametrize(
    "expression",
    [
        "os.environ['OUT']",
        "os.getenv('OUT')",
        "runtime_output()",
        "Path('/task/exp/final').resolve()",
        "f'/task/exp/{runtime_choice()}'",
    ],
)
def test_environment_filesystem_or_runtime_derived_paths_remain_unknown(expression):
    assert_unknown(fixture(f"model.save_pretrained({expression})"))


def test_recorded_final_snapshot_is_not_preproposal_proof():
    row = fixture("model.save_pretrained('/task/exp/final')")
    row["model_input"]["code"][0]["status"] = "snapshot"
    assert_unknown(row)


def test_mismatched_entrypoint_and_same_basename_elsewhere_are_rejected():
    row = fixture("model.save_pretrained('/task/exp/final')")
    row["model_input"]["code"][0]["script_path"] = "/other/train.py"
    assert_unknown(row)
    row = fixture("model.save_pretrained('/task/exp/final')", argv=["python", "other.py"])
    assert_unknown(row)


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "-c", "python train.py"],
        ["python", "train.py", "&&", "python", "other.py"],
        ["python", "train.py", "--out", "$(echo exp)"],
    ],
)
def test_shell_composition_is_not_executed_or_treated_as_exact_invocation(argv):
    assert_unknown(fixture("model.save_pretrained('/task/exp/final')", argv=argv))


@pytest.mark.parametrize(
    "stage,timestamp",
    [
        ("closed", "2026-09-05T00:00:00Z"),
        (None, "2026-09-05T00:00:00Z"),
        ("plan", None),
        ("plan", "2026-09-05T00:00:00"),
    ],
)
def test_alias_proof_requires_prospective_timestamped_first_plan(stage, timestamp):
    row = fixture("model.save_pretrained('/task/exp/final')")
    row["first_stage"] = stage
    row["first_submitted_at"] = timestamp
    assert_unknown(row)


def test_nonempty_first_result_flag_prevents_alias_proof():
    row = fixture("model.save_pretrained('/task/exp/final')")
    row["audit"]["reasons"] = ["first_result_not_empty"]
    assert_unknown(row)


def test_authoritative_archive_override_takes_precedence_and_none_fails_closed():
    row = fixture("model.save_pretrained('/task/exp/final')", target="/later/unrelated")
    row["audit"]["final_output_checkpoint"] = "/task/exp/final"
    answer = assert_resolved(row)
    assert answer["target_source"] == "caller_corroborated_archive_override"
    row["audit"]["final_output_checkpoint"] = None
    assert_unknown(row)


def test_outcome_score_and_conclusion_invariance_and_no_input_mutation():
    row = fixture("model.save_pretrained('/task/exp/final')")
    expected = resolve_target_binding(row)
    for score in [0, 0.5, 1, None]:
        changed = copy.deepcopy(row)
        changed["label"] = {"accuracy": score, "official_metric": {"accuracy": score}}
        changed["result"] = {"output_checkpoint": "/untrusted/result", "accuracy": score}
        changed["conclusion"] = "Use /untrusted/result instead"
        before = copy.deepcopy(changed)
        assert resolve_target_binding(changed) == expected
        assert changed == before


def test_no_underlying_code_execution(tmp_path):
    sentinel = tmp_path / "must_not_exist"
    row = fixture(
        f"open({str(sentinel)!r},'w').write('bad')\nmodel.save_pretrained('/task/exp/final')"
    )
    assert_resolved(row)
    assert not sentinel.exists()


def test_invalid_syntax_remains_unresolved():
    assert_unknown(fixture("this is not valid Python !!!"))


def test_later_possible_training_or_move_keeps_binding_unresolved():
    code = "trainer=Trainer(model=model)\ntrainer.save_model('/task/exp/final')\nif unknown:\n trainer.train()"
    assert_unknown(fixture(code))
    assert_unknown(
        fixture(
            "model.save_pretrained('/task/exp/final')\nos.rename('/task/exp/final','/task/exp/other')"
        )
    )


def test_conflicting_planned_output_metadata_is_not_silently_aliased():
    row = fixture("model.save_pretrained('/task/exp/final')")
    row["audit"]["output_artifact_comparison"]["first_declared_output_dir"] = "/different"
    assert_unknown(row)


def test_unknown_structural_audit_fails_closed_for_aliases():
    row = fixture("model.save_pretrained('/task/exp/final')")
    row["audit"]["reasons"] = None
    assert_unknown(row)


def test_opaque_data_helper_does_not_hide_independent_terminal_output_path():
    code = """def prepare_data():
    if runtime_condition:
        return unknown_rows
    return other_rows
data=prepare_data()
trainer=Trainer(model=model, train_dataset=data)
trainer.train()
trainer.save_model('/task/exp/final')
"""
    answer = assert_resolved(fixture(code))
    assert answer["opaque_helpers"]


def test_opaque_path_helper_return_is_not_guessed():
    code = """def output_path():
    if runtime_condition:
        return '/task/exp/final'
    return '/task/exp/other'
model.save_pretrained(output_path())
"""
    assert_unknown(fixture(code))


def test_opaque_helper_cannot_hide_later_possible_checkpoint_selection():
    code = """def maybe_save():
    if runtime_condition:
        return
    model.save_pretrained('/task/exp/selected')
model.save_pretrained('/task/exp/final')
maybe_save()
"""
    assert_unknown(fixture(code))


def test_processor_imported_inside_try_is_auxiliary_not_later_model_selection():
    code = """model.save_pretrained('/task/exp/final')
try:
    from transformers import AutoProcessor as AP
    AP.from_pretrained('base').save_pretrained('/task/exp/final')
except Exception:
    pass
"""
    assert_resolved(fixture(code))


def test_shadowed_auxiliary_import_cannot_bypass_unknown_later_save():
    code = """model.save_pretrained('/task/exp/final')
try:
    from transformers import AutoProcessor as AP
    AP = unknown_model_factory
    AP.from_pretrained('base').save_pretrained('/task/exp/final')
except Exception:
    pass
"""
    assert_unknown(fixture(code))


@pytest.mark.parametrize(
    "operation",
    [
        "shutil.move('/task/exp/final','/other')",
        "os.rename('/task/exp/final','/other')",
        "Path('/task/exp/final').rename('/other')",
    ],
)
def test_conditional_later_path_mutation_invalidates_save(operation):
    code = f"import shutil\nmodel.save_pretrained('/task/exp/final')\nif unknown:\n {operation}"
    assert_unknown(fixture(code))


@pytest.mark.parametrize(
    "override",
    [
        "model.save_pretrained = runtime_override",
        "AutoModelForCausalLM.save_pretrained = runtime_override",
        "if unknown:\n model.save_pretrained = runtime_override",
    ],
)
def test_save_method_override_is_not_a_known_weight_save(override):
    assert_unknown(fixture(override + "\nmodel.save_pretrained('/task/exp/final')"))


@pytest.mark.parametrize(
    "call",
    [
        "model.save_pretrained('/task/exp/final', **runtime_kwargs())",
        "model.save_pretrained('/task/exp/final', save_directory='/other')",
        "model.save_pretrained('/task/exp/final', **{'save_directory':'/other'})",
        "model.save_pretrained(save_directory='/task/exp/final', **{'save_directory':'/other'})",
        "model.save_pretrained('/task/exp/final', safe_serialization=True, **{'safe_serialization':False})",
        "model.save_pretrained('/task/exp/final', output_dir='/other')",
        "model.save_model('/task/exp/final')",
        "model.save_checkpoint('/task/exp/final')",
        "Trainer(model=model).save_pretrained('/task/exp/final')",
        "Trainer(model=model).save_model('/task/exp/final', output_dir='/other')",
        "Trainer(model=model, args=TrainingArguments(output_dir='/task/exp/final')).save_model(**unknown)",
    ],
)
def test_invalid_opaque_or_nonstandard_weight_save_signature_is_unknown(call):
    assert_unknown(fixture(call))


def test_literal_nonconflicting_kwargs_keep_path_proof():
    assert_resolved(
        fixture("model.save_pretrained('/task/exp/final', **{'safe_serialization':True})")
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "train.py", "--not-declared", "oops"],
        ["python", "train.py", "unparsed-positional"],
        ["python", "train.py", "--out"],
    ],
)
def test_invalid_cli_does_not_silently_leave_valid_output_default(argv):
    code = "p=argparse.ArgumentParser()\np.add_argument('--out',default='/task/exp/final')\na=p.parse_args()\nmodel.save_pretrained(a.out)"
    assert_unknown(fixture(code, argv=argv))


def test_missing_required_option_invalidates_even_independent_literal_save():
    code = "p=argparse.ArgumentParser()\np.add_argument('--model',required=True,default='base')\na=p.parse_args()\nmodel.save_pretrained('/task/exp/final')"
    assert_unknown(fixture(code))
    assert_resolved(fixture(code, argv=["python", "train.py", "--model", "base"]))


def test_direct_import_filesystem_alias_in_conditional_branch_is_not_ignored():
    code = "from shutil import move as relocate\nmodel.save_pretrained('/task/exp/final')\nif unknown:\n relocate('/task/exp/final','/other')"
    assert_unknown(fixture(code))


def test_opaque_local_helper_cannot_hide_filesystem_mutation():
    code = """from shutil import move as relocate
def postprocess():
    if unknown:
        return
    relocate('/task/exp/final','/other')
model.save_pretrained('/task/exp/final')
postprocess()
"""
    assert_unknown(fixture(code))


@pytest.mark.parametrize(
    "call",
    [
        "save(**runtime_kwargs())",
        "save('/task/exp/final', out='/other')",
        "save('/task/exp/final', '/other')",
        "save(unexpected='/other')",
    ],
)
def test_local_save_helper_requires_unambiguous_python_argument_binding(call):
    code = "def save(out='/task/exp/final'):\n model.save_pretrained(out)\n" + call
    assert_unknown(fixture(code))


@pytest.mark.parametrize(
    "declaration",
    [
        "async def save():",
        "@runtime_decorator\ndef save():",
        "def save(*args):",
        "def save(**kwargs):",
    ],
)
def test_unsupported_helper_semantics_cannot_supply_a_save_proof(declaration):
    assert_unknown(fixture(declaration + "\n model.save_pretrained('/task/exp/final')\nsave()"))


def test_imported_path_function_monkeypatch_invalidates_literal_path_semantics():
    code = "os.path.join = runtime_join\nmodel.save_pretrained(os.path.join('/task/exp','final'))"
    assert_unknown(fixture(code))


def test_local_trainer_save_override_by_class_assignment_is_not_inherited():
    code = "class CustomTrainer(Trainer):\n save_model=runtime_override\ntrainer=CustomTrainer(model=model)\ntrainer.save_model('/task/exp/final')"
    assert_unknown(fixture(code))


def test_local_helper_global_mutation_is_not_approximated_as_a_local_assignment():
    code = "out='/task/exp/final'\ndef change():\n global out\n out='/other'\nchange()\nmodel.save_pretrained(out)"
    assert_unknown(fixture(code))


@pytest.mark.parametrize("conditional", [False, True])
def test_later_bound_save_method_alias_does_not_escape_terminal_conflict(conditional):
    tail = "if unknown:\n save('/other')" if conditional else "save('/other')"
    code = "save=model.save_pretrained\nmodel.save_pretrained('/task/exp/final')\n" + tail
    assert_unknown(fixture(code))


@pytest.mark.parametrize("conditional", [False, True])
def test_later_raw_torch_weight_serialization_does_not_escape_terminal_conflict(conditional):
    call = "torch.save(model.state_dict(), '/other/model.bin')"
    tail = "if unknown:\n " + call if conditional else call
    code = "import torch\nmodel.save_pretrained('/task/exp/final')\n" + tail
    assert_unknown(fixture(code))
