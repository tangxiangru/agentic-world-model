"""Synthetic, outcome-independent checks for bounded configuration extraction."""

import copy
import json

import pytest

from tools.outcome_prediction.wm_code_features import FEATURE_KEYS, extract_config_features


def source(code, argv=(), hp=None):
    return {
        "plan": {"setup": {
            "command": {"argv": ["python", "train.py", *argv], "script": "/task/train.py", "cwd": "/task"},
            "method": {"hyperparams": hp or {}},
        }},
        "code": [{"script_path": "/task/train.py", "status": "reconstructed", "role": "training", "content": code}],
    }


def extract(code, argv=(), hp=None):
    return extract_config_features(source(code, argv, hp))


def test_argparse_defaults_argv_and_config_use_precedence():
    f, audit = extract('''
import argparse
from transformers import TrainingArguments as TA
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--ga', type=int, default=4)
    p.add_argument('--steps', type=int, default=40)
    args = p.parse_args()
    options = dict(learning_rate=args.lr / 2,
                   per_device_train_batch_size=args.bs,
                   gradient_accumulation_steps=args.ga,
                   max_steps=args.steps)
    cfg = TA(**options)
if __name__ == '__main__':
    main()
''', ['--lr', '0.00004', '--bs=16', '--steps', '100'], {'lr': 0.1})
    assert f['codecfg.learning_rate'] == 2e-5
    assert f['codecfg.batch_size'] == 16
    assert f['codecfg.grad_accum'] == 4
    assert f['codecfg.max_steps'] == 100
    assert f['codecfg.effective_batch_per_device'] == 64
    assert audit['field_sources']['learning_rate'] == ['configuration']
    assert not audit['runtime_certified']


@pytest.mark.parametrize('argv, expected', [([], 0.0), (['--pack'], 1.0), (['--pack', '--no-pack'], 0.0)])
def test_store_true_false_shared_destination(argv, expected):
    f, _ = extract('''
import argparse
from trl import SFTConfig
p = argparse.ArgumentParser()
p.add_argument('--pack', action='store_true', dest='packing')
p.add_argument('--no-pack', action='store_false', dest='packing', default=False)
a = p.parse_args()
c = SFTConfig(packing=a.packing)
''', argv)
    assert f['codecfg.packing'] == expected


def test_store_false_and_conditional_resolution():
    f, _ = extract('''
import argparse
from trl import SFTConfig
p = argparse.ArgumentParser()
p.add_argument('--no-gc', action='store_false', dest='gc', default=True)
a = p.parse_args()
if a.gc:
    cfg = SFTConfig(gradient_checkpointing=True)
else:
    cfg = SFTConfig(gradient_checkpointing=False)
''', ['--no-gc'])
    assert f['codecfg.gradient_checkpointing'] == 0


def test_import_aliases_dict_mutations_and_precision_separation():
    f, _ = extract('''
import torch as t
from transformers import TrainingArguments as Args, AutoModelForCausalLM
kw = {'learning_rate': 1e-5, 'lr_scheduler_type': 'cosine_with_min_lr'}
kw.update(weight_decay=0.01)
kw['lr_scheduler_kwargs'] = {'min_lr_rate': 0.1}
model = AutoModelForCausalLM.from_pretrained('not-a-feature', torch_dtype=t.float32)
cfg = Args(**kw, bf16=True, adam_beta2=0.95)
''')
    assert f['codecfg.weight_precision.fp32'] == 1
    assert f['codecfg.compute_precision.bf16'] == 1
    assert f['codecfg.min_lr_ratio'] == 0.1
    assert f['codecfg.beta2'] == 0.95
    assert f['codecfg.scheduler.cosine_with_min_lr'] == 1
    assert f['codecfg.weight_decay'] == 0.01


def test_static_function_argument_return_and_training_settings():
    f, _ = extract('''
from transformers import TrainingArguments
def opts(rate, bs=4):
    return dict(learning_rate=rate, per_device_train_batch_size=bs)
config = TrainingArguments(**opts(0.0001, bs=16))
''')
    assert f['codecfg.learning_rate'] == 0.0001
    assert f['codecfg.batch_size'] == 16


def test_optimizer_and_lora_configurations():
    f, _ = extract('''
from torch.optim import AdamW as Opt
from peft import LoraConfig
opt = Opt(parameters, lr=2e-5, betas=(0.9, 0.95), eps=1e-8, fused=True)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=['q_proj', 'v_proj'])
''')
    assert f['codecfg.optimizer.adamw'] == 1
    assert f['codecfg.beta2'] == 0.95
    assert f['codecfg.epsilon'] == 1e-8
    assert f['codecfg.optimizer_fused'] == 1
    assert f['codecfg.lora_present'] == 1
    assert f['codecfg.lora_alpha_per_rank'] == 2
    assert f['codecfg.lora_target.q_proj'] == 1
    assert f['codecfg.lora_target.k_proj'] == 0


def test_sampling_literals_without_token_identity():
    f, _ = extract('''
from vllm import SamplingParams
cfg = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9, top_k=40,
                     repetition_penalty=1.05, stop_token_ids=[42, 999])
''')
    assert f['codecfg.max_new_tokens'] == 512
    assert f['codecfg.eos_override'] == 1
    assert 999 not in f.values()


def test_dynamic_expression_does_not_backfill_stale_plan():
    f, audit = extract('''
import os
from transformers import TrainingArguments
c = TrainingArguments(learning_rate=float(os.environ['LR']))
''', hp={'lr': 0.1})
    assert f['codecfg.learning_rate'] is None
    assert 'learning_rate' in audit['unknown_fields']


def test_conflicting_constructors_are_unknown_not_averaged():
    f, audit = extract('''
from transformers import TrainingArguments
a = TrainingArguments(learning_rate=1e-5)
b = TrainingArguments(learning_rate=2e-5)
''')
    assert f['codecfg.learning_rate'] is None
    assert audit['conflicting_fields'] == ['learning_rate']


def test_unresolved_branch_poisons_affected_settings_only():
    f, audit = extract('''
from transformers import TrainingArguments
from peft import LoraConfig
lr = 1e-5
if unknown_runtime_condition:
    lr = 2e-5
    lora = LoraConfig(r=16)
    config = TrainingArguments(eos_token_id=3)
c = TrainingArguments(learning_rate=lr, num_train_epochs=3)
''')
    assert f['codecfg.learning_rate'] is None
    assert f['codecfg.lora_present'] is None
    assert f['codecfg.eos_override'] is None
    assert f['codecfg.epochs'] == 3
    assert audit['unsupported_control_flow'] == 1


def test_dead_function_and_false_branch_ignored():
    f, _ = extract('''
from transformers import TrainingArguments
def never_called():
    c = TrainingArguments(learning_rate=99)
if False:
    c = TrainingArguments(learning_rate=77)
c = TrainingArguments(learning_rate=1e-5)
''')
    assert f['codecfg.learning_rate'] == 1e-5


def test_outcome_and_identity_invariance():
    raw = source('''
from transformers import TrainingArguments
score = 0.123
report = "accuracy 0.123 after 12 steps"
print(report, score)
c = TrainingArguments(learning_rate=1e-5, output_dir='checkpoint-exp-123')
''')
    f, _ = extract_config_features(raw)
    altered = copy.deepcopy(raw)
    altered.update(example_id='private-id', official_accuracy=0.999, result={'score': 0.999})
    altered['plan']['result'] = {'score': 0.999}
    altered['plan']['setup']['method']['hyperparams']['other'] = 'accuracy 0.999'
    altered['code'][0]['content'] = altered['code'][0]['content'].replace('0.123', '0.999').replace('exp-123', 'private-identity')
    assert extract_config_features(altered)[0] == f
    assert set(f) == set(FEATURE_KEYS)
    assert 'private' not in json.dumps(f)
    assert not any(isinstance(v, str) for v in f.values())


@pytest.mark.parametrize('status', ['not_declared', 'missing', 'snapshot_only'])
def test_unavailable_source_uses_plan_but_not_snapshot(status):
    raw = source('from transformers import TrainingArguments\nc = TrainingArguments(learning_rate=0.9)', hp={'lr': 1e-5})
    raw['code'][0]['status'] = status
    f, audit = extract_config_features(raw)
    assert f['codecfg.learning_rate'] == 1e-5
    assert f['codecfg.entrypoint_available'] == 0
    assert audit['entrypoint_reason'] == 'missing_or_conflicting_reconstructed_entrypoint'


def test_actual_entrypoint_not_builder_or_other_training_script():
    raw = source('from transformers import TrainingArguments\nc = TrainingArguments(learning_rate=1e-5)')
    raw['code'].extend([
        {'script_path': '/task/build.py', 'role': 'data_builder_0', 'status': 'reconstructed', 'content': 'from transformers import TrainingArguments\nc = TrainingArguments(learning_rate=999)'},
        {'script_path': '/task/old_train.py', 'role': 'training', 'status': 'reconstructed', 'content': 'from transformers import TrainingArguments\nc = TrainingArguments(learning_rate=555)'},
    ])
    assert extract_config_features(raw)[0]['codecfg.learning_rate'] == 1e-5


def test_conflicting_command_scripts_refuse_source():
    raw = source('from transformers import TrainingArguments\nc = TrainingArguments(learning_rate=1e-5)')
    raw['plan']['setup']['command']['argv'][1] = 'other.py'
    f, audit = extract_config_features(raw)
    assert f['codecfg.learning_rate'] is None
    assert audit['entrypoint_reason'] == 'command_script_conflict'


def test_precision_description_whitelist_and_ambiguity():
    f, _ = extract('', hp={'precision': 'bf16 (weights loaded bf16, not upcast)'})
    assert f['codecfg.compute_precision.bf16'] == 1
    f, _ = extract('', hp={'precision': 'bf16 or fp16'})
    assert f['codecfg.compute_precision.bf16'] is None


def test_code_is_never_executed(tmp_path):
    target = tmp_path / 'never-created'
    f, _ = extract(f'''\nfrom pathlib import Path\nPath({str(target)!r}).write_text('bad')\nraise RuntimeError('bad')\n''')
    assert not target.exists()
    assert f['codecfg.entrypoint_available'] == 1


@pytest.mark.parametrize('expression', ['1/0', 'float("bad")', '2**100000000', 'min(3)', 'lambda: 0.9', 'float("inf")', 'float("nan")'])
def test_unsupported_or_nonfinite_expressions_are_missing(expression):
    f, _ = extract(f'from transformers import TrainingArguments\nc = TrainingArguments(learning_rate={expression})')
    assert f['codecfg.learning_rate'] is None


def test_repeat_option_uses_last_occurrence():
    f, _ = extract('''
import argparse
from transformers import TrainingArguments
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-5)
a = p.parse_args()
c = TrainingArguments(learning_rate=a.lr)
''', ['--lr=0.01', '--lr', '0.02'])
    assert f['codecfg.learning_rate'] == 0.02


def test_opaque_kwargs_do_not_certify_configuration():
    f, _ = extract('''
from transformers import TrainingArguments
c = TrainingArguments(**external_options)
''', hp={'lr': 1e-5})
    assert f['codecfg.learning_rate'] is None


def test_explicit_argparse_override_is_not_planned_command():
    f, _ = extract('''
import argparse
from transformers import TrainingArguments
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-5)
a = p.parse_args([])
c = TrainingArguments(learning_rate=a.lr)
''', ['--lr', '0.9'])
    assert f['codecfg.learning_rate'] is None


@pytest.mark.parametrize('expression, expected', [('None or 1e-5', 1e-5), ('2e-5 or 1e-5', 2e-5), ('True and 3e-5', 3e-5), ('False and unknown_call()', None)])
def test_boolean_operators_preserve_python_operands(expression, expected):
    f, _ = extract(f'from transformers import TrainingArguments\nc = TrainingArguments(learning_rate={expression})')
    assert f['codecfg.learning_rate'] == expected


def test_known_set_defaults_resolved():
    f, _ = extract('''
import argparse
from transformers import TrainingArguments
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-5)
p.set_defaults(lr=2e-5)
a = p.parse_args()
c = TrainingArguments(learning_rate=a.lr)
''')
    assert f['codecfg.learning_rate'] == 2e-5


def test_conditional_set_defaults_unknown_but_argv_can_override():
    code = '''
import argparse
from transformers import TrainingArguments
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-5)
if runtime_unknown:
    p.set_defaults(lr=2e-5)
a = p.parse_args()
c = TrainingArguments(learning_rate=a.lr)
'''
    assert extract(code)[0]['codecfg.learning_rate'] is None
    assert extract(code, ['--lr', '3e-5'])[0]['codecfg.learning_rate'] == 3e-5


def test_dynamic_set_defaults_unknown():
    f, _ = extract('''
import argparse
from transformers import TrainingArguments
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-5)
p.set_defaults(lr=external_value)
a = p.parse_args()
c = TrainingArguments(learning_rate=a.lr)
''')
    assert f['codecfg.learning_rate'] is None


@pytest.mark.parametrize('statement', ['return external_value', 'return', 'raise RuntimeError()'])
def test_early_termination_never_certifies_later_constructor(statement):
    f, _ = extract(f'''
from transformers import TrainingArguments
def main():
    {statement}
    c = TrainingArguments(learning_rate=0.8)
main()
''')
    assert f['codecfg.learning_rate'] is None


def test_conditional_return_makes_subsequent_config_unknown():
    f, _ = extract('''
from transformers import TrainingArguments
def main():
    if unknown_runtime_condition:
        return external_value
    c = TrainingArguments(learning_rate=0.8)
main()
''')
    assert f['codecfg.learning_rate'] is None


def test_custom_mask_is_only_syntax_evidence():
    f, audit = extract('''
def encode(example):
    labels = [-100] * prompt_length + answer_ids
    return labels
''')
    assert f['codecfg.label_mask_syntax'] == 1
    assert f['codecfg.completion_only'] is None
    assert audit['field_sources']['label_mask_syntax'] == ['syntax_evidence_not_execution']


def test_other_negative_constants_are_not_label_mask_evidence():
    f, _ = extract('output_score = -100\nlabels = [1, 2, 3]')
    assert f['codecfg.label_mask_syntax'] == 0


@pytest.mark.parametrize('argv', [
    ['bash', '-c', 'python train.py'],
    ['env', 'FLAG=1', 'python', 'train.py'],
    ['echo', 'train.py'],
    ['python', 'train.py', ';', 'python', 'other.py'],
    ['python', 'train.py', '--lr', '$(cat secret)'],
    ['python', 'train.py', '--lr', '`cat secret`'],
    ['python', 'train.py', '>', 'result'],
    ['python', '-c', 'exec(open("train.py").read())'],
    ['python', '-m', 'unverified.runner', 'train.py'],
])
def test_shell_or_unvalidated_launcher_refused(argv):
    raw = source('from transformers import TrainingArguments\nc = TrainingArguments(learning_rate=1e-5)')
    raw['plan']['setup']['command']['argv'] = argv
    f, audit = extract_config_features(raw)
    assert f['codecfg.entrypoint_available'] == 0
    assert f['codecfg.learning_rate'] is None
    assert audit['entrypoint_reason'] == 'unsupported_or_ambiguous_launcher'


@pytest.mark.parametrize('argv', [
    ['python3', '-u', 'train.py', '--lr', '2e-5'],
    ['/venv/bin/python3.13', 'train.py', '--lr', '2e-5'],
    ['torchrun', '--standalone', '--nproc_per_node=2', 'train.py', '--lr', '2e-5'],
    ['python', '-m', 'torch.distributed.run', '--nproc_per_node', '2', 'train.py', '--lr', '2e-5'],
    ['accelerate', 'launch', '--num_processes', '2', 'train.py', '--lr', '2e-5'],
])
def test_explicit_supported_launchers(argv):
    raw = source('''
import argparse
from transformers import TrainingArguments
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-5)
a = p.parse_args()
c = TrainingArguments(learning_rate=a.lr)
''')
    raw['plan']['setup']['command']['argv'] = argv
    f, _ = extract_config_features(raw)
    assert f['codecfg.entrypoint_available'] == 1
    assert f['codecfg.learning_rate'] == 2e-5


@pytest.mark.parametrize('condition', ['torch.runtime_dtype is None', 'torch.runtime_dtype == "bf16"', 'torch.runtime_dtype', 'not torch.runtime_dtype'])
def test_runtime_symbols_are_not_boolean_evidence(condition):
    f, _ = extract(f'''
import torch
from transformers import TrainingArguments
lr = 1e-5
if {condition}:
    lr = 2e-5
c = TrainingArguments(learning_rate=lr)
''')
    assert f['codecfg.learning_rate'] is None


@pytest.mark.parametrize('control', [
    'match external_value:\n    case 1:\n        lr = 2e-5',
    'try:\n    lr = 2e-5\nexcept* Exception:\n    lr = 3e-5',
])
def test_match_and_trystar_do_not_leave_stale_constants(control):
    f, _ = extract('from transformers import TrainingArguments\nlr = 1e-5\n' + control + '\nc = TrainingArguments(learning_rate=lr)')
    assert f['codecfg.learning_rate'] is None


def test_cli_end_of_options_respected():
    f, _ = extract('''
import argparse
from transformers import TrainingArguments
p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-5)
a = p.parse_args()
c = TrainingArguments(learning_rate=a.lr)
''', ['--', '--lr', '0.9'])
    assert f['codecfg.learning_rate'] == 1e-5
