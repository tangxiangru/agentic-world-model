"""Read-only exact-frozen-code replay. No models, training, eval, or scheduler calls.

Only fixture files made by apply_patch are read. Frozen source is loaded directly
from git objects in memory, without checkout or mutation of the worktree.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import yaml

REPO = Path('/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator')
FIXTURE = Path(__file__).parent
SHA = '833291799b37991f0ca0e22a5d6a42679916b167'
CHECK = 'parent_generation_config_valid'


def frozen(path):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', f'{SHA}:{path}'])


# preflight's sole awm.paths use is catalogue lookup; all calls supply pitfalls=[].
# Keep even that unused fallback deterministic, without importing current code.
awm = types.ModuleType('awm')
awm.paths = types.SimpleNamespace(REPO_ROOT=REPO)
sys.modules['awm'] = awm
package = types.ModuleType('frozen_d')
package.__path__ = []
sys.modules['frozen_d'] = package
loaded = {}
for name in ('schema', 'preflight'):
    path = f'awm/exp_protocol/{name}.py'
    source = frozen(path)
    module = types.ModuleType(f'frozen_d.{name}')
    module.__package__ = 'frozen_d'
    module.__file__ = f'git:{SHA}:{path}'
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    loaded[name] = module
    print('SOURCE', json.dumps({'path': path, 'sha256': hashlib.sha256(source).hexdigest()}))
schema, preflight = loaded['schema'], loaded['preflight']
assert preflight.GREEDY_INCOMPATIBLE == {
    'temperature': 1.0, 'top_k': 50, 'top_p': 1.0, 'typical_p': 1.0, 'min_p': None,
}
print('FROZEN_SHA', SHA)
print('GREEDY_INCOMPATIBLE', json.dumps(preflight.GREEDY_INCOMPATIBLE))


def direct(card, parent):
    replay = copy.deepcopy(card)
    replay['setup']['parent_checkpoint']['path'] = str(FIXTURE / parent)
    result = preflight.parent_generation_config_valid(preflight.Context(replay, None, {}))
    return {'status': result.status, 'detail': result.detail}


guard = REPO / 'results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1'
baseline = REPO / 'results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1'
cases = [
    ('g01r03_exp08_eval_only', guard / 'g01r03/task/memory/cards/exp-08.yaml', 'unsafe-temperature'),
    ('p00s02_exp07_eval_only', baseline / 'p00s02/task/memory/cards/exp-07.yaml', 'unsafe-p00s02'),
    ('p00s01_exp06_rft_save_failure', baseline / 'p00s01/task/memory/cards/exp-06.yaml', 'unsafe-temperature'),
    ('g01r03_exp07_merge_save_failure_then_repair', guard / 'g01r03/task/memory/cards/exp-07.yaml', 'unsafe-temperature'),
    ('p00s02_exp06_sft_repaired_in_code', baseline / 'p00s02/task/memory/cards/exp-06.yaml', 'unsafe-p00s02'),
]
for label, card_path, parent in cases:
    card = yaml.safe_load(card_path.read_text())
    validation = schema.validate_plan(card)
    assert validation.ok, (label, validation.render())
    result = direct(card, parent)
    assert result['status'] == 'fail', (label, result)
    valid = direct(card, 'valid-greedy')
    assert valid['status'] == 'pass', (label, valid)
    print('REAL_CARD_REPLAY', json.dumps({
        'case': label, 'family': card['setup']['method']['family'],
        'schema_ok': validation.ok, 'original_parent': card['setup']['parent_checkpoint']['path'],
        'argv': card['setup']['command']['argv'],
        'unsafe_parent_result': result, 'valid_parent_status': valid['status'],
        'only_card_change': 'setup.parent_checkpoint.path remapped to CPU JSON fixture',
    }))

# One schema-valid complete preflight fixture: exactly one failure in every family.
base = yaml.safe_load(frozen('skills/exp_protocol/example-card.yaml'))
base['setup']['data'] = [{'path': str(FIXTURE / 'data.jsonl'), 'source': 'local CPU fixture', 'n_examples': 1}]
base['setup']['command'] = {'argv': ['python', 'reproduce.py'], 'cwd': str(FIXTURE), 'script': str(FIXTURE / 'reproduce.py')}
base['setup']['output_dir'] = str(FIXTURE)
base['evaluation']['comparator']['path'] = str(FIXTURE / 'comparator.json')
for family in schema.METHOD_FAMILIES:
    card = copy.deepcopy(base)
    card['setup']['method']['family'] = family
    card['setup']['parent_checkpoint']['path'] = str(FIXTURE / 'unsafe-temperature')
    assert schema.validate_plan(card).ok
    report = preflight.run_preflight(card, pitfalls=[])
    failures = [r['check'] for r in report['results'] if r['status'] == 'fail']
    assert failures == [CHECK], (family, report)
    print('FULL_PREFLIGHT_FAMILY', json.dumps({'family': family, 'summary': report['summary'], 'failures': failures}))

for parent, expected in [('unsafe-temperature', 'fail'), ('unsafe-p00s02', 'fail'),
                         ('valid-greedy', 'pass'), ('stock', 'pass'), ('bare', 'skip')]:
    result = direct(base, parent)
    assert result['status'] == expected
    print('CONFIG_CONTROL', json.dumps({'parent': parent, **result}))

# Static inspection only: parse complete evaluator bodies; never execute them.
for task in (guard / 'g01r03/task', baseline / 'p00s02/task'):
    source = (task / 'evaluate.py').read_text()
    tree = ast.parse(source)
    calls = [(ast.unparse(n.func), n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = [(call, line) for call, line in calls if call.endswith(('.save_pretrained', '.save_model', '.train'))]
    assert not forbidden
    assert ('inspect_eval', 74) in calls
    print('EVALUATOR_STATIC', json.dumps({'path': str(task / 'evaluate.py'), 'save_or_train_calls': forbidden, 'inspect_eval_line': 74}))

print('PASS: 5 real-card replays; 8 schema-valid family preflights; 5 config controls; 2 static evaluator checks. No training/evaluation/model/Slurm calls.')
