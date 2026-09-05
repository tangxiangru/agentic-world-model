#!/usr/bin/env python3
"""CPU-only structural Inspect JSON audit; output directory must be explicit."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator')
BUNDLE = ROOT / 'results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
OUT = args.output_dir.resolve()
if not OUT.is_dir() or any(OUT.iterdir()):
    parser.error('--output-dir must be an existing empty directory')
RECEIPT = BUNDLE / 'formal-2026-09-02T210446.182614+0000.json'

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def filehash(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def key(sample):
    # Retain ID type and epoch, and never infer correspondence from array order.
    return canonical([sample['id'], sample.get('epoch')])

def prompt(messages):
    return [{k: m[k] for k in ('role', 'content', 'tool_calls', 'tool_call_id') if k in m}
            for m in messages]

def inspect_file(path):
    with path.open() as f:
        d = json.load(f)
    if not isinstance(d, dict) or not {'eval', 'samples', 'results', 'status'} <= d.keys():
        return None
    e, res, samples = d['eval'], d['results'], d['samples']
    rows = {}
    duplicates, sample_errors, invalid_scores = [], [], []
    scorekeys, scores, epochs = collections.Counter(), collections.Counter(), collections.Counter()
    event_configs = collections.Counter()
    for s in samples:
        k = key(s)
        if k in rows:
            duplicates.append(k)
        scorekeys.update(s.get('scores', {}).keys())
        value = s.get('scores', {}).get('match', {}).get('value')
        scores[canonical(value)] += 1
        epochs[canonical(s.get('epoch'))] += 1
        if value not in ('C', 'I'):
            invalid_scores.append(k)
        if s.get('error'):
            sample_errors.append(k)
        ev = [x for x in s.get('events', []) if x.get('event') == 'model']
        for x in ev:
            event_configs[canonical(x.get('config'))] += 1
        rows[k] = {
            'id': s['id'], 'epoch': s.get('epoch'), 'correct': value == 'C', 'value': value,
            'input_sha256': digest(s.get('input')), 'target_sha256': digest(s.get('target')),
            'prompt_sha256': digest([prompt(x.get('input', [])) for x in ev]),
            'model_event_n': len(ev),
        }
    assert not duplicates, (path, duplicates)
    declared = e.get('dataset', {}).get('sample_ids', [])
    declared_keys = {canonical([i, 1]) for i in declared}
    actual_keys = set(rows)
    result_scores = res.get('scores', [])
    metrics = {s['name']: s.get('metrics') for s in result_scores}
    result_accuracy = next((s.get('metrics', {}).get('accuracy', {}).get('value')
                            for s in result_scores if s.get('name') == 'match'), None)
    scored_n = sum(v in ('C', 'I') for v in [s.get('scores', {}).get('match', {}).get('value') for s in samples])
    correct_n = sum(r['correct'] for r in rows.values())
    reported_scored = next((s.get('scored_samples') for s in result_scores if s.get('name') == 'match'), None)
    n = len(samples)
    issues = []
    if d['status'] != 'success': issues.append('status_not_success')
    if res.get('total_samples') != n: issues.append('total_samples_mismatch')
    if res.get('completed_samples') != n: issues.append('completed_samples_mismatch')
    if reported_scored != scored_n: issues.append('scored_samples_mismatch')
    if result_accuracy is None or abs(result_accuracy - correct_n / scored_n) > 1e-12:
        issues.append('accuracy_mismatch')
    if actual_keys != declared_keys: issues.append('declared_actual_id_epoch_mismatch')
    if invalid_scores: issues.append('invalid_binary_scores')
    if sample_errors: issues.append('sample_errors')
    protocol = {
        'task': e.get('task'), 'task_version': e.get('task_version'),
        'task_args': e.get('task_args'), 'scorers': e.get('scorers'),
        'model_generate_config': e.get('model_generate_config'), 'model_args': e.get('model_args'),
        'config': e.get('config'), 'packages': e.get('packages'),
        'plan_steps_sha256': digest(d.get('plan', {}).get('steps')),
    }
    return {
        'path': str(path), 'sha256': filehash(path), 'bytes': path.stat().st_size,
        'status': d['status'], 'model': e.get('model'), 'created': e.get('created'),
        'started_at': d.get('stats', {}).get('started_at'),
        'completed_at': d.get('stats', {}).get('completed_at'),
        'dataset': {k: e.get('dataset', {}).get(k) for k in ('name', 'samples', 'shuffled')},
        'declared_sample_ids': declared, 'declared_id_n': len(declared),
        'actual_sample_n': n, 'unique_id_epoch_n': len(rows), 'epochs': dict(epochs),
        'total_samples': res.get('total_samples'), 'completed_samples': res.get('completed_samples'),
        'reported_scored_n': reported_scored, 'actual_scored_n': scored_n,
        'correct_n': correct_n, 'accuracy': result_accuracy,
        'score_keys': dict(scorekeys), 'score_values': dict(scores), 'metrics': metrics,
        'missing_declared_keys': sorted(declared_keys - actual_keys),
        'extra_actual_keys': sorted(actual_keys - declared_keys),
        'duplicate_keys': duplicates, 'sample_errors': sample_errors,
        'issues': issues, 'protocol': protocol, 'model_event_configs': dict(event_configs),
        'rows': rows,
    }

def pair(a, b, a_keys=None):
    ar, br = a['rows'], b['rows']
    ak = set(ar) if a_keys is None else set(a_keys)
    bk = set(br)
    common = ak & bk
    counts = collections.Counter()
    diffs = collections.defaultdict(list)
    for k in sorted(common):
        x, y = ar[k], br[k]
        name = ('both_correct' if x['correct'] and y['correct'] else
                'a_only' if x['correct'] else 'b_only' if y['correct'] else 'both_incorrect')
        counts[name] += 1
        if name in ('a_only', 'b_only'): diffs[name].append(k)
        for field in ('input_sha256', 'target_sha256', 'prompt_sha256'):
            if x[field] != y[field]: diffs[field + '_mismatch'].append(k)
    ao, bo = counts['a_only'], counts['b_only']
    n = ao + bo
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(ao, bo) + 1)) / (2 ** n)) if n else 1.0
    return {
        'a_path': a['path'], 'b_path': b['path'], 'a_model': a['model'], 'b_model': b['model'],
        'a_available_n': len(ak), 'b_available_n': len(bk), 'paired_n': len(common),
        'a_not_in_b': sorted(ak - bk), 'b_not_in_a': sorted(bk - ak),
        'both_correct': counts['both_correct'], 'a_only': ao, 'b_only': bo,
        'both_incorrect': counts['both_incorrect'], 'net_a_minus_b': ao - bo,
        'a_correct_paired': counts['both_correct'] + ao,
        'b_correct_paired': counts['both_correct'] + bo,
        'mcnemar_exact_two_sided_p': p,
        'mismatch_counts': {k: len(v) for k, v in diffs.items() if k.endswith('_mismatch')},
        'discordant_and_mismatched_keys': dict(diffs),
        'protocol_differences': {k: [a['protocol'].get(k), b['protocol'].get(k)]
                                 for k in a['protocol'] if a['protocol'].get(k) != b['protocol'].get(k)},
    }

receipt = json.loads(RECEIPT.read_text())
output = {'script': str(Path(__file__).resolve()), 'script_sha256': filehash(Path(__file__)),
          'receipt': {'path': str(RECEIPT), 'sha256': filehash(RECEIPT)}, 'cells': {}}
for cell in ('c01s04', 'c01s07'):
    statuspath = BUNDLE / cell / 'status.json'
    status = json.loads(statuspath.read_text())
    raw = Path(status['result_dir'])
    jobs = [j for j in receipt['jobs'] if j['cell_id'] == cell]
    assert len(jobs) == 1 and jobs[0]['job_id'] == status['job_id']
    logs = []
    skipped = []
    for path in sorted((raw / 'task/logs').glob('*.json')):
        obj = inspect_file(path)
        if obj is None: skipped.append(str(path))
        else: logs.append(obj)
    ancillary = {}
    for rel in ('evaluate.py', 'templates/gemma3.jinja', 'run_eval.sh', 'set_gen_config.py',
                'finalize.py', 'prep_ckpt.py', 'merge_lora.py'):
        p = raw / 'task' / rel
        if p.is_file(): ancillary[rel] = {'path': str(p), 'sha256': filehash(p)}
    finalgc = raw / 'final_model/generation_config.json'
    if finalgc.is_file():
        ancillary['final_model/generation_config.json'] = {
            'path': str(finalgc), 'sha256': filehash(finalgc), 'content': json.loads(finalgc.read_text())}
    trace = BUNDLE / cell / 'solve_parsed.txt.gz'
    ancillary['solve_parsed.txt.gz'] = {'path': str(trace), 'sha256': filehash(trace)}
    output['cells'][cell] = {
        'status_path': str(statuspath), 'status_sha256': filehash(statuspath), 'status': status,
        'receipt_jobs': jobs, 'ancillary_files': ancillary, 'logs': logs, 'non_inspect_json': skipped,
        'log_n': len(logs), 'n_histogram': dict(collections.Counter(x['actual_sample_n'] for x in logs)),
    }

def select(cell, model, n):
    found = [x for x in output['cells'][cell]['logs'] if x['model'] == model and x['actual_sample_n'] == n]
    if len(found) != 1:
        raise ValueError((cell, model, n, [(x['path'],x['model']) for x in found]))
    return found[0]

pairs = {}
for am, bm in [('eval_b70', 'eval_b160'), ('eval_b70', 'eval_b105'),
               ('eval_b70', 'eval_soup1'), ('eval_b70', 'eval_b70_rp'), ('eval_b105', 'eval_b70_rp')]:
    pairs['c01s04_' + am + '_vs_' + bm] = pair(
        select('c01s04', 'vllm/runs/' + am, 1319), select('c01s04', 'vllm/runs/' + bm, 1319))
for am, bm in [('g2_200', 'rft_v1'), ('g2_200', 'g3_150'), ('g3_150', 'g3_100')]:
    pairs['c01s07_' + am + '_vs_' + bm] = pair(
        select('c01s07', 'vllm/ckpt/' + am, 500), select('c01s07', 'vllm/ckpt/' + bm, 500))
g2 = select('c01s07', 'vllm/ckpt/g2_200', 500)
final = select('c01s07', 'vllm/final_model', 1000)
pairs['c01s07_g2_200_vs_final1000_shared500'] = pair(g2, final)
prefix = {canonical([i, 1]) for i in final['declared_sample_ids'][:500]}
suffix = {canonical([i, 1]) for i in final['declared_sample_ids'][500:]}
output['c01s07_prefix_check'] = {
    'first500_ids_equal_g2_ordered': final['declared_sample_ids'][:500] == g2['declared_sample_ids'],
    'first500_keys_equal_g2': prefix == set(g2['rows']),
    'first500_n': len(prefix), 'first500_correct': sum(final['rows'][k]['correct'] for k in prefix),
    'last500_n': len(suffix), 'last500_correct': sum(final['rows'][k]['correct'] for k in suffix),
}
output['pairs'] = pairs
assert {c: d['log_n'] for c, d in output['cells'].items()} == {'c01s04': 19, 'c01s07': 15}
assert all(not log['issues'] for cell in output['cells'].values() for log in cell['logs'])
for name, result in pairs.items():
    assert result['paired_n'] == result['a_available_n'], (name, 'incomplete A-side join')
    assert result['paired_n'] == sum(result[k] for k in ('both_correct', 'a_only', 'b_only', 'both_incorrect'))
    assert not result['mismatch_counts'], (name, 'paired input/target/prompt mismatch')
dest = OUT / 'structural-audit.json'
dest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
inventory_lines = ['cell\tcreated\tmodel\tstatus\tactual_n\tcompleted_n\tscored_n\tcorrect\taccuracy\tmax_tokens\tmax_connections\tgpu_memory_utilization\tsha256\tpath']
for cell, data in output['cells'].items():
    for x in data['logs']:
        g = x['protocol']['model_generate_config']; ma = x['protocol']['model_args']
        inventory_lines.append('\t'.join(map(str, [cell, x['created'], x['model'], x['status'],
            x['actual_sample_n'], x['completed_samples'], x['actual_scored_n'], x['correct_n'], x['accuracy'],
            g.get('max_tokens'), g.get('max_connections'), ma.get('gpu_memory_utilization'), x['sha256'], x['path']])))
(OUT / 'inventory.tsv').write_text('\n'.join(inventory_lines) + '\n')
pair_lines = ['comparison\tn\tboth_correct\ta_only\tb_only\tboth_incorrect\tnet_a_minus_b\texact_two_sided_p\tinput_target_prompt_mismatches']
for name, p in pairs.items():
    pair_lines.append('\t'.join(map(str, [name, p['paired_n'], p['both_correct'], p['a_only'], p['b_only'],
        p['both_incorrect'], p['net_a_minus_b'], p['mcnemar_exact_two_sided_p'], p['mismatch_counts']])))
(OUT / 'pairs.tsv').write_text('\n'.join(pair_lines) + '\n')
for cell, data in output['cells'].items():
    print(cell, 'logs=', data['log_n'], 'n_histogram=', data['n_histogram'])
    for x in data['logs']:
        g = x['protocol']['model_generate_config']; ma = x['protocol']['model_args']
        print(x['created'], x['model'], x['actual_sample_n'], x['correct_n'], x['accuracy'],
              'max_tokens=', g.get('max_tokens'), 'concurrency=', g.get('max_connections'),
              'memory=', ma.get('gpu_memory_utilization'), 'issues=', x['issues'])
print('OUTPUT', dest)
for name, p in pairs.items():
    print('PAIR', name, 'n=', p['paired_n'], 'both=', p['both_correct'],
          'A-only=', p['a_only'], 'B-only=', p['b_only'], 'neither=', p['both_incorrect'],
          'net=', p['net_a_minus_b'], 'p=', p['mcnemar_exact_two_sided_p'],
          'mismatches=', p['mismatch_counts'], 'protocol_changed=', list(p['protocol_differences']))
print('PREFIX', output['c01s07_prefix_check'])
