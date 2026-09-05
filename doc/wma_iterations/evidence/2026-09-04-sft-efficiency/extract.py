import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path('/home/robtang_google_com/gangda_workspace/agentic-world-model')
OUT = Path(__file__).resolve().parent
SOURCE = OUT / 'input-complete-cells.json'


def get(d, path, default=None):
    for key in path.split('.'):
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    return default if d is None else d


def cohort(c):
    b = c['batch']
    if '-r01-' in b:
        if re.search(r'r01-(wma|ctl)-x8-v1$', b):
            return 'R1 core', 'v0.2' if c['cell'].startswith('w') else 'Control'
        name = 'R1 ' + ('B-v2' if '-b-x8-v2' in b else 'B-v3' if '-b-x8-v3' in b else 'C-v2')
        return name, 'v0.2' if c['cell'].startswith('w') else 'Control'
    if '-v02-baseline-' in b:
        return 'R2 blocking', 'v0.2'
    if '-ctl-' in b:
        return 'R2 blocking', 'Control'
    variant = next((label for token, label in [('-ab-', 'A+B'), ('-a-', 'A'), ('-c-', 'C'), ('-d-', 'D')] if token in b), '?')
    return 'R2 blocking', variant


def load_overrides():
    overrides = {}
    for p in sorted(OUT.glob('*overrides.json')):
        value = json.loads(p.read_text())
        if isinstance(value, dict):
            value = value.get('overrides', value.get('cards', value.get('rows', [])))
        if not isinstance(value, list):
            raise ValueError(f'Unexpected override format {p}')
        for row in value:
            overrides[(row['cell'], row['card'])] = row
    return overrides


def summarize(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values) > 1 else None)


def main():
    overrides = load_overrides()
    cells, events, measurements = [], [], []
    for source in json.loads(SOURCE.read_text()):
        c = dict(source)
        c['cohort'], c['variant'] = cohort(c)
        c['arm'] = 'WMA' if c['cell'].startswith('w') else 'Control'
        c['manual_heldout_exposure'] = c['cell'] == 'w09r03'
        c['stages'] = []
        c['events'] = []
        completed = strict = rft = aborted = 0
        card_wall = 0.0
        best = {}
        local_path = Path(c['result_dir']) / 'task/memory/cards'
        for path in sorted(local_path.glob('exp-*.yaml')):
            card = yaml.safe_load(path.read_text())
            cid = path.stem
            family = get(card, 'setup.method.family')
            execution = get(card, 'result.execution')
            train_card = family in ('sft', 'rft')
            base_fits = int(train_card and execution == 'completed')
            ov = overrides.get((c['cell'], cid), {})
            fits = int(ov.get('completed_fits', base_fits))
            if fits < 0:
                raise ValueError('Negative fits')
            full_aborts = ov.get('aborted_full_intent')
            if full_aborts is None:
                full_aborts = int(train_card and execution in ('failed', 'killed') and fits == 0)
            aborted += full_aborts
            completed += fits
            strict += fits if family == 'sft' else 0
            rft += fits if family == 'rft' else 0
            if train_card:
                card_wall += float(get(card, 'result.wall_h', 0))
            protocol = get(card, 'evaluation.protocol', {})
            ev = dict(cell=c['cell'], batch=c['batch'], cohort=c['cohort'], variant=c['variant'], arm=c['arm'],
                      card=cid, family=family, execution=execution, completed_fits=fits,
                      completed_sft=strict, completed_rft=rft, completed_supervised=completed,
                      aborted_full_intent=full_aborts, cumulative_aborted_full_intent=aborted,
                      self_sample_mix=ov.get('self_sample_mix'), elapsed_h=get(card, 'situation.elapsed_h'),
                      card_wall_h=get(card, 'result.wall_h'), cumulative_training_card_wall_h=round(card_wall, 4),
                      training_steps=get(card, 'result.training_summary.steps'),
                      epochs=get(card, 'setup.method.hyperparams.epochs'),
                      rows=sum(float(d.get('n_examples') or 0) for d in get(card, 'setup.data', []) if isinstance(d, dict) and isinstance(d.get('n_examples'), (int, float))),
                      parent=get(card, 'setup.parent_checkpoint.origin'),
                      decision=get(card, 'conclusion.decision'),
                      planned_incumbent=get(card, 'situation.incumbent'),
                      protocol=protocol, source=str(path),
                      audit_note=ov.get('note', ''), audit_evidence=ov.get('evidence', []),
                      count_audited=bool(ov), measurements=[])
            for j, m in enumerate(get(card, 'result.measurements', [])):
                if not isinstance(m, dict) or m.get('metric') != 'accuracy' or not isinstance(m.get('value'), (int, float)):
                    continue
                n, score = m.get('n'), m['value']
                if not isinstance(n, (int, float)) or not 0 <= score <= 1:
                    continue
                n = int(n)
                mp = str(m.get('path') or '')
                internal = bool(re.search(r'internal|dev_clean|dev_holdout|local_dev|dev250|dev300|watch', mp, re.I))
                official = not internal and any(t in str(protocol).lower() for t in ('official', 'evaluate.py', 'inspect_evals', 'gsm8k test', 'gsm8k/test'))
                meas = dict(cell=c['cell'], card=cid, measurement_index=j, n=n, score=score,
                            completed_supervised=completed, completed_sft=strict,
                            family=family, evaluation_scope='declared_official' if official else 'local_or_uncertain',
                            path=mp, source=str(path), protocol=protocol, final_judge_score=False)
                if c['cell'] == 'c01r08' and cid == 'exp-08' and j in (0, 1):
                    meas['aggregation'] = 'mean_of_three_evaluations'
                    meas['verified_mean'] = 0.77078594895 if j == 0 else 0.7705332322466515
                    meas['aggregation_evidence'] = str(Path(c['result_dir']) / 'solve_parsed.txt') + ':10110'
                ev['measurements'].append(meas)
                measurements.append(meas)
                if official:
                    best[n] = max(best.get(n, -1), score)
            ev['best_observed'] = dict(best)
            c['events'].append(ev)
            events.append(ev)
        c.update(completed_supervised=completed, completed_sft=strict, completed_rft=rft,
                 aborted_full_intent_known=aborted, training_card_wall_h=round(card_wall, 4),
                 count_audited_cards=sum(e['count_audited'] for e in c['events']),
                 completed_training_cards=sum(e['family'] in ('sft', 'rft') and e['execution'] == 'completed' for e in c['events']))
        for axis in ('completed_supervised', 'completed_sft'):
            stages = []
            for k in range(c[axis] + 1):
                prior = [e for e in c['events'] if e[axis] <= k]
                active = [e for e in c['events'] if e[axis] == k]
                last = prior[-1] if prior else None
                stages.append(dict(k=k, best_observed=last['best_observed'] if last else {},
                                   event_cards=[e['card'] for e in active],
                                   fresh_ns=sorted({m['n'] for e in active for m in e['measurements'] if m['evaluation_scope'] == 'declared_official'}),
                                   card_boundary_only=True))
            c[axis + '_stages'] = stages
        c['trace_path'] = str(Path(c['result_dir']) / 'solve_parsed.txt')
        cells.append(c)
        measurements.append(dict(cell=c['cell'], card='FINAL', n=1319, score=c['accuracy'],
                                 completed_supervised=completed, completed_sft=strict, family='official_final',
                                 evaluation_scope='official_final', path=str(Path(c['result_dir']) / 'metrics.json'),
                                 final_judge_score=True))

    groups = defaultdict(list)
    for c in cells:
        groups[(c['cohort'], c['variant'])].append(c)
    summaries = []
    curves = []
    for (co, variant), cc in groups.items():
        summaries.append(dict(cohort=co, variant=variant, cells=[c['cell'] for c in cc],
                              final_score=summarize([c['accuracy'] for c in cc]),
                              completed_supervised=summarize([c['completed_supervised'] for c in cc]),
                              completed_sft=summarize([c['completed_sft'] for c in cc]),
                              completed_rft=summarize([c['completed_rft'] for c in cc]),
                              aborted_full_intent_known=sum(c['aborted_full_intent_known'] for c in cc),
                              training_card_wall_h=summarize([c['training_card_wall_h'] for c in cc])))
        for axis in ('completed_supervised', 'completed_sft'):
            for n in (150, 500, 1319):
                for k in range(1, max(c[axis] for c in cc) + 1):
                    eligible = [c for c in cc if c[axis] >= k]
                    points = [(c['cell'], c[axis + '_stages'][k]['best_observed'].get(n)) for c in eligible]
                    points = [(cell, score) for cell, score in points if score is not None]
                    curves.append(dict(cohort=co, variant=variant, axis=axis, eval_n=n, k=k,
                                       eligible_cells=len(eligible), cells=[x[0] for x in points],
                                       scores=[x[1] for x in points], **summarize([x[1] for x in points])))
    thresholds = []
    for c in cells:
        for axis in ('completed_supervised', 'completed_sft'):
            for n in (150, 500, 1319):
                for threshold in (0.70, 0.75, 0.80):
                    hit = next((s for s in c[axis + '_stages'] if s['best_observed'].get(n, -1) >= threshold), None)
                    thresholds.append(dict(cell=c['cell'], cohort=c['cohort'], variant=c['variant'], axis=axis,
                                           n=n, threshold=threshold, k=hit['k'] if hit else None,
                                           observation='hit' if hit else ('no_evaluation_at_this_n' if not any(n in s['best_observed'] for s in c[axis + '_stages']) else 'not_observed'), max_fits=c[axis]))
    metadata = dict(source=str(SOURCE), snapshot='2026-09-04 05:52:48 UTC', cells=len(cells),
                    definition='Completed substantive supervised optimizer schedules, including final-save failures; explicit smokes and unfinished attempts excluded and reported separately. Card family retained for SFT-only sensitivity.',
                    chronology='Card completion order. An evaluation is charged all completed fits up to its observation card, never retroactively assigned to the model-creation card. Within-card repeated fits are charged before the stored result.',
                    curves='Highest declared-official accuracy observed so far, separately by evaluation n. Different candidate decoders/concurrency are not normalized. No extrapolation after a trajectory ends; per-point sample counts reported.',
                    known_manual_exposure_cell='w09r03', historical_count_scope='R1 counts are confirmed lower bounds after complete card census and targeted raw-trace retry checks; not an exhaustive every-launch audit.')
    payload = dict(metadata=metadata, cells=cells, events=events, measurements=measurements,
                   summaries=summaries, curves=curves, thresholds=thresholds)
    (OUT / 'efficiency.json').write_text(json.dumps(payload, indent=2, default=str))
    with (OUT / 'trajectories.csv').open('w') as f:
        fields = ['cell','cohort','variant','accuracy','completed_supervised','completed_sft','completed_rft','completed_training_cards','aborted_full_intent_known','training_card_wall_h','manual_heldout_exposure','result_dir','manifest','spec']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows({k:c.get(k) for k in fields} for c in cells)
    with (OUT / 'measurements.csv').open('w') as f:
        fields = ['cell','card','completed_supervised','completed_sft','family','n','score','evaluation_scope','final_judge_score','path']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows({k:m.get(k) for k in fields} for m in measurements)
    event_fields = ['cell', 'card', 'family', 'execution', 'completed_fits', 'completed_sft', 'completed_rft', 'completed_supervised', 'aborted_full_intent', 'self_sample_mix', 'elapsed_h', 'card_wall_h', 'cumulative_training_card_wall_h', 'training_steps', 'epochs', 'rows', 'decision', 'count_audited', 'audit_note', 'source']
    with (OUT / 'events.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=event_fields); w.writeheader(); w.writerows({k:e.get(k) for k in event_fields} for e in events)
    with (OUT / 'thresholds.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(thresholds[0])); w.writeheader(); w.writerows(thresholds)
    print('cells',len(cells),'events',len(events),'measurements',len(measurements),'overrides',len(overrides))
    for s in summaries:
        print(s['cohort'],s['variant'],'n=',s['final_score']['n'],'score=',round(s['final_score']['mean']*100,2),
              'fullfits=',round(s['completed_supervised']['mean'],3),'SFT=',round(s['completed_sft']['mean'],3),
              'RFT=',round(s['completed_rft']['mean'],3),'partial>=',s['aborted_full_intent_known'])
    print('CORE n150 CURVES')
    for r in curves:
        if r['cohort'] in ('R1 core','R2 blocking') and r['variant'] in ('Control','v0.2') and r['axis']=='completed_supervised' and r['eval_n']==150:
            print(r)


if __name__ == '__main__':
    main()
