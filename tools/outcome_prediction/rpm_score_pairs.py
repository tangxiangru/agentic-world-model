"""Score frozen-judge arms, learned/kNN rankers and heuristics on the 89 RPM-style pairs (run: rpm_knn.py first)."""
import json, statistics, random, glob
from pathlib import Path
from collections import defaultdict
J=Path('data/analysis/rpm/judge'); JC=Path('data/analysis/rpm/judge_code')
labels={l['id']:l for l in json.load(open(J/'hidden_labels.json'))}
ex={json.loads(l)['example_id']:json.loads(l) for l in open('data/analysis/outcome_prediction/examples.jsonl')}
def load_judge(d):
    out={}
    for p in d.glob('pair-*.json'):
        x=json.load(open(p))
        if x.get('valid'): out[x['id']]=x['p_a']
    return out
methods={}
methods['judge v2: canonical recipes + scored history (inference-only)']=load_judge(J/'outputs/within_run')
methods['judge v3: + plan-time setup + launch scripts (inference-only)']=load_judge(JC/'outputs/code_within_run')
methods['judge v2 + labeled prior-run bank in context']=load_judge(J/'outputs/cross_run')
learned=defaultdict(dict)
for name in ('learned','learned_siblings','learned_contextual'):
    for l in open(f'data/analysis/rpm/{name}/pair_predictions.jsonl'):
        d=json.loads(l)
        for m,p in d['probabilities'].items():
            learned[f'{name}:{m}'][(d['a_id'],d['b_id'])]=p; learned[f'{name}:{m}'][(d['b_id'],d['a_id'])]=1-p
def bykey(key): return {pid:learned[key][(l['a_id'],l['b_id'])] for pid,l in labels.items() if (l['a_id'],l['b_id']) in learned[key]}
methods['learned: fixed logistic C=1 on prior runs (pre-specified)']=bykey('learned_siblings:fixed_recipe_numeric_logistic_C1')
methods['learned: recipe forest on prior runs (exploratory)']=bykey('learned_siblings:exploratory_recipe_numeric_forest')
S='data/analysis/rpm/judge_code/'
methods['kNN k=5 on prior-run bank (no fitting)']=json.load(open(S+'knn5.json'))
def cmp(x,y): return 0.5 if x==y else (1.0 if x>y else 0.0)
methods['heuristic: later-registered sibling']={pid:cmp(ex[l['a_id']]['first_submitted_at'],ex[l['b_id']]['first_submitted_at']) for pid,l in labels.items()}
methods['heuristic: more training examples']={pid:cmp(sum(d.get('n_examples',0) for d in ex[l['a_id']]['recipe']['data']),sum(d.get('n_examples',0) for d in ex[l['b_id']]['recipe']['data'])) for pid,l in labels.items()}
methods['random']={pid:0.5 for pid in labels}
v3=methods['judge v3: + plan-time setup + launch scripts (inference-only)']
methods['hybrid: mean(judge v3, fixed logistic C=1)']={pid:(v3[pid]+methods['learned: fixed logistic C=1 on prior runs (pre-specified)'][pid])/2 for pid in v3}
def percell(pa):
    bc=defaultdict(list)
    for pid,l in labels.items():
        p=pa.get(pid)
        if p is None: continue
        acc=0.5 if p==0.5 else float((p>0.5)==(l['y_a']>l['y_b']))
        chosen=l['y_a'] if p>0.5 else l['y_b'] if p<0.5 else (l['y_a']+l['y_b'])/2
        bc[l['cell_id']].append((acc,max(l['y_a'],l['y_b'])-chosen))
    return bc
rng=random.Random(0); cells=sorted({l['cell_id'] for l in labels.values()})
boots=[[rng.choice(cells) for _ in cells] for _ in range(4000)]
def macro(bc,sample=None):
    cs=sample or list(bc)
    return statistics.mean(statistics.mean(a for a,_ in bc[c]) for c in cs if c in bc)
def regret(bc): return statistics.mean(statistics.mean(r for _,r in bc[c]) for c in bc)*100
ref=percell(methods['judge v3: + plan-time setup + launch scripts (inference-only)'])
rows=[]
for name,pa in methods.items():
    bc=percell(pa); n=sum(len(v) for v in bc.values())
    ci=sorted(macro(bc,s) for s in boots); lo,hi=ci[100],ci[-100]
    diff=sorted(macro(bc,s)-macro(ref,s) for s in boots); dlo,dhi=diff[100],diff[-100]
    micro=statistics.mean(a for v in bc.values() for a,_ in v)
    rows.append((name,n,macro(bc),lo,hi,micro,regret(bc),macro(bc)-macro(ref),dlo,dhi))
    print(f'{name:62s} n={n:2d} macro={macro(bc):.3f} [{lo:.3f},{hi:.3f}] micro={micro:.3f} regret={regret(bc):5.2f}pp  Δ vs judge v3={macro(bc)-macro(ref):+.3f} [{dlo:+.3f},{dhi:+.3f}]')
json.dump(rows,open(S+'score_rows.json','w'))
