"""k-nearest-neighbour pairwise ranker over the labeled prior-run bank, same folds as the judge."""
import json, math, statistics, random
from pathlib import Path
from collections import defaultdict
J=Path('data/analysis/rpm/judge')
labels={l['id']:l for l in json.load(open(J/'hidden_labels.json'))}
folds=json.load(open(J/'folds.json'))
ex={json.loads(l)['example_id']:json.loads(l) for l in open('data/analysis/outcome_prediction/examples.jsonl')}
def feats(r):
    f=dict(r['numeric_features']); 
    for k,v in r['categorical_features'].items(): f[f'cat:{k}={v}']=1.0
    f['depth']=len(r['lineage'])
    return f
def knn_predict(train_ids, target, k):
    T=feats(ex[target])
    # z-scale numeric on train
    keys=sorted({k for t in train_ids for k in feats(ex[t])} | set(T))
    cols={k:[feats(ex[t]).get(k) for t in train_ids] for k in keys}
    mu={k:statistics.mean([v for v in cols[k] if v is not None] or [0]) for k in keys}
    sd={k:(statistics.pstdev([v for v in cols[k] if v is not None]) or 1.0) if sum(v is not None for v in cols[k])>1 else 1.0 for k in keys}
    def vec(f): return {k:((f.get(k)-mu[k])/sd[k] if f.get(k) is not None else 0.0) for k in keys}
    tv=vec(T)
    d=[]
    for t in train_ids:
        fv=vec(feats(ex[t])); d.append((math.sqrt(sum((tv[k]-fv[k])**2 for k in keys)), ex[t]['y']))
    d.sort(); nn=d[:k]
    w=[1/(x[0]+1e-3) for x in nn]
    return sum(wi*yi for wi,(_,yi) in zip(w,nn))/sum(w)
def score(pa, name):
    rows=[]
    for pid,l in labels.items():
        p=pa.get(pid); 
        if p is None: continue
        acc=0.5 if p==0.5 else float((p>0.5)==(l['y_a']>l['y_b']))
        chosen=l['y_a'] if p>0.5 else l['y_b'] if p<0.5 else (l['y_a']+l['y_b'])/2
        rows.append((l['cell_id'],acc,max(l['y_a'],l['y_b'])-chosen))
    bc=defaultdict(list)
    for r in rows: bc[r[0]].append(r)
    macro=statistics.mean(statistics.mean(x[1] for x in v) for v in bc.values())
    print(f'{name:40s} n={len(rows)} macro={macro:.3f} micro={statistics.mean(r[1] for r in rows):.3f} regret={statistics.mean(statistics.mean(x[2] for x in v) for v in bc.values())*100:.2f}')
    return {r_id:None for r_id in []}
for k in (3,5,10):
    pa={}
    for pid,l in labels.items():
        tr=folds[l['fold']]['train_ids']
        ya=knn_predict(tr,l['a_id'],k); yb=knn_predict(tr,l['b_id'],k)
        pa[pid]=0.5 if ya==yb else (1.0 if ya>yb else 0.0)
    score(pa,f'kNN k={k} (prior-run bank, whole-run CV)')
    json.dump(pa,open(f'data/analysis/rpm/judge_code/knn{k}.json','w'))
