import json,collections,re
from pathlib import Path
import yaml
P=Path('/tmp/wma-usage-20260904');cs=json.loads((P/'complete-cells.json').read_text());out=[]
def variant(b):
 if '-r01-' in b:return 'R1 historical'
 for tok,label in [('-v02-baseline','v0.2 blocking'),('-ab-','A+B'),('-a-','A'),('-c-','C'),('-d-','D'),('-e-','E'),('-f-','F')]:
  if tok in b:return label
 return 'Other'
for c in cs:
 if not c['cell'].startswith('w'):continue
 p=Path(c['result_dir']);cards=[]
 for f in sorted((p/'task/memory/cards').glob('exp-*.yaml')):
  d=yaml.safe_load(f.read_text());vpath=f.with_suffix('.verdict.json');v=json.loads(vpath.read_text()) if vpath.exists() else {}
  cards.append({'card':f.stem,'family':d.get('setup',{}).get('method',{}).get('family'),'execution':d.get('result',{}).get('execution'),'decision':d.get('conclusion',{}).get('decision'),'alternatives':d.get('situation',{}).get('alternatives_rejected') or [],'final_L3':v.get('levels',{}).get('L3_worth_now',{}).get('answer'),'has_verdict':bool(v),'flag':bool(v.get('leak_suspected'))})
 reqs=[]
 for f in sorted((p/'task/.wma/processed').glob('*.json')):
  r=json.loads(f.read_text());rp=p/'task/.wma/responses'/f.name;rr=json.loads(rp.read_text()) if rp.exists() else {};reqs.append({'id':r.get('request_id'),'cards':r.get('card_ids'),'created_at':r.get('created_at'),'state':rr.get('state'),'ranking':rr.get('ranking')})
 ls=(p/'solve_parsed.txt').read_text().splitlines();current='';std=[]
 for i,l in enumerate(ls,1):
  if re.match(r'^(Assistant|User) —',l):current=l
  if re.match(r'^\s+verdict: L0_runs=',l):
   m=re.search(r'L3_worth_now=(yes|no|defer)',l)
   if m:std.append({'line':i,'time_context':current,'L3':m[1],'summary':l.strip()})
 row={**c,'variant':variant(c['batch']),'cards':cards,'requests':reqs,'standard_summaries':std,'request_sizes':dict(collections.Counter(len(r['cards'] or []) for r in reqs)),'final_L3':dict(collections.Counter(r['final_L3'] for r in cards if r['has_verdict'])),'standard_L3':dict(collections.Counter(r['L3'] for r in std))};out.append(row)
(P/'usage-census.json').write_text(json.dumps(out,indent=2))
for v in dict.fromkeys(r['variant'] for r in out):
 rr=[r for r in out if r['variant']==v];cards=[c for r in rr for c in r['cards']];reqs=[q for r in rr for q in r['requests']]
 print(v,'cells',len(rr),'cards',len(cards),'family',dict(collections.Counter(c['family'] for c in cards)),'requests',len(reqs),'sizes',dict(collections.Counter(len(q['cards'] or []) for q in reqs)),'finalL3',dict(collections.Counter(c['final_L3'] for c in cards if c['has_verdict'])))
for r in out:
 if r['variant'] not in ['R1 historical','A','A+B','C']:
  print('CELL',r['cell'],r['variant'],'req',len(r['requests']),'std',len(r['standard_summaries']),'L3',r['standard_L3'],'unrun',[(c['card'],c['execution']) for c in r['cards'] if c['execution'] not in ['completed']])
