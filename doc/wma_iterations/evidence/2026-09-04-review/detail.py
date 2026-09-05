import json,collections,statistics
from pathlib import Path
from awm.wma import ledger
root=Path('/tmp/wma-deep-analysis');cells=json.loads((root/'complete-cells.json').read_text())
out={}
for batch in dict.fromkeys(c['batch'] for c in cells):
 cc=[c for c in cells if c['batch']==batch];rs=ledger.rows([Path(c['result_dir'])/'task' for c in cc]);
 if not rs: continue
 clean=[r for r in rs if not r['leak'] and r['scored'].get('L2','unscorable')!='unscorable']
 l3=collections.Counter(r['L3_answer'] for r in rs);negative=[];locks=collections.Counter();times=[];requests=0
 for c in cc:
  p=Path(c['result_dir'])
  for f in (p/'task/memory/cards').glob('exp-*.verdict.json'):
   v=json.loads(f.read_text()); lv=v['levels']
   if any(lv[k]['answer']!='yes' for k in ['L0_runs','L1_valid','L3_worth_now']): negative.append({'cell':c['cell'],'card':f.stem,'answers':{k:lv[k]['answer'] for k in ['L0_runs','L1_valid','L3_worth_now']},'flagged':v.get('leak_suspected')})
  for f in (p/'task/memory/cards').glob('exp-*.lock.json'):
   v=json.loads(f.read_text());w=v.get('wma') or {};locks[w.get('state','missing')]+=1
   if isinstance(w.get('waited_s'),(float,int)): times.append(w['waited_s'])
  requests+=len(list((p/'task/.wma/requests').glob('*.json')))
 out[batch]={'cells':len(cc),'L3':dict(l3),'negative_final':negative,'clean_L2_n':len(clean),'clean_L2_hits':sum(r['scored']['L2']=='in_interval' for r in clean),'clean_L2_width_mean':statistics.mean(r['L2_width'] for r in clean) if clean else None,'clean_L2_width_over_noise_mean':statistics.mean(r['L2_width_over_noise'] for r in clean) if clean else None,'terminal_locks':dict(locks),'terminal_lock_wait_sum_min':sum(times)/60,'terminal_lock_wait_mean_min':statistics.mean(times)/60 if times else None,'requests':requests}
(root/'detail.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
try:
 import scipy,matplotlib
 print('SCIPY',scipy.__version__,'MPL',matplotlib.__version__)
except ImportError as e: print(e)
