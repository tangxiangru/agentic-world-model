import json,collections,statistics,csv
from pathlib import Path
from awm.wma import ledger
from awm import slurm_queue
root=Path('/tmp/wma-deep-analysis')
d=json.loads((root/'results.snapshot.json').read_text())
s=json.loads(Path('/rmeng_data/robtang/slurm-queue/current.json').read_text())
s=slurm_queue.select_subqueue(s,'gangda_wma_evolve')
s['sources']=[x for x in s['sources'] if x.get('batch_id','').startswith('wma-')]
(root/'queue.snapshot.json').write_text(json.dumps(s,indent=2))
print(slurm_queue.render_snapshot(s,include_jobs=False))
jobs=[j for x in s['sources'] for j in x['jobs']]
print('JOB STATES',collections.Counter(j['state'] for j in jobs),'UNIQUE JOBS',len({j['job_id'] for j in jobs}))
print('registry batch cells',len({(x['batch_id'],j['cell_id']) for x in s['sources'] for j in x['jobs']}))
records=[]; allrows=[]
for r in d['results']:
    batch=r['batch_id']; qs=[j for x in s['sources'] if x['batch_id']==batch for j in x['jobs']]
    comp=[x for x in r['rows'] if x['complete']]; scores=[x['completed_attempt']['accuracy'] for x in comp]
    ds=[]
    for x in comp:
        p=Path(x['completed_attempt']['result_dir']); ds.append(p/'task')
        cards=list((p/'task/memory/cards').glob('exp-*.yaml'))
        verdicts=list((p/'task/memory/cards').glob('exp-*.verdict.json'))
        records.append({'batch':batch,'cell':x['cell_id'],'accuracy':x['completed_attempt']['accuracy'],'job':x['completed_attempt']['job_id'],'result_dir':str(p),'manifest':r['manifest'],'spec':r['spec'],'cards':len(cards),'verdicts':len(verdicts),'receipts':[q['path'] for q in s['sources'] if q['batch_id']==batch and any(j['cell_id']==x['cell_id'] for j in q['jobs'])]})
    ls=list(ledger.rows(ds)); allrows.extend(ls)
    summ=ledger.summarize(ls)
    (root/(batch+'.ledger.json')).write_text(json.dumps(summ,indent=2))
    summary={'batch':batch,'complete':len(comp),'intended':r['total'],'states':dict(collections.Counter(j['state'] for j in qs)),'mean':statistics.mean(scores) if scores else None,'sd':statistics.stdev(scores) if len(scores)>1 else None,'min':min(scores) if scores else None,'max':max(scores) if scores else None,'scores':scores,'cards':sum(x['cards'] for x in records if x['batch']==batch),'verdicts':sum(x['verdicts'] for x in records if x['batch']==batch)}
    if comp or any(j['state']=='RUNNING' for j in qs): print(json.dumps(summary))
    (root/(batch+'.summary.json')).write_text(json.dumps(summary,indent=2))
(root/'complete-cells.json').write_text(json.dumps(records,indent=2))
(root/'ledger.all.json').write_text(json.dumps(ledger.summarize(allrows),indent=2))
(root/'ledger.bytype.json').write_text(json.dumps(ledger.summarize(allrows,by='type'),indent=2))
print('CARDS',sum(x['cards'] for x in records),'VERDICTS',sum(x['verdicts'] for x in records))
print('LEDGER',json.dumps(ledger.summarize(allrows),indent=2))
