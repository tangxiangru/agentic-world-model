import json,subprocess,os,datetime,concurrent.futures
from pathlib import Path
p=Path('/tmp/wma-usage-20260904');sources=json.loads((p/'sources.json').read_text());manifests=sorted({s['manifest'] for s in sources})
def one(m):
 r=subprocess.run(['uv','run','--no-sync','awm','ptb','results',m,'--all','--json'],env={**os.environ,'UV_CACHE_DIR':'/tmp/uv-cache'},text=True,capture_output=True)
 if r.returncode:raise RuntimeError(m+':'+r.stderr)
 return json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rs=list(pool.map(one,manifests))
cs=[]
for r in rs:
 for c in r['rows']:
  if c['complete']:
   a=c['completed_attempt'];cs.append({'cell':c['cell_id'],'batch':r['batch_id'],'result_dir':a['result_dir'],'accuracy':a['accuracy'],'judge_flags':a['judge_flags'],'job':a['job_id'],'manifest':r['manifest'],'spec':r['spec']})
(p/'results.snapshot.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'results':rs},indent=2));(p/'complete-cells.json').write_text(json.dumps(cs,indent=2))
old=json.loads(Path('doc/wma_iterations/evidence/2026-09-04-sft-efficiency/input-complete-cells.json').read_text());keys={(x['batch'],x['cell']) for x in old};new=[c for c in cs if (c['batch'],c['cell']) not in keys];(p/'new-cells.json').write_text(json.dumps(new,indent=2))
print('complete',len(cs),'clean',sum(not c['judge_flags'] for c in cs),'new',len(new))
for c in new:print(c['cell'],c['batch'],c['accuracy'],c['judge_flags'])
