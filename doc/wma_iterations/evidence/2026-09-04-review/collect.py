import json,subprocess,os,datetime,collections,concurrent.futures
from pathlib import Path
sources=[s for s in json.loads(Path('/tmp/wma_analysis_sources.json').read_text()) if s.get('batch_id','').startswith('wma-')]
by_manifest={s['manifest']:s['batch_id'] for s in sources}
def run(item):
    manifest,batch=item
    p=subprocess.run(['uv','run','--no-sync','awm','ptb','results',manifest,'--all','--json'],env={**os.environ,'UV_CACHE_DIR':'/tmp/uv-cache'},capture_output=True,text=True)
    if p.returncode: return {'batch_id':batch,'error':p.stderr}
    d=json.loads(p.stdout)
    Path('/tmp/wma-deep-analysis/'+batch+'.json').write_text(json.dumps(d,indent=2))
    return d
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(run,by_manifest.items()))
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'results':results}
Path('/tmp/wma-deep-analysis/results.snapshot.json').write_text(json.dumps(out,indent=2))
for d in results:
    print(d['batch_id'],{k:v for k,v in d.items() if k not in ['batch_id','rows','manifest','incomplete_cells']},'rows',len(d.get('rows',[])),flush=True)
