import datetime,json,os,subprocess,sys,time
from pathlib import Path
task=Path(__file__).parent
argv=sys.argv[1:]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic()
p=subprocess.run(argv,cwd=task,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,
        'returncode':p.returncode,'seconds':time.monotonic()-start,'output':p.stdout}
with (task/'transcript.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
print(p.stdout,end='');print('RETURN_CODE',p.returncode,'SECONDS',record['seconds'])
raise SystemExit(p.returncode)
