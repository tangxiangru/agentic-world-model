import datetime,json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).parent
argv=['bash',str(root/'offline.sh')]+sys.argv[1:]
start=time.monotonic()
result=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,
        'returncode':result.returncode,'seconds':time.monotonic()-start,'output':result.stdout}
with (root/'transcript.jsonl').open('a') as fh:
    fh.write(json.dumps(record)+'\n')
print(result.stdout,end='')
print('RETURN_CODE',result.returncode,'SECONDS',record['seconds'])
raise SystemExit(result.returncode)
