# CPU-only reproduction of the exact frozen _live_plan predicate.
# Earlier plan/lock checks are stubbed success to isolate the documented mismatch.
from pathlib import Path
from types import SimpleNamespace
import tempfile
source = Path(__file__).with_name('frozen-live-plan.py.txt').read_text()
class ExecutionError(Exception): pass
with tempfile.TemporaryDirectory() as td:
 root=Path(td); cp=root/'exp-01.yaml'; main=root/'sample.py'; main.write_text('# no inference\n')
 future=root/'future-training.jsonl'
 card={'card_id':'exp-01','setup':{'command':{'script':str(main),'cwd':str(root),'argv':['python',str(main)],'env':{}},'data':[{'path':str(future)}]}}
 info={'schema_version':'test','card_id':'exp-01','locked_at':'2026-09-05T00:00:00Z','script':{'path':str(main),'sha256':'pinned'},'data':[{'path':str(future),'sha256':None}],'overrides':{'data_files_exist':'produced after sampling'}}
 def get(d,path):
  for key in path.split('.'):
   if not isinstance(d,dict): return None
   d=d.get(key)
  return d
 schema=SimpleNamespace(load_card=lambda _:card,validate_plan=lambda *_:SimpleNamespace(ok=True),get=get)
 lock=SimpleNamespace(read_lock=lambda _:info,LOCK_SCHEMA='test',verify_lock=lambda *_:SimpleNamespace(problems=[]))
 preflight=SimpleNamespace(run_preflight=lambda *_:{'results':[]})
 scope=dict(Path=Path,schema=schema,lock=lock,preflight=preflight,ExecutionError=ExecutionError,RESERVED_ENV=set())
 exec(source,scope)
 for case in ['future data with override','existing file but absent pinned hash','sampling card with no future training declaration']:
  if case.startswith('existing'): future.write_text('{}\n')
  if case.startswith('sampling'):card['setup']['data']=[];info['data']=[]
  try:scope['_live_plan'](cp,root);print(case+': PASS')
  except ExecutionError as ex:print(case+': '+str(ex))
