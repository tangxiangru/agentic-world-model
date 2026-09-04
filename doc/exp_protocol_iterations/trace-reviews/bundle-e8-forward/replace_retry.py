"""Corrected public API use: expected_old_identity is the full frozen manifest."""
import json
from pathlib import Path
from awm.exp_protocol.serving_artifacts import snapshot_serving_artifact,verify_serving_artifact,publish_serving_artifact
task=Path(__file__).parent
selections=task/'memory/selections'
old=json.loads((selections/'incumbent-before.json').read_text())
b=json.loads((selections/'b.json').read_text())
observations=[]
def invoke(label,**kwargs):
    try:
        record=publish_serving_artifact(task/'artifacts/export-b',task/'final_model',b['manifest'],b['expected_generation_sha256'],
            session_dir=task,task_id='independent-e8-serving-forward',reference_id=label,replace=True,**kwargs)
        item={'case':label,'outcome':'returned','record':record}
    except Exception as e:
        item={'case':label,'outcome':'raised','type':type(e).__name__,'message':str(e),'report':getattr(e,'report',None)}
    observations.append(item)
    (task/'replacement-observations.json').write_text(json.dumps(observations,indent=2,default=str)+'\n')
    print(json.dumps({'case':label,'outcome':item['outcome'],'detail':item.get('message'),
                      'record':item.get('record',{}).get('record_path'),'report':item.get('report')},default=str),flush=True)
    return item
invoke('full-manifest-without-quiescence',expected_old_identity=old)
invoke('wrong-full-incumbent-manifest',expected_old_identity=b['manifest'],target_quiescent=True,
       quiescence_evidence=str(task/'quiescence-evidence.json'))
assert snapshot_serving_artifact(task/'final_model')['identity_sha256']==old['identity_sha256']
success=invoke('caller-selected-B-replaces-A',expected_old_identity=old,target_quiescent=True,
       quiescence_evidence=str(task/'quiescence-evidence.json'))
assert success['outcome']=='returned'
record=success['record']
final=verify_serving_artifact(task/'final_model',b['manifest'],b['expected_generation_sha256'])
a=json.loads((selections/'a.json').read_text())
backup=verify_serving_artifact(Path(record['backup']),old,a['expected_generation_sha256'])
assert (task/'final_model/generation_config.json').read_bytes()==(task/'selected-b.json').read_bytes()
(task/'final-verification.json').write_text(json.dumps({'final':final,'backup':backup,'publication_record':record['record_path']},indent=2)+'\n')
print('FINAL_B_AND_BACKUP_A_VERIFIED',json.dumps({'final':final['identity_sha256'],'backup':backup['identity_sha256'],'backup_path':record['backup']}))
