"""Forward use of public publication API, only on reviewer-owned tiny artifacts."""
import hashlib,json
from pathlib import Path
from awm.exp_protocol.serving_artifacts import snapshot_serving_artifact,verify_serving_artifact,publish_serving_artifact
task=Path(__file__).parent
chosen={n:json.loads((task/f'memory/selections/{n}.json').read_text()) for n in ('a','b')}
dest=task/'final_model'
observations=[]

def invoke(label,fn):
    try:
        value=fn()
        result={'case':label,'outcome':'returned','result':value}
    except Exception as e:
        result={'case':label,'outcome':'raised','type':type(e).__name__,'message':str(e),'report':getattr(e,'report',None)}
    observations.append(result)
    (task/'publication-observations.json').write_text(json.dumps(observations,indent=2,default=str)+'\n')
    print(json.dumps(result,default=str),flush=True)
    return result

def publish(name,target=dest,**kwargs):
    x=chosen[name]
    return publish_serving_artifact(task/f'artifacts/export-{name}',target,x['manifest'],x['expected_generation_sha256'],
        session_dir=task,task_id='independent-e8-serving-forward',reference_id=f'caller-selected-{name}',**kwargs)

first=invoke('publish-selected-A-to-absent-final',lambda:publish('a'))
assert first['outcome']=='returned'
assert (dest/'generation_config.json').read_bytes()==(task/'selected-a.json').read_bytes()
assert not (dest/'trainer_state.json').exists()
frozen_old=snapshot_serving_artifact(dest)
(task/'memory/selections/incumbent-before.json').write_text(json.dumps(frozen_old,indent=2)+'\n')
old_identity=frozen_old['identity_sha256']

invoke('existing-final-without-replace',lambda:publish('b'))
invoke('replacement-without-quiescence',lambda:publish('b',replace=True,expected_old_identity=old_identity))
invoke('replacement-wrong-old-identity',lambda:publish('b',replace=True,expected_old_identity='0'*64,
    target_quiescent=True,quiescence_evidence='Owned synthetic fixture; no consumer has been started.'))
assert snapshot_serving_artifact(dest)['identity_sha256']==old_identity

evidence={'scope':str(dest),'ownership':'created only in this fresh reviewer task',
    'known_model_or_evaluator_consumers':[],'model_or_evaluator_was_ever_started':False,
    'metadata_calls':'all synchronous and returned before this update',
    'producer_exit_record':str(task/'memory/attempts/exp-01/b6241f74590e41b1b028d1c33aa4a15b/finish.json'),
    'limits':'caller-established local fixture condition; not a universal process-ownership proof'}
(task/'quiescence-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
updated=invoke('replace-incumbent-A-with-selected-B',lambda:publish('b',replace=True,expected_old_identity=old_identity,
    target_quiescent=True,quiescence_evidence=str(task/'quiescence-evidence.json')))
assert updated['outcome']=='returned'
assert (dest/'generation_config.json').read_bytes()==(task/'selected-b.json').read_bytes()
final_check=verify_serving_artifact(dest,chosen['b']['manifest'],chosen['b']['expected_generation_sha256'])
(task/'final-verification.json').write_text(json.dumps(final_check,indent=2)+'\n')
assert final_check['identity_sha256']==chosen['b']['manifest']['identity_sha256']
print('FINAL_SELECTED_B_EXACT',final_check['identity_sha256'])
