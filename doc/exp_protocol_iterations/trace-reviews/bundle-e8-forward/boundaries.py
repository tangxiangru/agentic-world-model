"""Independent invalid and explicit scope-boundary fixtures; never load model weights."""
import json,shutil
from pathlib import Path
from awm.exp_protocol.serving_artifacts import snapshot_serving_artifact,verify_serving_artifact,publish_serving_artifact
task=Path(__file__).parent
a=json.loads((task/'memory/selections/a.json').read_text())
b=json.loads((task/'memory/selections/b.json').read_text())
cases=task/'boundary-fixtures'
cases.mkdir()
observations=[]
def invoke(label,fn):
    try:
        value=fn(); row={'case':label,'outcome':'returned','value':value}
    except Exception as e:
        row={'case':label,'outcome':'raised','type':type(e).__name__,'message':str(e),'report':getattr(e,'report',None)}
    observations.append(row)
    (task/'boundary-observations.json').write_text(json.dumps(observations,indent=2,default=str)+'\n')
    print(json.dumps(row,default=str),flush=True)
    return row
source=task/'artifacts/export-a'
invoke('A-with-B-selected-generation-hash',lambda:verify_serving_artifact(source,a['manifest'],b['expected_generation_sha256']))

drift=cases/'normalized-but-unselected'
shutil.copytree(source,drift)
gen=json.loads((drift/'generation_config.json').read_text()); gen['temperature']=1.0
(drift/'generation_config.json').write_text(json.dumps(gen)+'\n')
invoke('publish-normalized-unselected-A',lambda:publish_serving_artifact(drift,task/'must-not-be-published',a['manifest'],a['expected_generation_sha256'],
    session_dir=task,task_id='independent-e8-serving-forward',reference_id='unselected-normalization'))

missing=cases/'missing-shard'
shutil.copytree(source,missing)
shard=missing/'model-00002-of-00003.safetensors'
shard.rename(cases/'held-missing-shard.safetensors')
invoke('missing-native-index-shard',lambda:snapshot_serving_artifact(missing))

named=cases/'named-template-directory'
shutil.copytree(source,named)
(named/'chat_templates').mkdir(); (named/'chat_templates/custom.jinja').write_text('{{ messages }}\n')
invoke('named-template-subdirectory',lambda:snapshot_serving_artifact(named))

# Demonstrate the documented limit: metadata validity does not align parameter shapes.
misaligned=cases/'config-weight-dimension-mismatch'
shutil.copytree(source,misaligned)
config=json.loads((misaligned/'config.json').read_text()); config['n_embd']=32
(misaligned/'config.json').write_text(json.dumps(config)+'\n')
def metadata_only_mismatch():
    current=snapshot_serving_artifact(misaligned)
    return verify_serving_artifact(misaligned,current,a['expected_generation_sha256'])
invoke('metadata-valid-but-config-weight-dimension-mismatch',metadata_only_mismatch)

# This payload is intentionally not a usable pickle checkpoint; no pickle is executed.
opaque=cases/'opaque-not-a-pickle'
opaque.mkdir()
for entry in a['manifest']['content']['files']:
    name=entry['path']
    if not name.startswith('model'):
        shutil.copy2(source/name,opaque/name)
(opaque/'pytorch_model.bin').write_bytes(b'INDEPENDENT REVIEW: opaque bytes, not executable pickle\n')
invoke('opaque-bin-without-opt-in',lambda:snapshot_serving_artifact(opaque))
def explicit_opaque():
    current=snapshot_serving_artifact(opaque,allow_opaque_weights=True)
    return verify_serving_artifact(opaque,current,a['expected_generation_sha256'],allow_opaque_weights=True)
invoke('opaque-bin-explicit-byte-only-opt-in',explicit_opaque)
print('Only reviewer-owned fixtures were changed; missing shard retained at',cases/'held-missing-shard.safetensors')
