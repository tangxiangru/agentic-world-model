"""Read-only reproduction: reverify final B and recoverable incumbent A."""
import json
from pathlib import Path
from awm.exp_protocol.serving_artifacts import verify_serving_artifact
task=Path(__file__).parent
items={n:json.loads((task/f'memory/selections/{n}.json').read_text()) for n in ('a','b')}
record=json.loads((task/'memory/serving-publications/711799eee2064a67bed086cc9beb949c.json').read_text())
for label,path,n in [('final',task/'final_model','b'),('backup',Path(record['backup']),'a')]:
    x=items[n]
    result=verify_serving_artifact(path,x['manifest'],x['expected_generation_sha256'])
    assert (path/'generation_config.json').read_bytes()==(task/f'selected-{n}.json').read_bytes()
    assert not (path/'trainer_state.json').exists()
    print(label,json.dumps(result,indent=2))
