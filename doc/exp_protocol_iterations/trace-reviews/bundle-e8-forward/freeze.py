import hashlib,json
from pathlib import Path
from awm.exp_protocol.serving_artifacts import snapshot_serving_artifact,verify_serving_artifact
task=Path(__file__).parent
selections=task/'memory/selections'
selections.mkdir(exist_ok=True)
for name in ('a','b'):
    source=task/f'artifacts/export-{name}'
    selected=snapshot_serving_artifact(source)
    generation_sha=hashlib.sha256((task/f'selected-{name}.json').read_bytes()).hexdigest()
    (selections/f'{name}.json').write_text(json.dumps({'manifest':selected,'expected_generation_sha256':generation_sha,
        'reference':'synthetic selected native fixture; never evaluated'},indent=2,default=str)+'\n')
    verified=verify_serving_artifact(source,selected,generation_sha)
    print(name,json.dumps({'manifest':selected,'verified':verified},indent=2,default=str))
