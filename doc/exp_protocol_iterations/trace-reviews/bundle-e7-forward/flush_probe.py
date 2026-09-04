import json
from pathlib import Path
from awm.exp_protocol.rendered_training import RenderedTrainingBundle
task=Path(__file__).parent
bundle=RenderedTrainingBundle.open_for_training(task/'memory/cards/exp-03.yaml')
collate=bundle.collator(return_tensors='python')
collate([bundle.dataset[0]])
bundle.flush_consumption()
input('READY: first use flushed; mutate the owned card fixture, then press Enter.\n')
record={'probe':'post-first-use-card-mutation','model_consumption':'unknown'}
try:
    bundle.flush_consumption()
except Exception as e:
    record.update(refused=True,error_type=type(e).__name__,error=str(e))
else:
    record.update(refused=False)
(task/'flush-probe.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
assert record['refused'], 'flush accepted changed locked plan'
