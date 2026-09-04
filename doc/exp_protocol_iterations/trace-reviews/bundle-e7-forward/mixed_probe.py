from pathlib import Path
from awm.exp_protocol.rendered_training import RenderedTrainingBundle
task=Path(__file__).parent
a=RenderedTrainingBundle.open_for_training(task/'memory/cards/exp-02.yaml')
b=RenderedTrainingBundle.open_for_training(task/'memory/cards/exp-03.yaml')
collate=a.collator(return_tensors='python')
try:
    collate([a.dataset[0],b.dataset[0]])
except Exception as e:
    print('MIXED_BUNDLES_REFUSED',type(e).__name__,str(e))
else:
    raise AssertionError('mixed receipt identities accepted')
a.flush_consumption()
b.flush_consumption()
print('CPU only; both model_consumption values remain unknown.')
