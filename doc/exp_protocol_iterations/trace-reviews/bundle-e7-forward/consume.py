import copy
import json
import sys
from pathlib import Path
from awm.exp_protocol.rendered_training import RenderedTrainingBundle

task = Path(__file__).parent
card = Path(sys.argv[1])
bundle = RenderedTrainingBundle.open_for_training(card)
collator = bundle.collator(pad_to_multiple_of=8, return_tensors='python')
features = [bundle.dataset[i] for i in range(len(bundle.dataset))]
batch = collator(features)
print('FEATURES',json.dumps(features))
print('BATCH',json.dumps(batch))
for i,row in enumerate(features):
    length=len(row['input_ids'])
    assert batch['input_ids'][i][:length] == row['input_ids']
    assert batch['labels'][i][:length] == row['labels']
    assert all(v == -100 for v in batch['labels'][i][length:])
    assert all(v == 0 for v in batch['attention_mask'][i][length:])
    assert all(v == 1 for v in batch['attention_mask'][i][:length])
altered=copy.deepcopy(features[0])
altered['labels'][-2] = 12345
try:
    collator([altered])
except Exception as e:
    print('ALTERED_FEATURE_REFUSED',type(e).__name__,str(e))
else:
    raise AssertionError('altered feature accepted')
bundle.flush_consumption()
print('DONE: CPU dataset access and collation only; actual model consumption unknown.')
