import copy
import json
import sys
from pathlib import Path
from awm.exp_protocol.rendered_training import RenderedTrainingBundle

task = Path(__file__).parent
bundle = RenderedTrainingBundle.open_for_training(Path(sys.argv[1]))
collator = bundle.collator(pad_to_multiple_of=8, return_tensors='python')
features = [bundle.dataset[i] for i in range(len(bundle.dataset))]
batch = collator(features)
for i,row in enumerate(features):
    mask=batch['attention_mask'][i]
    assert [v for v,m in zip(batch['input_ids'][i],mask) if m] == row['input_ids']
    assert [v for v,m in zip(batch['labels'][i],mask) if m] == row['labels']
    assert all(v == -100 for v,m in zip(batch['labels'][i],mask) if not m)
    assert len(mask) % 8 == 0
print('VALID_BATCH',json.dumps({'n':len(features),'lengths':[len(x['input_ids']) for x in features],
    'padded_widths':[len(x) for x in batch['input_ids']],
    'supervised_counts':[sum(v!=-100 for v in x['labels']) for x in features],
    'bos_counts':[x['input_ids'].count(2) for x in features]}))
for label,change in [('changed-label',lambda f:f['labels'].__setitem__(-2,12345)),
                     ('missing-identity',lambda f:f.pop('_awm_bundle_sha256'))]:
    altered=copy.deepcopy(features[0]); change(altered)
    try:
        collator([altered])
    except Exception as e:
        print('REFUSED',label,type(e).__name__,str(e))
    else:
        raise AssertionError(label+' accepted')
bundle.flush_consumption()
print('DONE: CPU dataset access and collation only; actual model consumption unknown.')
