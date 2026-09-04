"""Tiny native artifact serialization only: no forward, train, eval or inference."""
import hashlib,json
from pathlib import Path
import torch
from tokenizers import Tokenizer,models,pre_tokenizers
from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast,GenerationConfig
from awm.exp_protocol.save_contract import GenerationSaveContract

task=Path(__file__).parent
out=task/'artifacts'
out.mkdir()
vocab={word:i for i,word in enumerate(['<unk>','<bos>','<eos>','<pad>','red','blue','one','two','three','four','plus','equals','answer',':','5','9'])}
backend=Tokenizer(models.WordLevel(vocab=vocab,unk_token='<unk>'))
backend.pre_tokenizer=pre_tokenizers.Whitespace()
tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',bos_token='<bos>',eos_token='<eos>',pad_token='<pad>')
records=[]
for name,seed in [('a',31),('b',47)]:
    selected=(task/f'selected-{name}.json').read_bytes()
    frozen_hash=hashlib.sha256(selected).hexdigest()
    torch.manual_seed(seed)
    model=GPT2LMHeadModel(GPT2Config.from_pretrained(task/'random-init',local_files_only=True))
    model.generation_config=GenerationConfig(**json.loads(selected))
    target=out/f'export-{name}'
    saves=GenerationSaveContract(policy='inactive_sampling_v1')
    saves.check_before_compute(model)
    with saves.saving(model,target,selected_serving_json=selected):
        model.save_pretrained(target,safe_serialization=True,max_shard_size='8KB')
    tok.save_pretrained(target)
    assert (target/'generation_config.json').read_bytes()==selected
    # Recognized research-only files should not enter the selected serving manifest.
    (target/'trainer_state.json').write_text('{"reviewer_fixture":true}\n')
    records.append({'source':str(target),'seed':seed,'frozen_selected_generation_sha256':frozen_hash,
                    'save_records':saves.records,'model_forward_calls':0,'training_steps':0})
    del model
(task/'build-record.json').write_text(json.dumps(records,indent=2,default=str)+'\n')
print(json.dumps(records,indent=2,default=str))
