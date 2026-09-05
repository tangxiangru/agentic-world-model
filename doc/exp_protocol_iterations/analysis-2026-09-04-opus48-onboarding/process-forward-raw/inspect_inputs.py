"""CPU-only own synthetic input inspection. No models, generation or task execution."""
import ast,hashlib,json,sys
from pathlib import Path
from transformers import AutoTokenizer
task=Path(__file__).parent
domain=sys.argv[1]
source=task/'data'/('gpqa-style.jsonl' if domain=='gpqa' else 'code.jsonl')
tokenizer_dir='/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/data/ptb/hf/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d'
tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,token=False)
rows=[json.loads(l) for l in source.read_text().splitlines()]
observations=[]
for index,row in enumerate(rows):
    body=row['completion'].removesuffix('<end_of_turn>').rstrip()
    if domain=='code': ast.parse(body)  # syntax only; never execute the function
    prefix='<bos><start_of_turn>user\n'+row['prompt']+'<end_of_turn>\n<start_of_turn>model\n'
    completion=body+'<end_of_turn>\n'
    p=tok(prefix,add_special_tokens=False)['input_ids']
    c=tok(completion,add_special_tokens=False)['input_ids']
    labels=[-100]*len(p)+c
    assert (p+c).count(tok.bos_token_id)==1
    eot=tok.convert_tokens_to_ids('<end_of_turn>')
    assert c.count(eot)==1 and c[-2]==eot
    observations.append({'id':row['id'],'row':index+1,'raw_stop':row['completion'].endswith('<end_of_turn>'),
        'length':len(p+c),'supervised_tokens':len(c),'input_ids':p+c,'labels':labels,
        'kept':len(p+c)<=512,'syntax_parsed_only':domain=='code'})
audit={'domain':domain,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'raw_rows':len(rows),
       'unique_prompt_strings':len({r['prompt'] for r in rows}),
       'unique_prompt_completion_pairs':len({(r['prompt'],r['completion']) for r in rows}),
       'all_row_coverage':True,'observations':observations,'model_execution':'not_performed',
       'official_harness_template_equivalence':'not_established; reviewer synthetic prompt contract only'}
(task/f'{domain}-input-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='observations'},indent=2))
print('ROWS',json.dumps([{k:v for k,v in r.items() if k not in ('input_ids','labels')} for r in observations]))
