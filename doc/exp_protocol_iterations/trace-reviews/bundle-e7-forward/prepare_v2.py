import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
from awm.exp_protocol.rendered_training import RenderedSettings, RenderedTrainingBundle
from preprocessing import render

task=Path(__file__).parent
mode,source,output=sys.argv[1:4]
tok=AutoTokenizer.from_pretrained('/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/data/ptb/hf/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d',local_files_only=True,token=False)
settings=RenderedSettings(mode=mode,prompt_mode='template_replay' if mode=='joint_prefix' else 'pre_rendered',
    max_seq_len=256,stop_token='<end_of_turn>',answer_marker='ANSWER: ',tail_text='\n',
    pad_to_multiple_of=8,seed=13,max_drop_fraction=0.30)
prepared=RenderedTrainingBundle.prepare(sources=[task/source],render=render,tokenizer=tok,
    template_bytes=(task/'chat.jinja').read_bytes(),settings=settings,
    source_files=[task/'consume_v2.py',task/'preprocessing.py',task/'prepare_v2.py'],
    output=task/output,reuse='--reuse' in sys.argv)
print(json.dumps({'declaration':prepared.declaration,'data_entry':prepared.data_entry,'report':prepared.report},indent=2,default=str))
