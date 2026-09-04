import inspect
import importlib.metadata
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from awm.exp_protocol import sampling

print('versions', {p:importlib.metadata.version(p) for p in ['vllm','transformers','torch']})
for x in [LLM, RequestOutput, CompletionOutput, sampling.prepare_prompts,
          sampling.resolve_stop_ids, sampling.record_vllm, sampling.parse_recording,
          sampling.finite_float]:
    print(x.__name__, inspect.signature(x))
tok = AutoTokenizer.from_pretrained('/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/data/ptb/hf/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d', local_files_only=True)
p = sampling.prepare_prompts(['<bos><start_of_turn>user\nSynthetic fixture: add two blue stones to two blue stones.<end_of_turn>\n<start_of_turn>model\n'], tok, item_ids=[7], bos_policy='single_at_start')
print('prepared_type', type(p), 'prepared_value', p)
print('stop', sampling.resolve_stop_ids(tok, ['<end_of_turn>']))
print('cli_import')
from awm.cli import app
print(type(app).__name__)
