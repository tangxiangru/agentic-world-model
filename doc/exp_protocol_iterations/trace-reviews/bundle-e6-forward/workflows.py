"""Independent guide-selected synthetic workflows. No LLM.__init__ or inference."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from awm.exp_protocol import sampling

ROOT = Path('/tmp/e6-forward-sampling.ntRxXE')
CARD = ROOT / 'memory/cards/exp-01.yaml'
TOK = '/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/data/ptb/hf/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d'
tokenizer = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
inputs = [json.loads(line) for line in (ROOT/'inputs.jsonl').read_text().splitlines()]
texts = ['<bos><start_of_turn>user\n'+r['question']+'<end_of_turn>\n<start_of_turn>model\n' for r in inputs]
prepared = sampling.prepare_prompts(texts, tokenizer, item_ids=[r['id'] for r in inputs], bos_policy='single_at_start')
stop_ids = sampling.resolve_stop_ids(tokenizer, ['<end_of_turn>'])
params = SamplingParams(n=2, temperature=0.7, top_p=0.91, max_tokens=32,
                        stop_token_ids=stop_ids, stop=['END_FIXTURE'], seed=17)
before_params = repr(params)
llm = LLM.__new__(LLM)
llm.llm_engine = SimpleNamespace(get_tokenizer=lambda: tokenizer)
original_call = sampling._call_engine
engine_calls = []

def response_batch(kind):
    batch = []
    for i, p in enumerate(prepared):
        completions = []
        for j in range(2):
            text = 'ANSWER: ' + ('1e999' if (i,j)==(0,1) else '4' if i<2 else '7')
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids += [1 if (i,j)==(0,1) else stop_ids[0]]
            completions.append(CompletionOutput(index=j, text=text, token_ids=ids,
                cumulative_logprob=None, logprobs=None, finish_reason='stop',
                stop_reason=None if (i,j)==(0,1) else stop_ids[0]))
        batch.append(RequestOutput(request_id='synthetic-request-'+str(i), prompt=p.text,
            prompt_token_ids=list(p.token_ids), prompt_logprobs=None,
            outputs=completions, finished=True))
    if kind=='missing_draw': batch[-1].outputs.pop()
    if kind=='wrong_order': batch[0], batch[2] = batch[2], batch[0]
    if kind=='aborted': batch[-1].outputs[-1].finish_reason='abort'
    return batch

def file_hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def listing(path):
    return {p.name: {'size': p.stat().st_size, 'sha256': file_hash(p)} for p in sorted(path.iterdir()) if p.is_file()} if path.exists() else {}

def emit(label, value):
    print(label, json.dumps(value, default=str, sort_keys=True), flush=True)

def numeric_parser(text, input_metadata):
    return {'value':sampling.finite_float(text.split('ANSWER: ')[-1]), 'input':input_metadata}

def recovery_parser(text, input_metadata):
    try:
        value = sampling.finite_float(text.split('ANSWER: ')[-1])
    except (ValueError, OverflowError):
        return {'status':'nonfinite_answer', 'value':None, 'input':input_metadata}
    return {'status':'parsed', 'value':value, 'input':input_metadata}

def interrupted_parser(text, input_metadata):
    raise KeyboardInterrupt('synthetic parser interruption; no inference')

try:
    for label, fn in [
        ('double_bos', lambda:sampling.prepare_prompts(['<bos>'+texts[0]], tokenizer, bos_policy='single_at_start')),
        ('unknown_stop', lambda:sampling.resolve_stop_ids(tokenizer,['<not_a_real_fixture_stop>'])),
        ('multi_token_stop', lambda:sampling.resolve_stop_ids(tokenizer,['end fixture now'])),
    ]:
        try: emit(label, {'returned':fn()})
        except Exception as e: emit(label, {'exception':type(e).__name__, 'message':str(e)})
    for kind in ['valid', 'missing_draw', 'wrong_order', 'aborted']:
        def inert(*args, **kwargs):
            engine_calls.append(kind)
            emit('engine_intercept', {'kind':kind, 'arg_types':[type(a).__name__ for a in args], 'kwargs':list(kwargs)})
            return response_batch(kind)
        sampling._call_engine = inert
        out = ROOT/'recordings'/kind
        try:
            capture = sampling.record_vllm(llm, prepared, params, out, card_path=CARD,
                                           required_stop_tokens=['<end_of_turn>'])
            emit('capture_'+kind, capture)
        except BaseException as e:
            emit('capture_'+kind, {'exception':type(e).__name__, 'message':str(e)})
        emit('files_'+kind, listing(out))
        if (out/'capture.json').exists():
            raw_before = file_hash(out/'raw.jsonl')
            for parse_label, parser in [('numeric',numeric_parser), ('interrupted',interrupted_parser), ('recovery',recovery_parser)]:
                try: emit('parse_'+parse_label, sampling.parse_recording(out, parser))
                except BaseException as e: emit('parse_'+parse_label, {'exception':type(e).__name__, 'message':str(e)})
                emit('raw_unchanged_'+parse_label, file_hash(out/'raw.jsonl')==raw_before)
            emit('valid_files_after_parsing', listing(out))
            saved = listing(out)
            try:
                sampling.record_vllm(llm, prepared, params, out, card_path=CARD, required_stop_tokens=['<end_of_turn>'])
                emit('reuse', 'unexpected returned capture')
            except BaseException as e: emit('reuse', {'exception':type(e).__name__, 'message':str(e), 'unchanged':listing(out)==saved})
        elif (out/'raw.jsonl').exists():
            try: emit('parse_invalid_'+kind, sampling.parse_recording(out, recovery_parser))
            except BaseException as e: emit('parse_invalid_'+kind, {'exception':type(e).__name__, 'message':str(e)})
    emit('original_params_unchanged', repr(params)==before_params)
    emit('engine_calls', engine_calls)
finally:
    sampling._call_engine = original_call
