import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

MATH = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

tok = AutoTokenizer.from_pretrained('runs/sft1_bf16')
tok.chat_template = open('templates/gemma3.jinja').read()
gs = load_dataset('openai/gsm8k', 'main', split='train')
qs = list(gs['question'])[:400]
prompts = [tok.apply_chat_template([{"role": "user", "content": MATH.format(prompt=q)}],
                                   tokenize=False, add_generation_prompt=True) for q in qs]
llm = LLM(model='runs/sft1_bf16', gpu_memory_utilization=0.9, max_model_len=1536,
          dtype='bfloat16', max_num_seqs=512)
for temp in (1.0, 0.9):
    sp = SamplingParams(n=6, temperature=temp, top_p=0.95, top_k=64, max_tokens=400, seed=0, stop_token_ids=[1,106])
    t = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t
    fr, L = {}, []
    for o in outs:
        for c in o.outputs:
            fr[c.finish_reason] = fr.get(c.finish_reason, 0) + 1
            L.append(len(c.token_ids))
    print(f"TEMP {temp}: {dt:.0f}s finish={fr} meanlen={sum(L)/len(L):.0f} "
          f"tok/s={sum(L)/dt:.0f}", flush=True)
