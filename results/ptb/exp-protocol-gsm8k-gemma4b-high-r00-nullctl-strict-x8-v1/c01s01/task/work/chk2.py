from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("work/rft1_bf16")
print("tok eos:", tok.eos_token, tok.eos_token_id, "pad:", tok.pad_token_id)
llm = LLM(model="work/rft1_bf16", gpu_memory_utilization=0.5, max_model_len=1024)
p = ("<start_of_turn>user\nSolve the following math problem step by step. The last line of your "
     "response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer "
     "to the problem.\n\nJohn has 3 apples and buys 5 more. How many does he have?\n\nRemember to put "
     "your answer on its own line at the end in the form \"ANSWER: $ANSWER\" (without quotes) where "
     "$ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
     "Reasoning:<end_of_turn>\n<start_of_turn>model\n")
for name, sp in [("greedy", SamplingParams(temperature=0.0, max_tokens=256)),
                 ("greedy+stop106", SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[106]))]:
    o = llm.generate([p], sp)
    c = o[0].outputs[0]
    print("===", name, "finish:", c.finish_reason, "ntok:", len(c.token_ids), "last:", c.token_ids[-3:])
    print(repr(c.text))
