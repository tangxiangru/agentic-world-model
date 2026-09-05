from vllm import LLM, SamplingParams
llm = LLM(model="work/rft1_bf16", gpu_memory_utilization=0.5, max_model_len=1024)
p = ("<start_of_turn>user\nSolve the following math problem step by step. The last line of your "
     "response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer "
     "to the problem.\n\nJohn has 3 apples and buys 5 more. How many does he have?\n\nRemember to put "
     "your answer on its own line at the end in the form \"ANSWER: $ANSWER\" (without quotes) where "
     "$ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
     "Reasoning:<end_of_turn>\n<start_of_turn>model\n")
sp = SamplingParams(n=2, temperature=1.0, max_tokens=256, logprobs=0)
o = llm.generate([p], sp)
for c in o[0].outputs:
    print("finish:", c.finish_reason, "last ids:", c.token_ids[-4:], "stop_reason", c.stop_reason)
    print(repr(c.text[-80:]))
print("prompt ids head:", o[0].prompt_token_ids[:6])
