import torch, json
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("sft_gsm8k_v1")
model = Gemma3ForConditionalGeneration.from_pretrained("sft_gsm8k_v1", dtype=torch.bfloat16).cuda().eval()

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

# build fewshot system text like inspect does
tr = load_dataset("openai/gsm8k","main",split="train")
def s2f(q,a):
    DELIM="####"; parts=a.split(DELIM); target=parts.pop().strip(); reasoning=DELIM.join(parts).strip()
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"
import random
random.seed(42)
idx=random.sample(range(len(tr)),10)
fewshot="\n\n".join(s2f(tr[i]['question'],tr[i]['answer']) for i in idx)

q="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

for mode in ["zeroshot","fewshot"]:
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    if mode=="fewshot":
        user = fewshot + "\n\n" + user
    prompt = f"{tok.bos_token}<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, max_new_tokens=300, do_sample=False)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    print(f"===== {mode} ({ids.shape[1]} prompt tokens) =====")
    print(gen)
    print()
