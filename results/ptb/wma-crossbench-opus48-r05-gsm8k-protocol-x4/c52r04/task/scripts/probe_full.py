#!/usr/bin/env python3
"""Full greedy generation from the prompt on both models, token-level."""
import sys, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

model_path = sys.argv[1]
template = open("templates/gemma3.jinja").read()
tok = AutoTokenizer.from_pretrained(model_path)
eot = tok.convert_tokens_to_ids("<end_of_turn>")

PROMPT = ('Solve the following math problem step by step. The last line of your response '
          'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
          'answer to the problem.\n\nCody and his friend each eat 5 cookies. Cody eats 3 '
          'times as many as his friend. How many cookies do they eat together?\n\nRemember '
          'to put your answer on its own line at the end in the form "ANSWER: $ANSWER" '
          '(without quotes) where $ANSWER is the answer to the problem, and you do not need '
          'to use a \\boxed command.\n\nReasoning:')

pid = tok.apply_chat_template([{"role": "user", "content": PROMPT}], chat_template=template,
                              add_generation_prompt=True, tokenize=True)
model = Gemma3ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                       attn_implementation="eager").cuda().eval()
gen = list(pid)
with torch.no_grad():
    for _ in range(300):
        o = model(torch.tensor([gen]).cuda())
        nt = int(o.logits[0, -1].argmax())
        gen.append(nt)
        if nt == eot or nt == 1:
            break
text = tok.decode(gen[len(pid):])
print(f"=== {model_path} ===")
print(f"stopped_at_eot={gen[-1]==eot} n_new_tokens={len(gen)-len(pid)}")
print(repr(text[:600]))
