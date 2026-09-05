#!/usr/bin/env python3
"""Probe: does the model assign probability to <end_of_turn> (106) right after 'ANSWER: N'?
Compare exp-01 (stops) vs exp-02 (rambles) on the same prompt+partial-completion."""
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
# partial completion up to and including the answer number
partial = "Cody eats 5 x 3 = <<5*3=15>>15 cookies.\nTogether, they eat 5 + 15 = <<5+15=20>>20 cookies.\n\nANSWER: 20"
cid = tok(partial, add_special_tokens=False).input_ids
ids = pid + cid

model = Gemma3ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                       attn_implementation="eager").cuda().eval()
with torch.no_grad():
    out = model(torch.tensor([ids]).cuda())
    logits = out.logits[0, -1]  # next-token distribution after "...ANSWER: 20"
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, 8)
    print(f"=== {model_path} ===")
    print(f"P(<end_of_turn>={eot}) = {probs[eot].item():.4f}")
    print("top-8 next tokens:")
    for p, t in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {t:>7} {repr(tok.decode([t])):>20}  p={p:.4f}")

    # Now greedily continue 40 tokens to see behavior
    gen = list(ids)
    for _ in range(40):
        o = model(torch.tensor([gen]).cuda())
        nt = int(o.logits[0, -1].argmax())
        gen.append(nt)
        if nt == eot or nt == 1:
            break
    print("greedy continuation after answer:", repr(tok.decode(gen[len(ids):])))
