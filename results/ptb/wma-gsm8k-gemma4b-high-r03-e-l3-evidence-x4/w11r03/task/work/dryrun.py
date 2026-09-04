"""CPU dry run: render training rows exactly as the grader renders, check the
stop token, the answer marker, the loss mask and the length distribution."""
import json, random, numpy as np, sys
from transformers import AutoTokenizer
sys.path.insert(0, "work")
from train_sft import build_examples, BASE, TPL, STOP

tok = AutoTokenizer.from_pretrained(BASE)
tpl = open(TPL).read()
rows = [json.loads(l) for l in open("data/sft_train.jsonl")]
random.Random(0).shuffle(rows)
samp = rows[:8000]
ex, dropped = build_examples(tok, tpl, samp, 1536)
lens = np.array([len(e["input_ids"]) for e in ex])
print(f"sample={len(samp)} kept={len(ex)} dropped@1536={dropped} ({dropped/len(samp):.3%})")
for p in (50, 90, 99, 99.9):
    print(f"  p{p}: {int(np.percentile(lens,p))}")
print("  max:", lens.max(), " mean:", int(lens.mean()))
# marker / stop invariants over the WHOLE pool
bad_stop = bad_mark = 0
for r in rows:
    if not r["target"].endswith(STOP): bad_stop += 1
    if r["target"].count("ANSWER: ") != 1: bad_mark += 1
print("pool rows:", len(rows), "bad_stop:", bad_stop, "bad_marker:", bad_mark)

e = ex[0]
lab = [t for t in e["labels"] if t != -100]
print("=== decoded prompt (masked part) ===")
print(repr(tok.decode([i for i, l in zip(e["input_ids"], e["labels"]) if l == -100])[-400:]))
print("=== decoded target (loss part) ===")
print(repr(tok.decode(lab)))
print("last label token id:", lab[-1], tok.convert_ids_to_tokens([lab[-1]]))
print("eos ids in generation_config:", [1, 106], "-> <end_of_turn> is 106:", tok.convert_tokens_to_ids("<end_of_turn>"))
