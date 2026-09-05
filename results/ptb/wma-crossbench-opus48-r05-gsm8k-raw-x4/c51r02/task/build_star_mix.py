import json, random, re, collections
random.seed(1)

def clean_completion(comp, prompt_q=None):
    # cut at first ANSWER: line
    m = re.search(r"ANSWER:\s*([^\n]+)", comp)
    if not m: return None
    ans = m.group(1).strip()
    body = comp[:m.start()].rstrip()
    if len(body) < 5: return None
    return f"{body}\n\nANSWER: {ans}", ans

# ---- STaR cleaned, cap per question ----
byq = collections.defaultdict(list)
with open("star_raw.jsonl") as f:
    for line in f:
        r = json.loads(line)
        res = clean_completion(r["completion"])
        if res is None: continue
        comp, ans = res
        # sanity: body should not contain a second 'ANSWER:' or leaked 'model\n'
        if "ANSWER:" in comp[:comp.rfind("ANSWER:")]: continue
        byq[r["prompt"]].append(comp)

star_rows = []
CAP = 3
for p, comps in byq.items():
    # dedup, prefer shorter (cleaner) first then variety
    uniq = list(dict.fromkeys(comps))
    uniq.sort(key=len)
    take = uniq[:CAP]
    for c in take:
        star_rows.append({"prompt": p, "completion": c})
print("STaR cleaned rows:", len(star_rows), "from", len(byq), "questions")

# ---- reuse GSM8K gold zeroshot + fewshot + Orca from train_mix.jsonl ----
# train_mix already has: gsm zeroshot x2, gsm fewshot 5k, orca 55k
mix = [json.loads(l) for l in open("train_mix.jsonl")]
# keep gsm fewshot (long prompts) + a portion of orca + one copy of gsm zeroshot
# identify by heuristic: fewshot examples have very long prompts
def ptoklen(x): return len(x["prompt"])
gsm_fewshot = [r for r in mix if r["prompt"].count("Reasoning:") > 1]  # has fewshot blocks
orca_and_gsm = [r for r in mix if r["prompt"].count("Reasoning:") == 1]
print("fewshot in mix:", len(gsm_fewshot), "| single:", len(orca_and_gsm))

random.shuffle(orca_and_gsm)
# take all fewshot, plus 45k from single (mix of gsm gold + orca)
keep_single = orca_and_gsm[:45000]

out = star_rows + gsm_fewshot + keep_single
random.shuffle(out)
with open("train_star.jsonl","w") as f:
    for r in out:
        f.write(json.dumps(r)+"\n")
print("TOTAL:", len(out))
print("sample star completion:")
print(star_rows[0]["completion"][:300])
