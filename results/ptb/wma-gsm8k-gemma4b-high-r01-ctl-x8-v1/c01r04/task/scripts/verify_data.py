#!/usr/bin/env python3
"""CPU dry run: prove the training strings are byte-for-byte what the grader renders.

1. render one row through templates/gemma3.jinja (the file evaluate.py passes to vLLM)
   and compare with the row we built by hand;
2. every completion ends with the stop token the grader stops on;
3. 'ANSWER: ' appears exactly once and the last whitespace-word of the completion is
   the number the scorer will read (match(numeric=True, location='end'));
4. token length distribution vs max_seq_len.
"""
import json, sys, os, re, argparse
import numpy as np
from jinja2 import Environment
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_format import gemma_template, MATH_PROMPT_TEMPLATE, fewshot_prefix

from inspect_ai._util.text import strip_numeric_punctuation


def scorer_reads(completion_body: str) -> str:
    """Reimplementation of inspect_ai match(numeric=True, location='end')."""
    v = strip_numeric_punctuation(completion_body.strip().casefold())
    words = re.split(r"\s+", v)
    words.reverse()
    return next((w for w in words if w.replace(".", "").isnumeric()), words[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/ben/task/data/sft_train.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--model", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    a = ap.parse_args()

    tmpl_src, tmpl_hash = gemma_template()
    print(f"templates/gemma3.jinja sha256={tmpl_hash}")
    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    tmpl = env.from_string(tmpl_src)

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data)]
    print(f"{len(rows)} rows")

    # ---- 1. byte-for-byte template check on one 0-shot and one 10-shot row ----
    fs = fewshot_prefix()
    for want_fs in (False, True):
        r = next(x for x in rows if x["fewshot"] == want_fs)
        body = r["prompt"].split("<start_of_turn>user\n", 1)[1].rsplit("<end_of_turn>", 1)[0]
        if want_fs:
            question = body.split(MATH_PROMPT_TEMPLATE.split("{prompt}")[0].strip(), 1)[1]
        msgs = ([{"role": "system", "content": fs}] if want_fs else []) + [
            {"role": "user", "content": (body[len(fs) + 2:] if want_fs else body)}]
        rendered = tmpl.render(messages=msgs, add_generation_prompt=True, bos_token="<bos>")
        ours = "<bos>" + r["prompt"]
        assert rendered == ours, (
            f"MISMATCH fewshot={want_fs}\n--- jinja ---\n{rendered[-400:]!r}\n"
            f"--- ours ---\n{ours[-400:]!r}")
        print(f"  template match (fewshot={want_fs}): OK  ({len(rendered)} chars)")

    # ---- 2/3. stop token + answer marker ----
    bad_stop = bad_marker = bad_tail = 0
    for r in rows:
        c = r["completion"]
        if not c.endswith("<end_of_turn>\n"):
            bad_stop += 1
        body = c.rsplit("<end_of_turn>", 1)[0]
        if body.count("ANSWER: ") != 1:
            bad_marker += 1
        gold = body.rsplit("ANSWER: ", 1)[1].strip()
        if scorer_reads(body) != strip_numeric_punctuation(gold.casefold()):
            bad_tail += 1
    print(f"  completions not ending in <end_of_turn>: {bad_stop}")
    print(f"  completions without exactly one 'ANSWER: ': {bad_marker}")
    print(f"  completions where the scorer would read something other than the gold: {bad_tail}")

    # ---- 4. lengths ----
    n = min(4000, len(rows))
    lens = []
    for r in rows[:n]:
        lens.append(len(tok(r["prompt"]).input_ids) +
                    len(tok(r["completion"], add_special_tokens=False).input_ids))
    lens = np.array(lens)
    over = (lens > a.max_seq_len).mean()
    print(f"  tokens (first {n}): p50={np.percentile(lens,50):.0f} "
          f"p95={np.percentile(lens,95):.0f} p99={np.percentile(lens,99):.0f} max={lens.max()}")
    print(f"  share over max_seq_len={a.max_seq_len}: {over:.4%}")
    ok = (bad_stop == 0 and bad_marker == 0 and bad_tail == 0 and over <= 0.02)
    print("VERDICT:", "OK" if ok else "PROBLEM")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
