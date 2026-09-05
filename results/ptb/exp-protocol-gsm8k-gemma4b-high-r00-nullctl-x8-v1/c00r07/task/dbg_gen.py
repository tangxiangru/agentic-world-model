import os, json
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/home/ben/hf_cache/hub")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from gen_rft import PROMPT_TEMPLATE, extract_answer, norm_num

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()
    recs = [json.loads(l) for l in open("data/gsm8k_train.jsonl")][:200]
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=r["question"])}],
        tokenize=False, add_generation_prompt=True) for r in recs]
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    print("FIRST 6 IDS:", ids[0][:6])
    tp = [TokensPrompt(prompt_token_ids=i) for i in ids]

    llm = LLM(model="ckpt/sft1", gpu_memory_utilization=0.85, max_model_len=1280,
              enable_prefix_caching=False, dtype="bfloat16", max_num_seqs=512,
              disable_log_stats=True)

    import collections
    for name, prompts, deto in [("tokids_nodeto", tp, False),
                                ("tokids_deto", tp, True),
                                ("string_deto", texts, True)]:
        sp = SamplingParams(n=1, temperature=0.9, top_p=0.95, max_tokens=320,
                            seed=1234, detokenize=deto)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        fr = collections.Counter(c.finish_reason for o in outs for c in o.outputs)
        L = [len(c.token_ids) for o in outs for c in o.outputs]
        flat = [list(c.token_ids) for o in outs for c in o.outputs]
        dec = tok.batch_decode(flat, skip_special_tokens=True)
        ok = sum(1 for d, r in zip(dec, recs) if extract_answer(d) == norm_num(r["answer"]))
        print(f"== {name}: finish={dict(fr)} meanlen={sum(L)/len(L):.0f} acc={ok}/{len(recs)}")
        print("   sample:", repr(dec[0][:220]))


if __name__ == "__main__":
    main()
