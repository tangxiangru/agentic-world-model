# Experiment index

Task: post-train google/gemma-3-4b-pt for GSM8K. Grader: inspect_evals/gsm8k, 10-shot system
message, `match(numeric=True)` on a final "ANSWER: $N" line, rendered with templates/gemma3.jinja
(stop token `<end_of_turn>`). Protocol for every number below: `evaluate.py --limit 150
--max-connections 8 --gpu-memory-utilization 0.35`.

| card | intervention | vs | accuracy @n=150 | verdict | decision |
|------|--------------|----|-----------------|---------|----------|
| — | base_model (untuned snapshot cc012e0a) | — | 0.0400 | — | comparator, eval_base.json |
| exp-01 | LoRA r=32 SFT, GSM8K train split, grader-exact rendering, 1 epoch | base_model | 0.3267 (+0.287) | supported | adopt |
| exp-02 | same recipe, ~2.8 epochs (11-min wall cap) | exp-01 | 0.4533 (+0.127) | supported | adopt → final_model |

Submitted: `final_model/` = exp-02's merged checkpoint (Gemma3ForConditionalGeneration + tokenizer
+ processor; generation_config eos_token_id [1, 106] so vLLM stops on `<end_of_turn>`).

## Starting points for the next agent
- The curve had not turned over at ~2.8 epochs (loss 0.49 → 0.31, score still rising). 4–6 epochs
  is the cheapest untested lever.
- RFT: sample rationales from exp-02, keep the ones whose answer is right, retrain. Now possible
  because exp-02 follows the format; it was not possible before exp-01 existed.
- n=150 has stderr ≈ 0.04. It cannot resolve anything smaller than ~8 points. Raise n before
  trusting a small delta.
- Data build is `build_data.py` → `train_data.jsonl` (7463 rows, GSM8K **train** only, the
  grader's 10 few-shot items removed, contamination check passed, 0 matches).

## Protocol debt (read this)
`awm exp_protocol lock` never returned a success in this session. exp-01's first lock attempt
failed two real checks — `stop_token_consistent` and `comparator_same_protocol` — and both were
fixed (the stop token was moved into the `completion` field; the base model was scored at n=150).
Before the lock could be re-run, `/home/ben/awm` was lost to a stale NFS file handle and the CLI
stayed unavailable for the rest of the session, so exp-01 and exp-02 are unlocked and unclosed by
the tool. Both cards had sections 0–4 written before their launch command ran, exp-01 passed
`check`, and the checks the CLI would have run were verified by hand — including the
`template_unreachable` pitfall, where the training prefix was compared byte-for-byte against
`tok.apply_chat_template(..., chat_template=templates/gemma3.jinja)` on a system-prompt row (MATCH).
Treat the two cards as trustworthy in content but unverified by the tool.

## Pitfalls hit (add to skills/exp_protocol/pitfalls.yaml if they recur)
- Gemma-3's 262k vocab makes `logits.float()` the memory bottleneck, not activations: 8192
  tokens/batch OOM'd on an 80 GB H100 asking for 7.8 GiB in one allocation. 3072 tokens/batch +
  grad accum 3 + gradient checkpointing fixed it. Cost ~3 min.
- A training run launched with `nohup ... &` inside an agent shell dies when that tool call times
  out — it took the process group with it, twice, silently leaving a stale log. `setsid` plus a
  properly detached background call fixed it. Cost ~8 min, the single most expensive mistake here.
