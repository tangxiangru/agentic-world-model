# PTB batch: gemma-3-4b-pt -> GSM8K

## What shipped
`final_model/` = exp-06 weights (exp-02 -> exp-04 -> exp-06 chain), bf16,
`generation_config.json` with `do_sample: false, temperature: 0.0, eos_token_id: [1, 106]`.

| card | what | dev-150 | dev-500 | full 1319 |
|------|------|---------|---------|-----------|
| exp-01 | base checkpoint, inherited sampling | 0.0867 | - | - |
| exp-02 | SFT 120k OpenMathInstruct-2 gsm8k rows, sampled decode | 0.7000 | - | - |
| exp-03 | same weights, greedy decode config | 0.7467 | - | - |
| exp-04 | + 200k rows (140k OMI-2 fresh + 60k MetaMathQA GSM) | 0.8133 | 0.7920 | 0.7877 |
| exp-05 | + 130k further distilled rows | 0.8067 | 0.7920 | - |
| exp-06 | + RFT on 13.4k own verified solutions (shipped) | 0.8067 | 0.7960 | 0.7968 |

## The three things that actually mattered
1. **Match the grader's string exactly.** `scripts/render_sft.py` renders through a
   sha256-pinned copy of `templates/gemma3.jinja` and every target ends with
   `<end_of_turn>` (token 106), which is what vLLM stops on. Termination alone:
   75/150 completions hit max_tokens before, 0/150 after.
2. **One answer marker, and it must be last.** The scorer is
   `match(location="end", numeric=True)` - it reverses the completion and takes the
   first numeric word, i.e. the LAST number anywhere in the output.
3. **`generation_config.json` is the served decode policy.** vLLM's
   `--generation-config` defaults to `auto`, so the base model's `do_sample: true,
   top_k: 64, top_p: 0.95` was being applied at grading. Writing `temperature: 0.0`
   was +4.7 points for zero training and made every later eval deterministic.

## What was flat
exp-05 and exp-06 both. The OpenMathInstruct-2 gsm8k + augmented_gsm8k **problem**
pool is only ~100k problems; rounds one and two consumed it, and after that more rows
are more solutions to problems already covered. Training loss kept falling
(0.280 -> 0.220 -> 0.192 -> 0.182) while accuracy did not move.

## For a next batch
- Start from a larger problem pool, or generate new problems rather than new solutions.
- The exp-06 sampling produced ready-made preference pairs (1029 problems where all 4
  samples were wrong, 7306 mixed) and RFT threw the wrong ones away - DPO would use them.
- Save intermediate checkpoints (`--save-steps`): every training card here kept only
  its final, so no checkpoint sweep was ever possible. The WMA flagged this twice.

## Correction found during final verification
Greedy decoding is deterministic *given identical batching*, not across invocations.
`final_model` scored 0.8067 (121/150) at `--max-connections 16 --gpu-memory-utilization
0.85` and 0.8000 (120/150) under the bare `python evaluate.py --limit 150` defaults
(max-connections 2, gpu-memory-utilization 0.3) - the same weights and the same
temperature-0 policy. Different batch shapes change kernel reduction order and can flip
an argmax on a near-tie. So the "stderr collapses to 0" claim in exp-03 is too strong:
there is ~1 item of numerical jitter at n=150 even at temperature 0. This is also why
the exp-04-vs-exp-06 one-item gap at n=150 was not worth weighing against the 12-item
gap at n=1319 - it was inside the jitter.

## Verified final state
`python evaluate.py --limit 150` (bare defaults, fresh process, model dir `final_model`):
accuracy 0.8000, 150/150 stop at `<end_of_turn>`, 0 items missing an ANSWER line,
mean completion 149 tokens. Log: `eval/final_model_default_dev150_full.json`.
