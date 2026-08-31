# Reconstructed experiment cards — r-c2a5a7bb

Base model post-trained: `Qwen/Qwen3-4B-Base` · benchmark: gsm8k · budget: 10 h, 1x H100.
12 launches, in launch order. Adopted / submitted: **exp-11** (`final_model`, a copy of the exp-10 checkpoint).

`elapsed_h` is empty on every card: this digest carries no turn timestamps (`# block header: --- [event index] turn=N act --- (no timestamps in this run)`).
The stream does carry five `timer.sh` readings of *remaining* budget — 6:52 [160], 4:40 [208], 4:15 [231], 2:39 [246], 2:36 [258] — which are noted on the cards they bracket but not converted into `elapsed_h`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 48 | null | sft (LoRA r=64) | base_model | train_data.jsonl (27,463 = 7,463 gsm8k gold + 20k MetaMathQA GSM) | 2e-4 / 2 | failed | — (crashed before step 1) | inconclusive | iterate |
| exp-02 | 72 | null | sft (LoRA r=64) | base_model | train_data.jsonl (27,463) | 2e-4 / 2 | completed | 0.06 @ limit 50 | inconclusive | reject |
| exp-03 | 139 | null | sft (LoRA r=64) | base_model | train_data.jsonl (27,463) | 2e-4 / 3 | killed | — (died at ~3,870/5,151 steps, never saved) | inconclusive | abandon_line |
| exp-04 | 165 | null | sft (LoRA r=64) | base_model | train_data_v2.jsonl (22,463 = 7,463 gold + 15k MetaMathQA GSM, raw `<\|im_start\|>` text) | 2e-4 / 2 | completed | 0.04 @ limit 50 | contradicted | reject |
| exp-05 | 197 | null | sft (full FT) | base_model | train_data_v2.jsonl (22,463) | 2e-5 / 2 | failed | — (crashed in the first batch) | inconclusive | iterate |
| exp-06 | 202 | null | sft (full FT) | base_model | train_data_v2.jsonl (22,463) | 2e-5 / 2 | completed | 0.60 @ limit 50 | supported | adopt |
| exp-07 | 227 | null | sft (full FT) | base_model | train_data_v3.jsonl (52,389 = gold x3 + 30k MetaMathQA GSM) | 2e-5 / 3 | killed | — (died at ~1,113/9,825 steps, ~9 h projected) | inconclusive | iterate |
| exp-08 | 240 | null | sft (continuation) | exp-06 | train_data_v3.jsonl (52,389) | 1e-5 / 2 | completed | 0.64 @ limit 50 | inconclusive | adopt |
| exp-09 | 254 | null | other (packaging) | exp-08 | — | — | completed | 0.66 @ limit 150 | inconclusive | adopt |
| exp-10 | 284 | null | sft (continuation) | exp-08 | train_data_v4.jsonl (57,315 = gold x5 + 20k clean MetaMathQA GSM) | 5e-6 / 1 | completed | 0.70 @ limit 50 · 0.693 @ limit 150 | inconclusive | adopt |
| exp-11 | 303 | null | other (packaging) | exp-10 | — | — | completed | — (final_model never evaluated after the copy) | inconclusive | adopt |
| exp-12 | 341 | null | sft (continuation) | exp-10 | train_data_v4.jsonl (57,315) | 3e-6 / 2 | killed | — (last event in the digest; no result) | inconclusive | abandon_line |

## Reading the run

The run is one long fight with the serving format, then three small data iterations.
exp-01 through exp-04 are all LoRA at 2e-4 and all fail the same way: the training text and the
prompt the evaluator actually sends disagree over the Qwen3 template's `<think>` block, and the
merged checkpoints have no stop token at `<|im_end|>`, so they run past the answer into invented
problems — 0.06, then 0.04, with 102k and 195k output tokens for 50 answers.
exp-06 is the break: same data, full-parameter fine-tuning at 2e-5, greedy decoding and
`eos_token_id [151643, 151645]` — 0.60 at limit 50 and output tokens down to 9.3k.
Everything after it is continuation training on progressively more GSM8K-weighted mixtures
(x3, then x5) at 1e-5, 5e-6 and 3e-6.

Five cards carry `adopt`. exp-09 and exp-11 are the two successive writes to `final_model`, and
exp-06, exp-08 and exp-10 are the training cards whose checkpoints they package or parent. Only
exp-11 survives to the end of the record.

## Comparators and noise

No baseline was ever measured on the base model, under any protocol, so exp-01 and exp-02 have no
comparator. exp-04, exp-06 and exp-08 are each compared with their predecessor at limit 50, and
exp-10 with exp-09 at limit 150. Only exp-06's delta (+0.56) is outside noise; exp-08's +4.0 points
sits inside a 6.9-point standard error, and exp-10's +6.0 (limit 50) and +3.3 (limit 150) inside
6.5 and 3.8 points respectively. Hence one `supported`, one `contradicted` (exp-04, which measured
lower than the checkpoint it was meant to beat) and ten `inconclusive`.

## Open questions

- The digest ends on the exp-12 launch, so it is not settled whether `final_model_v7` was ever
  written or copied over `final_model`. exp-11 is the last write to the submission directory that
  the record shows.
- No training loss, step count or runtime survives for exp-02, exp-04, exp-08 or exp-10: every
  trainer's console output was truncated for size, and the figures on those cards are the agent's
  own readings.
- `train_v5.py` in the workspace is the [235] rewrite (continuation from `final_model`), not the
  from-scratch text exp-07 ran; likewise `train.py` is the [68] rewrite, not what exp-01 ran. Both
  launched texts survive only in the digest.
- `prepare_data_v5.py` is in the workspace but was never run: its few-shot system message is 1,683
  tokens against a 768-token training window [337]/[338].
- Event indices are not unique in this digest: turns 118–131 and turn 141 restart the counter at low
  numbers. Every launch cited here is in the main sequence; the exp-12 failure-analysis locators name
  the turn as well as the index.
- The workspace holds a contamination judgement reading `contamination detected` and an
  allowed-model judgement reading `only allowed use detected`. Neither is produced or discussed
  anywhere in the digest, and the agent ran no contamination check of its own, so every card records
  `contamination_check: not_run`.

No smoke runs or dry runs appear in this digest; `provenance.smoke_runs` is empty on all 12 cards.
The two crash cards (exp-01, exp-05) are full launches that died on library-API mismatches, not
deliberately truncated pipeline checks.
