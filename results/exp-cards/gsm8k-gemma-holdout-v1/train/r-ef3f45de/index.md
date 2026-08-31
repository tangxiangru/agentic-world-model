# r-ef3f45de - reconstructed experiment cards

Base model: Qwen/Qwen3-4B-Base | benchmark: gsm8k | budget: 10 h, one H100.
16 launches carded. The digest carries no timestamps ("block header: --- [event
index] turn=N act --- (no timestamps in this run)"), so every `elapsed_h` is
null; where the agent ran `bash timer.sh` the remaining-budget reading is quoted
in the card's `training_summary.notes` instead. The stream ends at event [552]
with the last training run at step 1000 of 3125 and unmerged.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 76 | null | sft | base_model | MetaMathQA GSM+MATH & gsm8k train, 402,317 | 2e-4 / 1 | killed | - | inconclusive | iterate |
| exp-02 | 82 | null | sft | base_model | MetaMathQA GSM+MATH & gsm8k train, 402,317 | 2e-4 / 1 | killed | - | inconclusive | iterate |
| exp-03 | 92 | null | sft | base_model | MetaMathQA GSM+MATH & gsm8k train, 402,317 | 2e-4 / 1 | killed | - | inconclusive | abandon_line |
| exp-04 | 133 | null | sft | base_model | MetaMathQA GSM & gsm8k train, 247,471 | 2e-4 / 1 | killed | - | inconclusive | abandon_line |
| exp-05 | 157 | null | sft | base_model | MetaMathQA GSM & gsm8k train, 247,471 | 2e-4 / 1 | failed (OOM) | - | inconclusive | abandon_line |
| exp-06 | 165 | null | sft | base_model | MetaMathQA GSM & gsm8k train, 247,471 (packed) | 2e-4 / 1 | completed | - (its merge measured on exp-07) | inconclusive | adopt |
| exp-07 | 217 | null | merge (v1 -> final_model) | exp-06 | - | - | completed | accuracy 0.080, n=50 | inconclusive | reject |
| exp-08 | 266 | null | sft | base_model | MetaMathQA GSM+MATH & gsm8k train, 100k of 402,317, EOS-terminated | 2e-4 / 1 | completed | - (its merge measured on exp-09) | inconclusive | adopt |
| exp-09 | 276 | null | merge (v2 -> final_model_v2) | exp-08 | - | - | completed | accuracy 0.040, n=50 (exp-07 0.080 same limit) | contradicted | reject |
| exp-10 | 331 | null | sft | base_model | MetaMathQA GSM & gsm8k train, 50k of 247,471, messages format, no packing | 2e-4 / 2 | completed | - (its merge measured on exp-11) | inconclusive | adopt |
| exp-11 | 360 | null | merge (v3 -> final_model) | exp-10 | - | - | completed | accuracy 0.020, n=50 (exp-09 0.040 same limit) | contradicted | reject |
| exp-12 | 388 | null | sft | base_model | MetaMathQA GSM & gsm8k train, 50k of 247,471, eval-shaped raw text | 2e-4 / 1 | completed | - (its merge measured on exp-13) | inconclusive | adopt |
| exp-13 | 403 | null | merge (v4 -> final_model) | exp-12 | - | - | completed | accuracy 0.040, n=50 (exp-11 0.020 same limit) | contradicted | reject |
| exp-14 | 425 | null | sft (+lm_head/embed unfrozen) | base_model | MetaMathQA GSM & gsm8k train, 20k of 247,471, eval-shaped raw text | 2e-4 / 1 | completed | - (its merge measured on exp-15) | inconclusive | adopt |
| exp-15 | 435 | null | merge (v5 -> final_model) | exp-14 | - | - | completed | accuracy 0.320, n=50 (exp-13 0.040 same limit) | supported | adopt |
| exp-16 | 519 | null | sft (+lm_head/embed unfrozen) | base_model | gsm8k train x5 with <<calc>> & MetaMathQA GSM, 50k of 277,363 | 1.5e-4 / 2 | killed | - | inconclusive | abandon_line |

Submission: exp-15 - final_model holding the merged trained_model_v5 adapter
(LoRA plus unfrozen lm_head and embed_tokens) with generation_config
eos_token_id [151643, 151645]. It is the last explicit state of final_model in
the stream and the only candidate above 0.08. exp-06, exp-08, exp-10, exp-12 and
exp-14 are marked adopt as the parents of the merge cards downstream of them.

The line of the run: the model could do the arithmetic from the first candidate
onward, but would not stop. Packing taught it to continue past <|im_end|>
(exp-07); adding <|endoftext|> to packed rows made it worse (exp-09); dropping
packing made it worse again because the messages path inserted a <think> block
the eval never emits (exp-11); matching the eval's raw text exactly still left it
emitting no stop token at all (exp-13), because LoRA leaves lm_head and
embed_tokens frozen. Unfreezing them (exp-14/exp-15) took 0.040 to 0.320 and made
the model terminate. The residual failure - a stray non-ASCII token glued to the
answer digits, which costs 20 of the 34 remaining errors - was what exp-16 was
launched to fix when the budget ran out.

Smoke tests: two, both on exp-10 - a 20-step no-packing speed probe at [322]
(no output recorded) and its rerun at [324] (~2.3 s/step).
