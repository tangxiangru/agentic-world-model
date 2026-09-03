# Post-training google/gemma-3-4b-pt for GSM8K

`final_model/` holds the submission. Full experimental record: `memory/index.md`
and `memory/cards/exp-01..exp-08.yaml` (written before each run, closed after).

## Result

| model | test --limit 150 | test --limit 500 | train-holdout probe (n=300) |
|---|---|---|---|
| base gemma-3-4b-pt | 0.053 | - | - |
| exp-02 SFT (67k rows, 2 ep) | 0.713 | - | 0.730 |
| exp-03 SFT+RFT (185k rows, 1 ep) | 0.713 | 0.740 | 0.760 |
| exp-05 exp-03 + 0.5 ep | 0.693 | - | 0.743 |
| exp-06 soup(exp-02, exp-03) | 0.713 | 0.740 | 0.747 |
| **exp-08 soup(exp-03, exp-05) = final_model** | **0.713** | **0.748** | 0.757 |

Base 0.053 -> final 0.748 on 500 test items (stderr 0.019).

## What mattered

1. **Format and stopping, not reasoning, cost the base model almost everything.**
   The grader is `inspect_ai match(numeric=True, location="end")`: it reads the
   last numeric token of the completion. The base model answered, then imitated
   the 10-shot block by inventing further problems (95/150 completions had a
   second `ANSWER:`), so the graded number belonged to an invented problem.
   Fixing target shape + stop token took 0.053 -> 0.713.
2. **Two artifact settings the grader reads from the model directory.**
   - `tokenizer_config.json` `eos_token` -> `<end_of_turn>` (id 106), the token
     `templates/gemma3.jinja` terminates turns with and the one every training
     target ends with. The base model declares `<eos>` instead.
   - `generation_config.json` -> `{"do_sample": false, "temperature": 0.0}`.
     vLLM adopts the model's generation config as its server defaults; the base
     config's `top_k=64, top_p=0.95` means the grader would otherwise sample at
     T=1.0.
3. **Data**: OpenMathInstruct-2 rows with `problem_source in {gsm8k,
   augmented_gsm8k}` (GSM8K *train*-derived) + the GSM8K train split +
   rejection-sampled solutions from our own checkpoint. 300 train items were
   held out as a local probe; everything was passed through
   `../contamination_check.py` against the test copy (0 hits).
4. **Scaling the corpus past ~56k distinct problems, a second epoch, and
   cross-run weight averaging all landed inside the noise of exp-03.** The last
   gain came from trajectory averaging, and only through termination: on the 486
   of 500 items where both models stop, exp-03 and the soup are tied 367-366.

## Files

| file | role |
|---|---|
| `build_dev.py` | 300-item GSM8K-train holdout probe |
| `build_data.py` | SFT corpus builder (`data/sft_v1..v4.jsonl`) |
| `rft_sample.py` | rejection sampling from a checkpoint |
| `train_sft.py` | completion-only SFT; renders prompts with the grader's own jinja |
| `eval_local.py` | offline probe scorer (mimics the grader's matcher) |
| `make_soup.py`, `package_ckpt.py` | artifact assembly |

## Two ways a run here produces a clean-looking wrong answer

- gemma-3's 262k vocab makes the fp32 logit tensor ~11 GiB for one micro-batch;
  naive HF loss OOMs at bs=8. `liger_kernel` fused linear CE fixes it.
- A checkpoint saved with `do_sample=False, temperature=0.0` cannot be a
  *training parent*: HF's `GenerationConfig.save_pretrained` rejects that pair
  and every checkpoint save raises before the weights land. `train_sft.py` and
  `make_soup.py` install a valid generation config after loading and write the
  greedy one at the end. This cost one run (exp-04).
