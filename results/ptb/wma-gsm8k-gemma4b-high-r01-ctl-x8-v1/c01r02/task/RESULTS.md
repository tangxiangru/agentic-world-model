# GSM8K post-training of google/gemma-3-4b-pt

`final_model/` = `ckpts/exp-05/final`, verified from a cold process with `verify_final.py`.

| card | intervention | dev-150 | dev-500 |
|---|---|---:|---:|
| exp-01 | base snapshot cc012e0a, as shipped | 0.040 | - |
| exp-02 | completion-only SFT, 84k GSM8K-train-derived CoT | 0.593 | - |
| exp-03 | same weights, greedy `generation_config.json` | **0.753** | 0.706 |
| exp-04 | + 2nd SFT stage, disjoint 81k rows (rejected) | 0.740 | - |
| exp-05 | + RFT on 17.4k verified self-samples | **0.773** | **0.720** |
| - | `final_model/` re-scored from a cold process | 0.767 | - |

Every number: `python evaluate.py --model-path <p> --limit <n> --max-connections 16
--gpu-memory-utilization 0.85`, unmodified `evaluate.py` and `templates/`.

## Data provenance
GSM8K **train** split only, plus two public augmentations of it
(`meta-math/MetaMathQA` GSM_AnsAug/GSM_Rephrased, `nvidia/OpenMathInstruct-2`
problem_source gsm8k/augmented_gsm8k) and the model's own verified samples over the
train questions. `../contamination_check.py` run on every training file:
83760 / 81279 / 17373 documents scanned, **0 contaminated** in each.

## Files
- `build_data.py` -> `data/sft_v{2,3}.jsonl`; `gen_rft.py` + `clean_rft.py` -> `data/rft_v1_clean.jsonl`
- `train_sft.py` (completion-only loss, token+label budgeted batching), `run_eval.sh`, `analyze_eval.py`, `verify_final.py`
- `memory/cards/exp-01..06.yaml` — the experiment record; `memory/index.md` — the summary table
