# Progress notes (BFCL post-training of gemma-3-4b-pt)

## Key facts
- Base model snapshot: /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
- Eval: `bash eval_model.sh <model_dir> <limit> <max_conn>` (auto-copies preprocessor/processor json).
- GOTCHA: after training, MUST copy `preprocessor_config.json` + `processor_config.json` from base snapshot into the model dir, else vLLM fails (multimodal Gemma3 needs image processor). eval_model.sh does this automatically.
- Baseline (base model): accuracy 0.0.

## Data
- prep_data_v2.py -> train_v2.jsonl : 47,235 usable single-call examples from argilla/apigen-function-calling (xLAM+distilabel), python->JSON type conversion, with descriptions. Decon: 0 matches.
- toolace_prep.py -> train_toolace.jsonl : 2,071 single-call from Team-ACE/ToolACE. Decon: 0 matches.
- train_v4.jsonl = train_v2 + train_toolace = 49,610.

## Training (train.py): full FT, flash_attn2, paged_adamw_8bit, bf16, grad ckpt, completion-only loss, maxlen 1280, bs8 accum4 (eff 32), lr1e-5 cosine.

## Results
- v2 (1 epoch, train_v2): **accuracy 0.90** (full 100). SAVED as final_model.
- v3 (2 epochs, train_v2): accuracy 0.93. Promoted to final_model.

## Failure analysis of v2 (10 wrong):
- 2 unwinnable eval artifacts: estimate_derivative passes lambda arg; eval's ast.literal_eval drops it.
- ~4 optional-arg convention (sort_array reverse, investment adjust_for_inflation x2, book_room price).
- ~2 arg-order swap (math_lcm a/b).
- ~2 value interp (final_velocity sign, stock diffandsplits).
- Realistic ceiling ~96-98%.

## final_model = best so far (currently v3 @ 0.93, verified complete + all configs present).
- v4 (2 epochs, apigen+toolace 49.6k): training (~9% at last check, ETA ~2.5h). Chain PID 10421 -> eval_model.sh sft_v4 100 6 -> v4_chain.log / metrics_sft_v4_lim100.json.
- PLAN: after v4 eval, run larger-limit eval (300-400) on top candidate(s) to make robust final decision (100-sample eval has high variance).
