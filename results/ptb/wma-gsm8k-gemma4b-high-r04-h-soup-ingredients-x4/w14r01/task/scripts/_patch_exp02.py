p = "memory/cards/exp-02.yaml"
s = open(p).read()
s = s.replace(
    "--max-seq-len, '2304', --seed, '0']",
    "--max-seq-len, '2304', --seed, '0', --save-strategy, steps, --save-steps, '1085']",
)
s = s.replace(
    "  checkpoints:\n    every_steps: null\n    keep: all",
    "  checkpoints:\n    every_steps: 1085\n    keep: all",
)
s = s.replace(
    '    - option: "Change the decode arm in this card',
    '    - option: "save_strategy=epoch (2 checkpoints) as first written"\n'
    '      reason: "WMA cheaper_variant: checkpoint selection (C5) is worth up to +/-15 pp and the epoch-1 point of a 2-epoch cosine schedule is not a natural stopping point. --save-steps 1085 gives checkpoints at 0.5/1.0/1.5/2.0 epochs plus final at the same training cost; save_only_model=True keeps each to 8.6 GB instead of ~40 GB."\n'
    '    - option: "Change the decode arm in this card',
)
extra = (
    "  smoke_runs:\n"
    "    - command: [python, scripts/train_sft.py, --model, base_snapshot, --data, data/sft_omi.jsonl, --out, ckpts/_smoke2, --limit, '320', --epochs, '1', --bs, '16', --grad-accum, '2', --save-strategy, steps, --save-steps, '4']\n"
    '      outcome: "WMA tier-2 precondition: exercises the mid-training Trainer checkpoint path (the first smoke used --save-strategy no) together with the new first-batch label-masking assertion. Reported: SMOKE2_OUTCOME"\n'
    '    - command: ["in-script assertion in scripts/train_sft.py, after Trainer construction"]\n'
    "      outcome: \"WMA tier-1 precondition: the first collated batch is read back and asserted - supervised span contiguous, prompt span fully -100, last supervised label == token 106 <end_of_turn>, and the six tokens before the span decode to '<start_of_turn>model' + newline.\"\n"
    '    - command: ["in-script assertion in scripts/train_sft.py, after saving"]\n'
    '      outcome: "WMA tier-1 precondition: every checkpoint-*/ and final/ receives the base tokenizer, processor and generation_config, then asserts 106 in eos_token_id and that top_k/top_p are present and non-null, so any mid-run checkpoint is loadable by vLLM on exactly the exp-01 decode arm."\n'
)
s = s.replace("  smoke_runs:\n", extra, 1)
open(p, "w").write(s)
print("ok")
