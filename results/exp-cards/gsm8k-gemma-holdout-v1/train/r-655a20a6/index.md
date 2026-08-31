# r-655a20a6 — gsm8k / HuggingFaceTB/SmolLM3-3B-Base / 10 h / 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 125 | null | sft (LoRA r=128) | base_model | train_split.jsonl (gsm8k train x2 + MetaMathQA GSM, 254,744) | 2e-4 / 1 | completed (~1.7 h) | accuracy 0.46 (n=50, eval_results_50.json) | inconclusive | adopt |
| exp-02 | 152 | null | sft (full FT) | base_model | train_split.jsonl (gsm8k train x2 + MetaMathQA GSM, 254,744) | 2e-5 / 2 | killed (stream ends inside the launch) | none | inconclusive | abandon_line |

Not cards: three relaunches of the same LoRA recipe that crashed at trainer construction
before any training step ([105] SFTConfig.max_seq_length renamed, [114] batched
formatting_func, [119] missing tokenizer.chat_template); they are recorded on exp-01 as
`provenance.smoke_runs`. Data preparation ([66], [95]) is recorded as
`setup.data[].build_command`, not as its own card.
