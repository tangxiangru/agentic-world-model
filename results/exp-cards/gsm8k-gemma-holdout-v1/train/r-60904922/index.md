# r-60904922 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 176 | null | sft | base_model | data/gsm8k_train_sft.jsonl (6961, openai/gsm8k train) | 5e-5 / 1.0 | completed | none (mean train_loss 0.4365) | inconclusive | adopt |
| exp-02 | 201 | null | merge | exp-01 | - | - / - | completed | accuracy 0.16 (n=50, agent's local scorer) | inconclusive | reject |
| exp-03 | 226 | null | merge | exp-01 | - | - / - | completed | accuracy 0.350 (n=20, official; base 0.350) | supported | reject |
| exp-04 | 253 | null | sft | base_model | data/gsm8k_benchmark_train_sft.jsonl (6961, 10-shot benchmark format) | 3e-5 / 0.5 | completed | none (mean train_loss 0.4431) | inconclusive | adopt |
| exp-05 | 323 | null | merge | exp-04 | - | - / - | completed | accuracy 0.350 (n=20, official; base 0.350) | contradicted | reject |
| exp-06 | 393 | null | rft | base_model | data/rft_train_1000.jsonl (719 self-generated correct traces, keep_rate 0.719) | 3e-5 / 2.0 | completed | none (no trainer summary in the stream) | inconclusive | adopt |
| exp-07 | 414 | null | merge | exp-06 | - | - / - | completed | accuracy 0.51 (n=100, official; base 0.51) | contradicted | adopt |
| exp-08 | 443 | null | other (packaging) | exp-07 | - | - / - | completed | none (final 20-sample eval printed no score) | inconclusive | adopt |

exp-08 is the submission: `/home/ben/task/final_model`, a copy of the merged RFT
checkpoint from exp-07.

Smoke runs (not cards): [111] and [118] prepare_data.py crashes on TinyGSM
program rendering and [147] a 0.2-epoch smoke train, all recorded on exp-01;
[362] a 200-prompt pilot of the trace generator, recorded on exp-06.

The digest header states this run recorded no event timestamps, so every
`elapsed_h` is null; the only clock readings are timer.sh at [9] (9:59 of 10 h
remaining) and [267] (9:15 remaining).

Three candidates were built (plain GSM8K SFT, benchmark-format SFT, RFT on
self-generated correct traces) and all three tied the base model under the
official evaluator — 0.350 on the 20-sample slice and 0.51 on the 100-sample
slice. The workspace snapshot holds none of the scripts or eval outputs the
cards cite, so no hyper-parameter comes from an argparse default and no cited
json could be read back.
