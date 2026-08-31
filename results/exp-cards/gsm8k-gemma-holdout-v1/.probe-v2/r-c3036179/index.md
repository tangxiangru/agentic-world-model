# Reconstructed experiment cards

One card per launch, in launch order. Base model post-trained: `Qwen/Qwen3-1.7B-Base`;
benchmark gsm8k; budget 10 h on one H100. `elapsed_h` is hours since the run started,
`launch_i` is the digest event index carrying `setup.command.argv`.

| exp-NN | launch_i | elapsed_h | family        | parent     | datasets                                   | lr/epochs                 | execution | best measurement | verdict      | decision     |
|--------|----------|-----------|---------------|------------|--------------------------------------------|---------------------------|-----------|------------------|--------------|--------------|
| exp-01 | [59]     | 0.11      | sft           | base_model | 20 synth (short prompts)                   | 1e-5 / 1ep (max_steps 2)  | completed | -                | inconclusive | abandon_line |
| exp-02 | [66]     | 0.13      | sft           | base_model | gsm8k-train x2 + 12k synth (short prompts) | 2e-5 / 2ep                | completed | 0.1600 n=150     | contradicted | reject       |
| exp-03 | [125]    | 0.54      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 1e-5 / 2ep                | completed | 0.4267 n=150     | supported    | adopt        |
| exp-04 | [164]    | 1.29      | sft           | exp-03     | gsm8k-train (fixed 10-shot)                | 5e-6 / 2ep                | completed | 0.4533 n=150     | supported    | adopt        |
| exp-05 | [230]    | 2.08      | sft           | exp-04     | gsm8k-train (fixed 10-shot) + 6k synth     | 2e-6 / 1ep                | completed | 0.4467 n=150     | contradicted | reject       |
| exp-06 | [274]    | 2.68      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 2e-5 / 2ep                | completed | 0.6854 full      | inconclusive | adopt        |
| exp-07 | [317]    | 3.34      | sft           | exp-06     | gsm8k-train (fixed 10-shot)                | 5e-6 / 2ep                | completed | 0.6133 n=150     | inconclusive | adopt        |
| exp-08 | [362]    | 4.01      | sft           | exp-07     | gsm8k-train (fixed 10-shot)                | 2e-6 / 2ep                | completed | 0.5400 n=150     | supported    | adopt        |
| exp-09 | [411]    | 4.69      | sft           | exp-08     | gsm8k-train (fixed 10-shot)                | 1e-6 / 2ep                | completed | 0.6267 n=150     | contradicted | reject       |
| exp-10 | [457]    | 5.35      | sft           | exp-08     | gsm8k-train (fixed 10-shot)                | 1e-6 / 1ep                | completed | 0.5000 n=150     | contradicted | reject       |
| exp-11 | [490]    | 5.72      | other         | exp-08     | -                                          | -                         | completed | 0.5600 n=150     | supported    | adopt        |
| exp-12 | [500]    | 5.76      | decode-config | exp-11     | -                                          | -                         | completed | 0.6333 n=150     | supported    | adopt        |
| exp-13 | [507]    | 5.79      | decode-config | exp-12     | -                                          | -                         | completed | 0.4867 n=150     | contradicted | reject       |
| exp-14 | [513]    | 5.81      | decode-config | exp-13     | -                                          | -                         | completed | 0.6467 n=150     | inconclusive | adopt        |
| exp-15 | [554]    | 6.03      | other         | exp-06     | -                                          | -                         | completed | 0.6467 n=150     | inconclusive | adopt        |
| exp-16 | [599]    | 6.09      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 2e-5 / 1ep                | killed    | -                | inconclusive | abandon_line |
| exp-17 | [612]    | 6.13      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 2e-5 / 1ep                | completed | 0.7028 full      | supported    | iterate      |
| exp-18 | [661]    | 6.58      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 2e-5 / 2ep                | completed | 0.6725 full      | contradicted | reject       |
| exp-19 | [737]    | 7.45      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 2e-5 / 1ep (stop @234)    | completed | 0.6846 full      | contradicted | reject       |
| exp-20 | [754]    | 7.71      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 2e-5 / 1ep (stop @234)    | completed | 0.6869 full      | contradicted | reject       |
| exp-21 | [800]    | 7.99      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 2e-5 / 1ep (stop @234)    | completed | 0.6267 n=150     | contradicted | reject       |
| exp-22 | [832]    | 8.19      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 1.5e-5 / 1ep (stop @234)  | completed | 0.7051 full      | supported    | adopt        |
| exp-23 | [858]    | 8.45      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 1.25e-5 / 1ep (stop @234) | completed | 0.7005 full      | contradicted | reject       |
| exp-24 | [882]    | 8.71      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 1.5e-5 / 1ep (stop @260)  | completed | 0.6467 n=150     | contradicted | reject       |
| exp-25 | [924]    | 8.95      | sft           | base_model | gsm8k-train (fixed 10-shot)                | 1.75e-5 / 1ep (stop @234) | completed | 0.6800 n=150     | contradicted | reject       |
| exp-26 | [947]    | 9.15      | other         | exp-22     | -                                          | -                         | completed | 0.6960 full      | contradicted | adopt        |
| exp-27 | [986]    | 9.31      | decode-config | exp-26     | -                                          | -                         | completed | 0.7005 full      | supported    | iterate      |

Submitted model: **exp-26** - `final_model` packaged from exp-22's weights
(`run_fixed_lr15e6_mid_seed1234`, 1.5e-5 stopped at step 234 of a one-epoch cosine
schedule, full-set 0.7051). exp-27 then tuned only the packaged generation cap.
