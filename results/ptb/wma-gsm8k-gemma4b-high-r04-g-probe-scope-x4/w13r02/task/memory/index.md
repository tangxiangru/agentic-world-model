# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.0867 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.7 | sft | base_model | closed | yes | contradicted | iterate | accuracy=0.0733 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.3 | sft | exp-02 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6467 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 3.6 | decode-config | exp-03 | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-04-greedy |
| exp-05 | 4.3 | rft | exp-03 | closed | yes (re-locked 1x) | contradicted | adopt | accuracy=0.718 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.5 | decode-config | exp-05 | closed | yes | contradicted | reject | accuracy=0.71 | /home/ben/task/ckpts/exp-06-ckpt400-greedy |
| exp-07 | 7.1 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.6933 | /home/ben/task/ckpts/exp-07/final |
