# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | decode-config | base_model | closed | yes | supported | adopt | accuracy=0.0733 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.25 | sft | base_model | closed | yes | contradicted | iterate | accuracy=0.06 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.9 | sft | exp-02 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6267 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.5 | decode-config | exp-03 | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-04-greedy |
| exp-05 | 5.5 | rft | exp-03 | closed | yes | contradicted | reject | accuracy=0.6733 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.4 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.714 | /home/ben/task/ckpts/exp-04-greedy |
| exp-07 | 7.0 | merge | exp-03 | closed | yes | contradicted | reject | accuracy=0.666 | /home/ben/task/ckpts/exp-07-soup |
| exp-08 | 7.5 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.706 | /home/ben/task/final_model |
