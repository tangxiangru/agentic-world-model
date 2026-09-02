# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | adopt | accuracy=0.04 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.5933 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 2.75 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.74 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.6 | rft | exp-04 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7733 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.2 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.7667 | /home/ben/task/final_model |
