# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.08 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.3 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6067 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.0 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 3.05 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.7333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.65 | sft | exp-04 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.9 | rft | exp-05 | closed | yes | contradicted | reject | accuracy=0.7533 | /home/ben/task/ckpts/exp-06/final |
