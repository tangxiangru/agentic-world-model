# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.04 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.37 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5533 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.35 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7133 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.75 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 3.4 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.68 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 5.75 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.6333 | /home/ben/task/ckpts/exp-06-soup3 |
| exp-07 | 5.9 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.706 | /home/ben/task/ckpts/exp-03-greedy |
| exp-08 | 6.0 | other | exp-04 | closed | yes | contradicted | adopt | accuracy=0.718 | /home/ben/task/ckpts/exp-05/final |
| exp-09 | 6.1 | decode-config | exp-05 | closed | yes | contradicted | reject | accuracy=0.724 | /home/ben/task/ckpts/exp-09-reppen |
| exp-10 | 6.15 | other | exp-05 | closed | yes | contradicted | adopt | accuracy=0.706 | /home/ben/task/final_model |
