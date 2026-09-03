# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | decode-config | base_model | closed | yes | supported | adopt | accuracy=0.06 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 1.0 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.3 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 3.6 | rft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.68 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.7 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.712 | /home/ben/task/ckpts/exp-04/final |
| exp-06 | 5.9 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.718 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.9 | merge | exp-04 | closed | yes | inconclusive | reject |  | /home/ben/task/ckpts/exp-04/final |
| exp-08 | 8.0 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.714 | /home/ben/task/final_model |
