# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.06 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.42 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.0 | decode-config | exp-02 | closed | yes | contradicted | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.35 | rft | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.7733 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 3.6 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.785 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 5.1 | rft | exp-05 | closed | yes | contradicted | iterate | accuracy=0.7867 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.25 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.78 | /home/ben/task/final_model |
