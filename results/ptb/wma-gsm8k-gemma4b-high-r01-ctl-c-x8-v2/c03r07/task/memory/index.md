# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.04 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.62 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6133 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.42 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 4.72 | rft | exp-02 | closed | yes | contradicted | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.97 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.8067 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.65 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.7467 | /home/ben/task/ckpts/exp-06/final |
