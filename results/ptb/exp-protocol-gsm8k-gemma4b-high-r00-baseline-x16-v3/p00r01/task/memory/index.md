# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.045 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.58 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 4.1 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.57 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 6.3 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.57 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 8.1 | merge | exp-03 | closed | yes | contradicted | reject | accuracy=0.6073 | /home/ben/task/ckpts/exp-05 |
