# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes | supported | adopt | accuracy=0.0333 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.0 | sft | base_model | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.68 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.9 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.72 | /home/ben/task/ckpts/soup23 |
| exp-05 | 6.1 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 8.0 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/final_model |
| exp-07 | 8.5 | other | exp-04 | closed | yes | contradicted | reject | accuracy=0.706 | /home/ben/task/ckpts/soup23 |
