# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.0533 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.2 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.78 | /home/ben/task/ckpts/exp-02_greedy |
| exp-04 | 3.6 | sft | base_model | closed | yes | inconclusive | iterate | accuracy=0.7933 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.5 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.7733 | /home/ben/task/ckpts/soup2 |
| exp-06 | 6.9 | other | exp-04 | closed | yes | contradicted | adopt | accuracy=0.806 | /home/ben/task/ckpts/exp-04_greedy |
| exp-07 | 7.3 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.7933 | /home/ben/task/final_model |
