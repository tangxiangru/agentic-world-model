# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | adopt | accuracy=0.0867 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.35 | sft | base_model | closed | yes | contradicted | iterate | accuracy=0.02 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.4 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.62 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.5 | decode-config | exp-03 | closed | yes | supported | adopt | accuracy=0.68 | /home/ben/task/final_model |
| exp-05 | 5.05 | rft | exp-03 | closed | yes | inconclusive | iterate | accuracy=0.6733 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-06 | 6.95 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.6664 | /home/ben/task/final_model |
| exp-07 | 7.75 | merge | exp-03 | closed | yes | contradicted | reject | accuracy=0.6664 | /home/ben/task/ckpts/soup_03_05_greedy |
