# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.0533 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.62 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.2 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 2.25 | other | exp-02 | closed | yes | contradicted | reject | accuracy=0.6133 | /home/ben/task/ckpts/exp-02/ep1 |
| exp-05 | 4.25 | rft | base_model | closed | yes | contradicted | iterate | accuracy=0.6933 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-06 | 6.35 | other | exp-05 | closed | yes | inconclusive | iterate | accuracy=0.704 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-07 | 6.5 | merge | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7005 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-08 | 6.9 | other | exp-05 | closed | yes | contradicted | adopt | accuracy=0.6907 | /home/ben/task/final_model |
| exp-09 | 7.05 | other | exp-05 | closed | yes | contradicted | reject | accuracy=0.6922 | /home/ben/task/final_model |
| exp-10 | 7.1 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7316 | /home/ben/task/final_model |
