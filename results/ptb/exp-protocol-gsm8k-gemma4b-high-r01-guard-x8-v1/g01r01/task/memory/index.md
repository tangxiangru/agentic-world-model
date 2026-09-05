# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | adopt | accuracy=0.0267 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp02/final |
| exp-03 | 2.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp02_greedy |
| exp-04 | 4.5 | rft | exp-02 | closed | yes | contradicted | iterate | accuracy=0.7 | /home/ben/task/ckpts/exp04/final |
| exp-05 | 6.2 | other | exp-04 | closed | yes | contradicted | adopt | accuracy=0.724 | /home/ben/task/ckpts/exp04_greedy |
| exp-06 | 6.5 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.7195 | /home/ben/task/final_model |
| exp-07 | 6.9 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.7218 | /home/ben/task/ckpts/exp07/final |
| exp-08 | 7.0 | other | exp-07 | closed | yes | contradicted | iterate | accuracy=0.714 | /home/ben/task/final_model |
| exp-09 | 7.3 | merge | exp-07 | closed | yes | supported | adopt | accuracy=0.7202 | /home/ben/task/ckpts/soup27 |
