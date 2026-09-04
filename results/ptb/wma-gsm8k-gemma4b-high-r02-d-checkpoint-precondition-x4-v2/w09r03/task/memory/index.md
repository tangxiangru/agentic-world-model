# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | adopt | accuracy=0.06 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.85 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6267 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.15 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-03-greedy-ep1 |
| exp-04 | 4.0 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.6933 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.0 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.74 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 5.3 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/soup-exp02-greedy |
| exp-07 | 7.5 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.76 | /home/ben/task/final_model |
| exp-08 | 7.6 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.712 | /home/ben/task/final_model |
