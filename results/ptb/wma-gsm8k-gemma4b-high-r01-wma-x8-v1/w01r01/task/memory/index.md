# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.0667 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.2 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.3 | decode-config | exp-02 | closed | yes | contradicted | iterate | accuracy=0.72 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 2.5 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.4 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.7278 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.0 | sft | exp-05 | closed | yes | inconclusive | adopt | accuracy=0.7346 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 8.5 | decode-config | exp-06 | closed | yes | supported | adopt | accuracy=0.7824 | /home/ben/task/final_model |
