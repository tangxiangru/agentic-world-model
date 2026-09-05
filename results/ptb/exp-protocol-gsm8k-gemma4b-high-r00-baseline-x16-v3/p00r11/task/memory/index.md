# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.25 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.0667 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.55 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.62 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.6 | rft | exp-02 | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp-03/checkpoint-702 |
| exp-04 | 5.1 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.6533 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.75 | merge | exp-03 | closed | yes | contradicted | reject | accuracy=0.68 | /home/ben/task/ckpts/exp-05-merge |
| exp-06 | 5.9 | decode-config | exp-03 | closed | yes | supported | adopt | accuracy=0.664 | /home/ben/task/ckpts/exp-03/final |
| exp-07 | 6.2 | decode-config | exp-03 | closed | yes | supported | adopt | accuracy=0.702 | /home/ben/task/final_model |
| exp-08 | 6.9 | rft | exp-03 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.671 | /home/ben/task/ckpts/exp-08/final |
| exp-09 | 8.05 | decode-config | exp-03 | closed | yes | contradicted | reject | accuracy=0.7013 | /home/ben/task/ckpts/decode-reppen |
