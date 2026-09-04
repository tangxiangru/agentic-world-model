# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.0667 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.25 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.5467 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.05 | sft | base_model | closed | yes | supported | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 6.0 | rft | exp-03 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-05 | 6.6 | rft | exp-03 | closed | yes | inconclusive | iterate | accuracy=0.7533 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.15 | other | exp-05 | closed | yes | inconclusive | adopt | accuracy=0.762 | /home/ben/task/ckpts/exp-05/final |
| exp-07 | 7.2 | rft | exp-03 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.766 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 8.0 | merge | exp-07 | closed | yes | contradicted | reject | accuracy=0.758 | /home/ben/task/ckpts/exp-08/soup |
| exp-09 | 8.05 | other | exp-07 | closed | yes | inconclusive | adopt | accuracy=0.7475 | /home/ben/task/ckpts/exp-07/final |
| exp-10 | 8.1 | decode-config | exp-07 | closed | yes | supported | adopt | accuracy=0.702 | /home/ben/task/final_model |
| exp-11 | 8.25 | other | exp-07 | closed | yes | supported | adopt | accuracy=0.764 | /home/ben/task/final_model |
