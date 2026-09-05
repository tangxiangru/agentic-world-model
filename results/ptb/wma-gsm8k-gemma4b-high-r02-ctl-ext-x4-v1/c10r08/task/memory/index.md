# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | adopt | accuracy=0.0667 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.75 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6067 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.55 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.71 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 2.65 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.764 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.95 | rft | exp-04 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-06 | 5.75 | rft | exp-04 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-07 | 6.55 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.754 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 7.2 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.744 | /home/ben/task/ckpts/exp-08_avg |
| exp-09 | 7.55 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.756 | /home/ben/task/ckpts/exp-09/final |
