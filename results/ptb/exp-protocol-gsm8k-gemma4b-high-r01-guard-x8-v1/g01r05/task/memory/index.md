# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.25 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.055 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.71 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.6 | rft | exp-02 | closed | yes | inconclusive | iterate |  |  |
| exp-04 | 4.6 | rft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.72 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.0 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.7536 | /home/ben/task/ckpts/exp-04/final |
| exp-06 | 7.1 | rft | exp-04 | closed | yes | inconclusive | adopt | accuracy=0.7619 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.9 | merge | exp-06 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7536 | /home/ben/task/ckpts/exp-07/final |
