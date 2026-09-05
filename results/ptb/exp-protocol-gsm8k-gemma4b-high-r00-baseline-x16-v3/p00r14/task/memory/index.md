# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.25 | other | base_model | closed | yes | supported | adopt | accuracy=0.06 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.46 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.92 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.66 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 3.8 | rft | base_model | closed | yes | supported | adopt | accuracy=0.6596 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.75 | sft | exp-04 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-06 | 8.05 | sft | exp-04 | closed | yes | inconclusive | adopt | accuracy=0.6733 | /home/ben/task/ckpts/exp-06/final |
