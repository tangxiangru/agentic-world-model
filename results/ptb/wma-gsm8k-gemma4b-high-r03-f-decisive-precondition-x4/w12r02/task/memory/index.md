# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.06 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.55 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7933 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 4.05 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.82 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.85 | sft | exp-02 | closed | yes (re-locked 1x) | inconclusive | abandon_line |  |  |
| exp-05 | 6.1 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.78 | /home/ben/task/ckpts/exp-05 |
| exp-06 | 6.45 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.7733 | /home/ben/task/ckpts/exp-06/final |
