# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | adopt | accuracy=0.07 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.575 | /home/ben/task/ckpts/exp-02 |
| exp-03 | 1.05 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.635 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 1.15 | sft | base_model | closed | yes | supported | adopt | accuracy=0.665 | /home/ben/task/ckpts/exp-04 |
| exp-05 | 4.05 | sft | exp-04 | closed | yes | inconclusive | iterate | accuracy=0.675 | /home/ben/task/ckpts/exp-05 |
| exp-06 | 6.85 | rft | exp-05 | closed | yes | contradicted | reject | accuracy=0.675 | /home/ben/task/ckpts/exp-06 |
| exp-07 | 6.45 | merge | exp-05 | closed | yes | supported | adopt | accuracy=0.735 | /home/ben/task/ckpts/soup-C |
