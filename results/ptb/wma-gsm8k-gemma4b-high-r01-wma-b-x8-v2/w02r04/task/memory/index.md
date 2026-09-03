# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.0667 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.35 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.6 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.05 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.68 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 2.2 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.68 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.35 | rft | exp-04 | closed | yes (re-locked 1x) | contradicted | iterate | accuracy=0.68 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.2 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.686 | /home/ben/task/ckpts/exp-05/final |
| exp-07 | 7.5 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.666 | /home/ben/task/ckpts/soup-0405 |
| exp-08 | 7.75 | other | exp-05 | closed | yes | inconclusive | reject | accuracy=0.696 | /home/ben/task/ckpts/exp-05/final |
