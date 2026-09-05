# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | adopt | accuracy=0.0333 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.3 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6267 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.2 | decode-config | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.66 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 3.6 | rft | base_model | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-04_greedy |
| exp-05 | 5.7 | decode-config | exp-04 | closed | yes | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-05_ck500_greedy |
| exp-06 | 6.2 | merge | exp-04 | closed | yes | contradicted | iterate | accuracy=0.7467 | /home/ben/task/ckpts/exp-06_soup_greedy |
| exp-07 | 6.4 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.714 | /home/ben/task/final_model |
| exp-08 | 6.95 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.674 | /home/ben/task/ckpts/exp-08_greedy |
