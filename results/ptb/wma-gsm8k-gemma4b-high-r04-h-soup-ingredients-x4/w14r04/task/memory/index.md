# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.08 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.62 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.68 | /home/ben/task/ckpts/exp-02/final_eval |
| exp-03 | 2.75 | sft | exp-02 | closed | yes (re-locked 1x) | inconclusive | reject | accuracy=0.68 | /home/ben/task/ckpts/exp-03/final_eval |
| exp-04 | 6.05 | rft | exp-03 | closed | yes | inconclusive | reject | accuracy=0.6533 | /home/ben/task/ckpts/exp-04/final_eval |
| exp-05 | 6.98 | merge | exp-04 | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/soup3_eval |
| exp-06 | 7.45 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.7 | /home/ben/task/ckpts/soup3_eval |
