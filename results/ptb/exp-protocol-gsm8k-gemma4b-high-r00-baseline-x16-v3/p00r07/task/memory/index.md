# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | decode-config | base_model | closed | yes | supported | adopt | accuracy=0.1267 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.27 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5867 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.85 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.692 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.15 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.736 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.05 | sft | exp-04 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-06 | 6.15 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.724 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 8.15 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.734 | /home/ben/task/ckpts/exp-07-soup |
| exp-08 | 8.3 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.71 | /home/ben/task/ckpts/exp-08/final |
