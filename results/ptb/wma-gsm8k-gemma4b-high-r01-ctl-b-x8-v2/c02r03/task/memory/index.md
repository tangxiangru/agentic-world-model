# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | decode-config | base_model | closed | yes | supported | adopt | accuracy=0.0933 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 1.1 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 6.4 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.7 | merge | exp-02 | closed | yes | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-04-soup |
| exp-05 | 6.0 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.4 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-06-soup3 |
| exp-07 | 6.5 | decode-config | exp-04 | closed | yes | contradicted | adopt | accuracy=0.732 | /home/ben/task/ckpts/exp-04-soup |
| exp-08 | 6.9 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.718 | /home/ben/task/ckpts/exp-08-soup25 |
| exp-09 | 7.0 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.76 | /home/ben/task/final_model |
| exp-10 | 7.0 | decode-config | exp-04 | closed | yes | contradicted | adopt | accuracy=0.738 | /home/ben/task/ckpts/exp-03/final |
| exp-11 | 7.6 | decode-config | exp-03 | closed | yes | inconclusive | iterate | accuracy=0.7067 | /home/ben/task/final_model |
| exp-12 | 7.7 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.76 | /home/ben/task/final_model |
