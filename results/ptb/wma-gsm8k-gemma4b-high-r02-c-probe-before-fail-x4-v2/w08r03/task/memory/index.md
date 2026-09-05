# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.06 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.55 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5533 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.9 | decode-config | exp-02 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6533 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 3.85 | sft | base_model | closed | yes (re-locked 3x) | contradicted | reject | accuracy=0.6867 | /home/ben/task/ckpts/exp-04/final_greedy |
| exp-05 | 6.35 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.66 | /home/ben/task/ckpts/soup_a_greedy |
| exp-06 | 6.5 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.6399 | /home/ben/task/ckpts/exp-04/final_greedy |
| exp-07 | 6.9 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.649 | /home/ben/task/ckpts/exp-07/final_greedy |
