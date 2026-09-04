# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.0867 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 1.0 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp-02/epoch2 |
| exp-03 | 3.0 | decode-config | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.7333 | /home/ben/task/ckpts/variants/exp02e2_greedy |
| exp-04 | 3.6 | rft | base_model | closed | yes | contradicted | reject | accuracy=0.7 | /home/ben/task/ckpts/exp-04/epoch2 |
| exp-05 | 5.6 | merge | exp-02 | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/variants/soup_greedy |
| exp-06 | 6.1 | sft | base_model | closed | yes | inconclusive | adopt | accuracy=0.78 | /home/ben/task/ckpts/variants/soup3_greedy |
