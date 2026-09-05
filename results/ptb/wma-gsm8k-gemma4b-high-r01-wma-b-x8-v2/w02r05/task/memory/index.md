# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.03333 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.64667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.75 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.69333 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 3.65 | other | exp-02 | closed | yes | inconclusive | abandon_line |  |  |
| exp-05 | 3.7 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.65333 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 4.5 | rft | base_model | closed | yes | supported | adopt | accuracy=0.73333 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.0 | merge | exp-06 | closed | yes | supported | adopt | accuracy=0.73313 | /home/ben/task/ckpts/soup_d |
| exp-08 | 7.6 | decode-config | exp-07 | closed | yes | contradicted | reject | accuracy=0.721 | /home/ben/task/ckpts/soup_d_reppen |
| exp-09 | 7.8 | merge | exp-07 | closed | yes | supported | reject | accuracy=0.73161 | /home/ben/task/ckpts/soup_e |
