# exp-04 inputs

family=merge: this card reads no training data. Its inputs are two checkpoint
directories, averaged tensor by tensor:

1. /home/ben/task/ckpts/exp-03/final   (alpha 0.5)
2. /home/ben/task/ckpts/exp-02/final   (alpha 0.5)

Both descend from snapshot cc012e0a6d0787b4adcc0fa2c4da74402494554d.
The 300-item held-out probe scored by the diagnostic is data/probe300.jsonl,
declared under problem.watch_set.
