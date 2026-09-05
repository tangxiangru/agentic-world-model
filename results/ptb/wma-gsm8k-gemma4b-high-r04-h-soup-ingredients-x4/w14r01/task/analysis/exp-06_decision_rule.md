# exp-06 selection rule — written 2026-09-04T15:14Z, BEFORE any n=1319 read finished

Candidates (all greedy arm, all read at n=1319 inside exp-06):
  A  ckpts/exp-04/checkpoint-1484   (dev-150: 0.7600)
  B  ckpts/exp-05_soup              (dev-150: 0.7533)
  C  ckpts/exp-04/checkpoint-742    (dev-150: 0.7333)

Unpaired SE of a difference between two 1319-item reads is ~1.6 pp.

Rule, fixed in advance:
1. If the best n=1319 accuracy exceeds the second best by >= 2.0 pp, package the best.
2. If the top two are within 2.0 pp, package B (the soup), on the grounds that a
   uniform weight average over three trajectory points has lower selection
   variance than a single point chosen as the argmax of noisy reads. If B is not
   one of the top two in that case, package the earlier checkpoint of the two.
3. A McNemar paired count of discordant items between the top two is reported
   either way (it costs no GPU), but it does not override rules 1-2.
4. final_model/ is only overwritten if the winner's n=1319 accuracy is at least
   as high as the n=1319 accuracy of whatever final_model currently holds
   (checkpoint-1484), and PACKAGED.json records the n=1319 protocol string.

Caveat recorded now: the 1319 items include the 150 that produced the original
ranking, so ~11% of this read is not independent evidence.
