# Checkpoint inventory for the 1,000-cell evaluation matrix

## Recommendation

Use the breadth-first design: **400 checkpoint-weight candidates under two shared
generation policies, followed by 200 targeted checkpoint-policy cells**. This is
better aligned with the proof-of-concept question than evaluating only 200
checkpoints under four policies. Two common policies identify the main serving
contrast across twice as many training recipes; the targeted cells can then test
nonlinearity, tails, and the manually identified close pairs.

The proposed 400 are a **conditional selection**, not 400 hash-certified unique
weight tensors. Run the weight/load preflight below before assigning GPU cells.

## Scope and census

The authoritative scope is the 579 rows for which
`prefix_recipes_v6/labels.jsonl` has
`output.eligible_for_training == true`. This is the 582 complete ten-pass labels
minus the three quarantined rows:

- `r0-25-exp-02`
- `aime-r0-11-exp-02`
- `aime2-r0-12-exp-05`

No outcome value was copied into the inventory or used to select checkpoints.

| Population | GSM8K | AIME | Total |
|---|---:|---:|---:|
| Eligible archived checkpoints | 326 | 253 | 579 |
| Eligible from-base training outputs | 190 | 123 | 313 |
| Continued-training candidates | 86 | 60 | 146 |
| Parameter-merge candidates | 40 | 17 | 57 |
| Candidate weight artifacts before hash deduplication | 316 | 200 | 516 |
| Same/selected-weight serving bundles | 10 | 53 | 63 |

`candidate_weight_bool` is deliberately a conservative semantic proxy. It keeps
training outputs and parameter-space merges and removes decoding-only,
evaluation-only, copy, hardlink, symlink, and checkpoint-selection bundles where
the cards show no new training or merged tensors. It does **not** establish that
all 516 candidates have different tensor bytes.

The broad `table_v2` recipe family is retained as `raw_recipe_family`, while
`weight_change_kind` corrects known category mismatches. In particular, six rows
called SFT are actually parameter soups, two SFT rows are evaluation bundles,
and four rows called merges are re-evaluations of existing weights. These
corrections use the recorded launch command and card evidence, not outcomes.

## Proposed 400-checkpoint breadth set

The machine-readable ID list is [selected_ids_400.json](selected_ids_400.json),
and every row plus its selection reasons is in
[checkpoint_inventory.json](checkpoint_inventory.json).

The SHA-256 of the exact selected-ID file is
`fffcce6b7ab680537437a9817b3cb3f4a8c27428a914a01d0c9abd708f855e97`;
the same digest is stored at the inventory's top level.

| Selected population | GSM8K | AIME | Total |
|---|---:|---:|---:|
| From-base core | 190 | 123 | 313 |
| Continuations or parameter merges | 50 | 37 | 87 |
| Total | 240 | 160 | 400 |

The selection covers all 124 source sessions. It contains 313 base-training
outputs, 65 continued-training outputs, and 22 parameter merges. Its raw recipe
mix is 334 SFT, 38 RFT, 21 merge, 4 RL, 2 distillation, and one real full-SFT
checkpoint whose family is `other` because its launch script was unavailable.
It includes 63 LoRA-origin and 337 full-finetune/merge checkpoints.

The continuation selection is reproducible and outcome-free:

1. Include every eligible from-base checkpoint.
2. Allocate 50 GSM8K and 37 AIME continuation/merge slots.
3. Force a representative from each session lacking an eligible from-base row.
4. Keep one best-available RL or distillation row per session.
5. Maximize continuation-session coverage, prioritizing reconstructed code,
   available entrypoints, matching snapshots, and clean v6 review flags.
6. Fill residual slots by recipe family, weight operation, LoRA status, and
   dataset-source novelty, with lexicographic experiment ID as the final tie
   breaker.

This retains all 313 eligible queries from the outcome-free manual matching
study. Their grades are 19 A, 219 B, 74 C, and 1 D. Three queries point to a
quarantined best-match partner; the inventory flags those partners as
unavailable rather than silently substituting a different checkpoint. Grade-A
and mutual grade-B pairs are marked as diagnostic anchors, but grades do not
affect membership because every eligible from-base query is already included.

## Recipe and lineage readiness

For the 516 candidate-weight artifacts, training-script status is:

| Code status | GSM8K | AIME | Total |
|---|---:|---:|---:|
| Reconstructed | 283 | 180 | 463 |
| Unavailable | 18 | 16 | 34 |
| Blocked | 15 | 4 | 19 |

For the selected 400, 371 are reconstructed, 20 unavailable, and 9 blocked.
There are 177 selected rows carrying the v6 content-review flag and three with
`v6_missing_declared_code`: `gsm2-r0-19-exp-01`,
`gsm2-r0-19-exp-03`, and `r0-17-exp-04`. These are conditional selections that
must clear preflight; their flags are not silently treated as complete code.
The 313-row from-base core alone has 290 reconstructed, 16 unavailable, and 7
blocked scripts; 312/313 have no missing-declared-code flag, while 123 require
the existing v6 content review.

The inventory carries normalized parent IDs and a parent-connected-component
proxy. There are **zero declared cross-session learned-weight parent edges** in
`table_v2`; all normalized learned-weight edges stay within a session. This is
not proof that undeclared cross-session copying never happened. Use the whole
source session (`split_group_primary`) as the primary grouped split, which also
protects against shared trajectory prefixes.

Sixty-three rows are classified as same/selected-weight serving bundles. An
exact representative is bound for only 24 of them, where trajectory/card text
explicitly says byte-identical, bit-identical, symlink, hardlink, or otherwise
identifies the unchanged source. For the rest, `alias_targets` records only the
declared parent candidates and labels that basis as unverified; the manifest
does not pretend those are tensor hashes.

## Metadata availability

The immutable HF metadata revision
`446127629d7b271d537390e69bfb2d960a3aa515` contains an archived
`generation_config.json` for all 579 eligible rows. The inventory records the
exact pinned local path for each. This closes generation-config *file*
availability, but it does not establish unique weights: the metadata release
provides both `generation_config.json` and `config.json` for all 582 labeled
checkpoints (and therefore all 400 selected checkpoints), but it does not provide
tensor-shard hashes or the full tokenizer artifacts needed to establish weight
identity and a complete serving bundle. Only five of the selected `config.json`
files happen to be present in this local cache; that cache state must not be
confused with remote HF availability.

## Mandatory preflight

Before treating the proposed 400 as distinct checkpoint axes:

1. Fetch each exact archive and hash the ordered model-weight payloads only
   (`*.safetensors` plus the shard index where applicable), separately from
   tokenizer and generation metadata.
2. Collapse equal weight hashes to one representative. Preserve all aliases and
   their recipe/card IDs so the audit trail remains intact.
3. Verify that every representative loads with the pinned evaluator backend and
   resolves the expected architecture, tokenizer, chat template, EOS/stop
   behavior, context limit, and dtype.
4. For LoRA-origin artifacts, distinguish adapter-only archives from merged
   models and bind the exact base-model revision before hashing or serving.
5. Replace any failed or duplicate selected candidate with the highest-ranked
   unselected candidate from the same benchmark/method stratum. Reallocate cells
   freed by deduplication to targeted generation-policy arms rather than silently
   reducing breadth.

The alternative if exactly 100 continuations are required is 413 checkpoints
under two shared policies (826 cells), leaving 174 targeted cells. The current
400-checkpoint set leaves a round 200 targeted cells and already covers every
session.

## Auditable sources

- `data/analysis/wm_exp_designs/prefix_recipes_v6/{inputs,audit,labels}.jsonl`
- `data/analysis/wm_exp_designs/table_v2/experiments.jsonl`
- `data/analysis/wm_exp_designs/twins_manual/matches/*.json`
- HF metadata snapshot
  `https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/tree/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/`

Useful validation commands:

```sh
jq '.checkpoints | length' checkpoint_inventory.json
jq '[.checkpoints[] | select(.candidate_weight_bool)] | length' checkpoint_inventory.json
jq '[.checkpoints[] | select(.selection_400)] | length' checkpoint_inventory.json
jq '[.checkpoints[] | select(.selection_400) | .session] | unique | length' checkpoint_inventory.json
jq 'length == 400 and (unique | length) == 400' selected_ids_400.json
```
