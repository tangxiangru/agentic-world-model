# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.55 | sft | base_model (gemma-3-4b-pt @cc012e0a) | completed | yes (override: stop_token_consistent) | supported | adopt | 0.580 acc @n=150 | /home/ben/task/final_model |

Note: `awm exp_protocol close` could not be run for exp-01 -- /home/ben/awm became a stale
NFS file handle after the lock step, so the CLI is unavailable. Sections 5-6 of
memory/cards/exp-01.yaml are filled in and the file parses as valid YAML; this index row
was appended by hand in place of the tool's rebuild.
