# World-model agent

> **Status:** proposed design, not current behavior · **Scope:** PostTrainBench scientist runs
>
> **Principle:** the scientist owns training and the final interpretation; the
> world-model agent advises, and a deterministic runtime executes the protocol.

## 1. Boundary and invariants

The worktree has an untracked, currently unverified `awm experiment` prototype, not a WMA:
`awm wm`, yields, seals, policy arms, and WMA memory are new work.

| component | responsibility |
|---|---|
| **Scientist** | hypothesis, candidate training, replies, outcome, card disposition |
| **Runtime/sidecar** | validation, frozen state, evaluator execution, budgets, timeouts, pause/resume, seal |
| **WMA policy** | retrieval, prediction, objections, evaluation requests, checkpoint advice, memory; no candidate training |

Target invariants: accepted inputs are immutable; every WMA fact cites a persisted
path and locator; the sidecar alone changes state and appends idempotent events;
arms share runtime, prompt template, disclosure condition, data, budgets, and
timeouts; all persisted or fitted WMA state learns only from train-side sessions.

## 2. Experiment record

One card is one problem, hypothesis, intervention, and outcome, in **one
file**: `{dir}/memory/cards/exp-NN.yaml`, schema `awm-experiment-card-v1`
([Experiment card v1](./experiment-card-v1.md)) plus the fields the runtime
needs (`setup.resume_argv`, `setup.output_dir`, `setup.progress`,
`setup.base_model`, `evaluation.diagnostic.items`, `result.pings_acted_on`).

| sections | written by | when | runtime action |
|---|---|---|---|
| 1–4 `problem`, `hypothesis`, `setup`, `evaluation` | scientist | before launch | `propose` validates and grounds them; accept freezes a copy and its hash |
| 5–6 `result`, `conclusion` | scientist | after review | `finalize` validates them and re-checks that 1–4 still hash to the frozen copy |

The runtime's copy under `wm/cards/exp-NN/card.yaml` is the record of what
was frozen; the scientist's file may gain sections 5–6 but never change 1–4.
The same template serves every arm, including the null-arm control.

Lifecycle: `draft -> frozen -> running -> awaiting_review -> closed`.

Yields keep the card `running`. Withdrawal writes execution `not_run` and
decision `abandon_line` and closes directly; abort requires review. Other
dispositions are `adopt | reject | iterate`. The adopted parent chain is the
recipe for `{submission}`.

## 3. Lifecycle

```mermaid
sequenceDiagram
    participant S as Scientist
    participant R as Runtime
    participant W as WMA
    S->>R: propose(card)
    R->>W: grounded card + memory
    W-->>R: brief + contract
    R-->>S: brief ping
    S-->>R: accept / valid override
    Note over S,W: amend repeats proposal; withdraw closes
    R->>R: freeze and score parent
    S->>R: launch training
    loop checkpoints
        S->>R: checkpoint(path, step)
        R->>R: continue or evaluate
        R->>W: observation
        W-->>R: ping
        R-->>S: deliver
        R->>R: auto-resume notice/request; hold decision
        S-->>R: required reply
        R->>R: apply reply/timeout
    end
    opt checkpoint selected
        R->>R: create candidate seal
    end
    S->>R: result + conclusion
    opt card adopted
        R->>R: update incumbent and submission
    end
    R-->>S: close notice
    R->>W: close and train-side memory update
```

## 4. Ping protocol

`awm-ping-v1` files are canonical; `inbox.md`, stdout, and Stop hooks are views.
Each has `ping_id`, `card_id`, monotonic `seq`, `kind`, `summary`, and evidence
`{path, locator}`. Reply-required pings add `options` and a concrete
`{action, after_s}` `timeout_action`.

| kind | emitted | reply choices | timeout fallback |
|---|---|---|---|
| `brief` | proposal | `accept`, `amend`, `override`, `withdraw` | `remain_draft` |
| `notice` | evaluation/close | none | none |
| `yield_request` | unscheduled evaluation | `accept`, `reject` | `reject` |
| `decision` | rule/WMA/completion | `continue`, `more_eval`, `select:<obs-id>`, `abort` | frozen `timeout_action` |

Replies are idempotent; conflicts fail. An optional prediction names `{metric,
horizon, delta_mean, delta_sd, basis}`. Amendments are bounded; override needs a
reason and cannot waive schema, path, budget, or leakage checks.

Proposed CLI, scoped by `--dir` or harness-only `AWM_SESSION_DIR`:

```text
awm wm --dir DIR propose CARD
awm wm --dir DIR reply PING --choose OPTION [--why TEXT] [--amend FILE]
awm wm --dir DIR checkpoint CARD_ID CHECKPOINT --step N [--final]
awm wm --dir DIR serve
```

## 5. Evaluation contract

The brief returns grounding, precedents, an optional prediction, objections with fixes,
and a contract. The contract owns evaluation/decisions; fixed `config.yaml`
owns `total_runtime`/`requested_subset` budgets, timeouts, and amendment limits;
frozen `setup.resume_argv` belongs to the scientist. Rules are typed data.

`propose` rejects writable paths escaping `{dir}`, missing locators/gold,
watch-count mismatches, and contamination failures. External read-only inputs
must match the harness allowlist and a pinned logical ID, revision, or hash.

```yaml
schema_version: awm-evaluation-contract-v1
evaluators:
  - {name: dev150, kind: official, adapter: posttrainbench.gsm8k,
     metric: accuracy, direction: higher, n: 150, seed: 0,
     stderr: {method: bernoulli}}
  - {name: diag, kind: custom, metric: arithmetic_slip_rate, direction: lower,
     adapter: arithmetic_slip_v1, items: wm/cards/exp-03/evaluators/diag.jsonl, n: 120,
     stderr: {method: bootstrap, seed: 0, resamples: 2000}}
standing_yields:
  progress: {unit: optimizer_step, total: 1200}
  cadence: {kind: fractions, values: [0.25, 0.5, 0.75, 1.0]}
  evaluators: [dev150]
on_request: [diag]
rule_schema: awm-rule-v1
rules:
  - {id: regress, op: consecutive_threshold, evaluator: dev150,
     field: delta_vs_parent, comparator: lt, threshold: -0.03, count: 2,
     action: abort}
selection: {evaluator: dev150, direction: higher, completion_policy: best_observation}
```

Acceptance hashes the plan, contract, evaluator code/data, and parent manifest,
scores comparable parent baselines, and validates progress totals. Mid-run
yields start only after the first full trainer checkpoint proves restoration of
optimizer, scheduler, RNG, and data position. Runtime budgets cover evaluation
and reload. WMA inference is reported separately but remains inside the same
PostTrainBench wall deadline.

## 6. Yields, observations, and decisions

An idempotent checkpoint callback returns `0` to continue or `3` after saving to
pause. On `3`, the sidecar evaluates under the GPU lease. Notices and pending
requests auto-resume; decisions wait for reply/timeout and resume only on
`continue`.

Standing yields follow the contract. Accepted requests apply at the next save
(or `--final`) without blocking training. They and `more_eval` charge both
budgets; `more_eval` evaluates the same checkpoint and reissues the decision.

An observation records input hashes; checkpoint path, step, and manifest;
metric value/unit, `n`, uncertainty method, artifacts, deltas, watch transitions,
time, and rules. Pings reference observations, never the reverse. Official
adapters parse task metrics; custom evaluators also emit item records.

At completion, `best_observation` resolves to a concrete `select:<obs-id>`
timeout. Selection load-verifies and seals every load-critical file's path,
size, and SHA-256 plus evaluator hashes, metrics, decision, and ping. Only later
`adopt` atomically updates the incumbent and `{submission}`; abort/withdraw skip
sealing.

Ungrounded WMA prose is replaced by a deterministic observation/rule message.
A split guard—not `tools/exp_cards_post.py`—gates every build input and update.

## 7. WMA arms and memory

Policy interface: `on_proposal -> Brief`, `on_observation -> Ping`, `on_reply ->
Ping | None`, and `on_close` / `on_outcome -> MemoryUpdate`.

Arms are cumulative:

| arm | deterministic grounding/rules | retrieval | LLM advice/requests | learned prior/posterior |
|---|---:|---:|---:|---:|
| `null` | yes | no | no | no |
| `retrieval` | yes | yes | no | no |
| `llm` | yes | yes | yes | no |
| `predictor` | yes | yes | yes | yes |

Sidecar-only `AWM_WM_MEMORY` stores raw artifacts, structured interactions,
retrieval/predictor state, and grounded notes tagged by session, arm, and split.
Train cards may seed retrieval only after a deterministic audit gate; they
remain marked, never live evidence.
Held-out sidecars mount memory read-only and discard updates. Predictions name
an evaluator and horizon; a host-side PTB post-grading importer later sends
train-only `official_outcome` to `on_outcome`. Withdrawals remain censored.

## 8. Storage and integration

```text
{dir}/wm/
  config.yaml  inbox.md  events.jsonl
  cards/exp-NN/
    card.yaml  contract.yaml  manifest.json  state.json  result.yaml
    grounding/  evaluators/  observations/  pings/  replies/  seal.json
```

| component | change |
|---|---|
| `awm/experiment.py` | reuse file/hash/locking patterns; replace schemas, ledger, refreeze, runner, review |
| `awm/cli.py`, `.claude/hooks/` | add four scoped commands and a pending-reply hook |
| split/trajectory/analysis assets | adapt split, event, digest, evidence, and identity logic into runtime gates |
| PTB runner and scientist assets | add external sidecar/agent shim, post-grade importer, prompt, template, and skill |
| new `awm/wm/` | add `schema.py`, `protocol.py`, `runtime.py`, `memory.py`, and `agents/` |
| conformance tests | cover replay, recovery, freeze drift, resume, adapters, budgets, seals, leakage, outcomes |

Each run records disclosure plus prompt, policy, model, retrieval-index, and
predictor hashes. The PTB runner keeps the sidecar, memory, endpoint, and
credentials outside the scientist namespace; it materializes sanitized cited
evidence under `grounding/`. `{dir}/wm/` is the canonical persisted channel.

## 9. Study protocol and open decisions

Freeze memory before held-out runs. Hold scientist scaffold/model/config/prompt
hash, task/base model, seeds, and deadline fixed. Primary outcome: official
accuracy. Also report WMA GPU time, training time, decisions, acceptance, and
calibration.

| decision | resolve with |
|---|---|
| disclose arm assignment to scientist? | separate, stratified behavior pilot |
| runtime-fixed vs WMA-proposed rules | cross-arm variance/fairness |
| reply timeouts | pilot response times |
| every-checkpoint vs fractional cadence | checkpoint rate, reload cost, training duration |

## 10. Implementation (2026-08-31)

| piece | where |
|---|---|
| schemas, grounding checks, contract validation | `awm/wm/schema.py` |
| pings, replies, ledger, inbox, stdout view | `awm/wm/protocol.py` |
| propose → brief → freeze → checkpoint hook → worker → seal → finalize | `awm/wm/runtime.py` |
| evaluators (official `evaluate.py`, custom item scorer) | `awm/wm/evaluators.py`, `awm/wm/score_items.py` |
| memory (raw / structured / notes, precedents, seeding from `results/exp-cards/`) | `awm/wm/memory.py` |
| arms: `null`, `retrieval` (deterministic), `llm` / `traj` (autonomous Claude Code, read-only, cited JSON; `traj` = raw prior runs only), `predictor` stub | `awm/wm/agents/` |
| trainer hook (`TrainerCallback` + plain function) | `awm/wm/hook_example.py` (copied into `{dir}/wm/`) |
| CLI `awm wm init|propose|reply|checkpoint|worker|finalize|status|pending|memory` | `awm/cli.py` |
| Claude Code Stop hook (hold the turn while a reply is owed) | `.claude/hooks/wm_pending_reply.py` |
| scientist skill | `.claude/skills/awm-wm/SKILL.md` |
| one prompt, one template (the null arm is the control) | `input/instruction.md`, `input/exp-card.template.yaml` (copy of `doc/exp-card-specs/experiment-card.template.yaml`) |
| end-to-end test with a fake trainer and fake graders | `tests/test_wm.py` |

Simplifications relative to the sections above: the worker is spawned by the
hook (no long-running `serve`); rules support `consecutive_threshold` only;
official per-item results are not yet read back from inspect logs (the watch
set carries the item-level signal); reload cost is not measured at freeze.
