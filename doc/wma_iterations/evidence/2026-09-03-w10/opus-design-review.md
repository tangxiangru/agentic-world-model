# Round 04 candidate review — G/H/I/J

**Approve G and H (both with narrowed text). Defer I and J.** Nothing modified, submitted, cancelled, committed or pushed; scorer, fence, prompt, schema and cost accounting untouched.

I read the four packet files, `skills/wma/SKILL.md`, `skills/wma/change_types.md`, `skills/wma_meta/SKILL.md`, the Round 02/03 records, and the 30 completed `w10r01..04` terminal verdicts plus their cards. `uptake-levers.md` was present, so I used it rather than the fallback. No in-flight candidate outcome, no `c10r04`, no extension cell was read.

---

## 1. Corrections to the packet's claims

**Confirmed as stated** (so you don't re-verify): 143 probes and 20 `changed=none` reproduce exactly from the 30 terminal verdicts (per cell 37/34/40/32); 30 verdicts = 8+7+9+6; 43 response records exist; w10r03/exp-07's −1.33 pp → +3.0 pp reversal (.740/.7267 at n=150 → .705/.735 at n=400); w10r02/exp-04's .694 four-way / .704 best ingredient / .710 two-way; 11/30 flags with 7 env-only verdicts and 4 package-source verdicts, 7+18=25 commands.

**C1 — I's determinism premise is contradicted by the skill's own manual.** `change_types.md:96` records gsm8k n=1319 greedy *byte-identical copies* at 0.15–0.45 pp spread (near-greedy t=1e-6 at 1.6 pp); only bfcl n=100 is 0.0 pp bit-exact. `uptake-levers.md` adds c10r02/exp-07 (same aggregate, item churn) and w10r03/exp-08 (concurrency 16→2 changed 8/150 labels). "Rerunning identical deterministic predictions" is not an established state of the world, and the C18 row already says repeats buy "the standard error, not a verdict."

**C2 — I's deficit does not exist in the baseline.** Across 210 w10 suggestions, 16 already propose a larger n, and w10r02/exp-05 explicitly *declines* a repeat for I's own stated reason: *"[tier 4, 0 min] do not spend a C18 repeat on this: both arms are greedy on the same first-500 items, so the comparison is deterministic…"*. The 25 repeat-mentioning suggestions are mostly correct uses — repeats offered where the arm is **sampled** (w10r01/exp-01: "A sampled n=150 read has a ~3.3 pp floor"; w10r02/exp-01) or where n is already the full 1319 and no larger ruler exists (w10r04/exp-05, exp-06). Both cited reversals are cards the scientist wrote *as* the larger-N arbitration, under v0.2, unprompted. Separately, §5's closing line already reads "needs a C18 repeat **or a larger n**" — so the C18 row's "only correct move" is an internal inconsistency §5 already overrides. That is a copy-edit, not a testable mechanism.

**C3 — J's primary is arithmetically unreachable.** Its mechanism base is ~2–4 probes of 143 (my scan finds 1 probe text naming a later card; `decision-harms.md` names 3–4 counting w10r02/exp-01 p2). All 20 `changed=none` probes together carry 14.5 self-reported cpu_min of 165.8 — a **5.4% ceiling** on per-cell request time even if every one were removed, which J must not do (w10r02/exp-03 p3 found `save_total_limit=2` at `changed=none`). Against a 20% target on a baseline of 80.11/58.76/69.58/59.66 min (mean 66.78, SD 10.01, **CV 15.0%**): a fully realized 20% effect gives ≈0.38 power at n=4/arm; the deliverable ~2–5% gives ≈0.07. Also, v0.2 already contains "Run a probe only if its result would change a level" (`SKILL.md:120`), and the actual cost driver is relocks (43 cycles for 30 cards, each a full ~6 min wait), which J does not touch.

**C4 — two wording fixes in G, and the metric is misnamed.** The directory deleted in w10r01/exp-05 was the installed package's `__pycache__` (`/usr/local/lib/python3.10/dist-packages/gguf/__pycache__`), not "outside scratch"; w10r04/exp-05's outside token is `/dev/shm`, a shared-memory filesystem, not "hardware". And `n_leak_suspected` is a **fence-violation counter** — the audit positively identified no card outcome or held-out content in any of the 25 commands. G's write-up must not claim reduced leakage.

**C5 — G's draft text smuggles in J's mechanism.** The access-audit draft opens *"Name the level it could change and check that every explicit input is in the session…"*. That first clause duplicates `SKILL.md:120` in substance and makes G and J non-separable. Its "base confidence on the in-scope evidence that remains" also overlaps Round 02's C. Both are removed below.

**C6 — H's premise is *stronger* than the packet claimed, and its scope should shrink.** `change_types.md:47` literally reads "a variance-reduction move, **the default when time is short**." That phrase is quoted verbatim in the `evidence[].note` of **4 of the 5** w10 soup cards, and in w10r03/exp-06 it enters the L3 reasoning directly: *"…and because C6 is the manual's default when time is short. Two conditions keep this from being a clean yes."* — on the one soup that came in 4.7 pp under its strong ingredient (.6933 vs .7400). That is a documented text→prior→L3 channel, better evidence than the packet offered.

But the packet's *other* H clauses are already baseline behavior, and including them would make the readout unattributable. w10r02/exp-04 already suggested the subset (*"also make the 2-ingredient soup of ck650 + ck1350 … both end-of-epoch points and both the lower n=150 reads"*) and already demanded same-ruler scores (*"only ck1350 has an n=500 score. Score ck650, ck1293 and ck1757 at n=500 as well"*); w10r01/exp-07 already named the trajectory relationship (*"exp-03/best is the initialisation of exp-04's RFT run…not an average of two decorrelated models"*) and already offered keeping the incumbent.

**C7 — H's control citation.** c10r03/exp-05's .764→.748→.720 is a **train-derived n=500 probe**, not official GSM8K. And c10r01–03 contain **0 `.verdict.json` files** (`not_attached`), so no control card can enter any H or G denominator.

**C8 — the C6 denominator must not be the reviewer's own label.** Six w10 cards carry C6, but w10r03/exp-05 is a `train_sft.py` continued-training card with an incidental label.

**C9 — "L0/L1 recall falls by >0.05" is not measurable.** w10 L0/L1 hit is 1.0 with 2 recovered negatives across 43 cycles. Replaced below.

**C10 — the 1.5× guard cannot be evaluated on total review cycles.** I checked all 43 `.wma/responses/*.json`: keys are `{schema_version, request_id, card_ids, state, completed_at, backend, model, effort, ranking, errors}` — **no cost field**. The baseline's per-cycle cost does not exist in the artifacts.

**C11 — attribution fix.** "w10r01 exp06 n500 reverses n150 checkpoint order" is better stated as the exp-04/ck250-vs-soup pair tracked across exp-06 (n=500) and exp-08 (n=1319): .740/.7133 at n=150 → .666/.700 at n=500 → .6740/.6983 at n=1319. exp-06 is the four-candidate n=500 read; exp-08 is the full-split arbitration.

---

## 2. G — approved. Refined single edit

One bullet into `SKILL.md` Probes → Rules, after the "Online mode" bullet.

> - **Keep a probe inside your inputs.** Every explicit input a probe reads must be the session, the named WMA skill, or the supplied `history/`, and everything it writes must stay in your own scratch (run Python probes with `PYTHONDONTWRITEBYTECODE=1`). A capacity or feasibility check names the session or your scratch — `df /session`, the preflight report, `system_monitor.log` — never a home directory, `/dev/shm`, or an installed package tree; do not import a package in order to inspect its installation, and do not copy an outside file into scratch or rewrite its path to get around this. Never repair, clean up, or restore anything outside your scratch: if a probe touches an outside path or leaves a side effect, stop it and say so in `evidence`. When the check you want cannot stay inside these bounds, leave the mechanism unverified under the basis rule above and give the scientist a costed precondition instead.

**Why it is justified:** v0.2 already states the fence (`SKILL.md:48-50`) but passively ("if you come across"), says nothing about writes, and gives no guidance for a probe that wants outside information. It is violated at 11/30, in all four cells. One sentence — a disk check names the session or scratch — addresses 7 of the 11 flags directly.

**Preregistration**
- **Primary:** frozen `n_leak_suspected / n_verdicts` from the unchanged `_read_outside`, per cell and pooled. Baseline 11/30 (2/8, 4/7, 3/9, 2/6). Gate: 0 across all 4 cells. Power: under the baseline per-card rate, P(0 flags over ~30 cards) ≈ 1×10⁻⁶; at the cell level, with 4/4 baseline cells flagged, P(4/4 clean) ≤ 0.08 on the pessimistic one-sided bound. Four cells is decisive for a zero and informative for a partial.
- **Mechanism, reported separately:** ambient-capacity flags (baseline 7 commands / 7 verdicts); installed-package-source flags (18 / 4); outside writes or repairs (baseline 1).
- **Vacuous-pass guards:** delivered verdicts vs requests (30 terminal / 43 cycles), closed-card coverage (30/30), `lock.wma.state`, no `timeout`/`failed`/`--no-wma-wait`. A lower rate from fewer reviews is failure.
- **Discipline guard (replaces the unmeasurable recall test):** counts of manual-§4 blocker classes raised as preconditions (LoRA `modules_to_save`, `save_total_limit`, the GenerationConfig save trap, `--max-tokens`/`max_model_len`, concurrency garbage, null decode fields), plus the count and disposition of L0/L1 `no`. Fewer than 5 blocker findings or fewer than 2 L0/L1 negatives → **inconclusive**, not a pass.
- **Falsification:** any frozen flag; any audited outside write/repair; a flag drop with fewer delivered verdicts or lower coverage; loss of a §4 blocker class w10 caught; hand audit finding evasion.
- Manual transcript audit is mandatory — the scanner is lexical and existence-dependent, so zero is necessary, not sufficient.

---

## 3. H — approved. Scope shrunk to the prior correction

Replace only the C6 row of `change_types.md` §3. Type-definition, tier and cost cells stay byte-identical so the label denominator cannot drift.

> | **C6** weight averaging (uniform soup) | average checkpoints of one trajectory | 3 / 4 | which checkpoints; same base and tokenizer establish compatibility, not benefit; whether every ingredient has a score on the same ruler and decode, and whether they are adjacent snapshots of one trajectory or the ends of separate runs / one eval | ≈ 0 on average, but two final submissions were uniform soups beating every ingredient (0.8522 vs 0.8476/0.8423); a variance-reduction move whose gain is bounded by its weakest ingredient — not a default, and never justified by the clock alone | minutes + one eval |

Excluded on purpose (already baseline behavior — see C6 above): offering the two-ingredient subset, demanding same-ruler ingredient scores as a *suggestion*, naming the trajectory relationship, keeping the incumbent under uncertainty. Keeping them would leave you unable to tell the edit's effect from v0.2's.

**Preregistration**
- **Eligibility, adjudicated mechanically:** a soup card is one whose `setup.command` invokes the run's merge script with ≥2 checkpoint inputs. Baseline = **5 cards over 4 cells** (w10r01/exp-07, w10r02/exp-04, w10r03/exp-06, w10r03/exp-09, w10r04/exp-05; 1/1/2/1). Do not use the reviewer's C6 label — it gives 6 and over-counts.
- **Primary A — prior propagation (sharpest, countable):** occurrences of "default when time is short", or any clock-only justification for a merge, in `evidence[].note` or the L3 note of a soup card. Baseline 4 of 5, plus the L3 citation in w10r03/exp-06. Target 0.
- **Primary B — advice content:** fraction of soup cards whose verdict cites a same-ruler score for every named ingredient (or names the missing one as a precondition) **and** states the ingredient relationship. Baseline 1/5 and 2/5.
- **Secondary — uptake, explicitly underpowered:** action on an offered subset or incumbent alternative. Baseline 1/1 where offered; denominator ~2/wave. Report counts, never a rate; **inconclusive at n<3**.
- **Score guard, not a target:** paired same-ruler delta of the chosen model vs the best ingredient, and merge+eval minutes. The edit must not become a two-ingredient mandate — heterogeneous soups are not asserted to be bad.
- **Falsification:** the phrase or a clock-only justification persists; advice appears but no soup card changes its ingredient set, incumbent decision or L3 when a weaker ingredient is present; fewer than 3 soup cards in the wave (**inconclusive**); a soup that would have won is suppressed; C6 label frequency departs materially from w10's 6/30.

---

## 4. I and J — deferred

**I** — defer. The mechanism it proposes is already executed by v0.2 (16/210 suggestions propose a larger n; w10r02/exp-05 explicitly declines a repeat on prefix-identical greedy arms), its determinism premise is contradicted by `change_types.md` §5, and the residual defect is an internal contradiction §5 already overrides. Of the 8 C18-labelled cards, only 4 are genuine ranking cards, and 2 of those 4 (w10r01/exp-08, w10r03/exp-07) *are* the target behavior occurring unprompted. **Revival condition:** a cited case where a v0.2 verdict recommended a C18 repeat *instead of* an available larger n and the scientist spent the evaluation on it. None exists in w10.

**J** — defer. Mechanism base ~2–4 probes of 143; hard ceiling 5.4% of per-cell request time; primary target 20% on a metric with 15.0% CV at n=4 (≈0.07 power at the deliverable effect). It also restates an existing rule (`SKILL.md:120`), and the real cost driver is relocks, which it does not touch. **Revival condition:** either a mechanism count with a baseline rate ≥ ~15% of probes, or a redesign aimed at relock count (43 cycles / 30 cards) rather than probe count — the latter is a protocol candidate, not a skill candidate, and cannot be combined with a skill edit in one candidate.

---

## 5. Shared guards, with two corrections

- **Cost splits into two quantities**, because the baseline's per-cycle cost is not recoverable (C10): **(1)** terminal-verdict mean ≤ 1.5 × w10's `$1.98061 / 6.16373 min` = **`$2.97092 / 9.24560 min`**; **(2)** review cycles per closed card ≤ w10's **43/30 = 1.433** (per cell 1.50 / 1.43 / 1.33 / 1.50 — a tight, low-variance metric) plus a stated tolerance. Both are needed: a candidate that lowers terminal cost while adding relocks passes (1) and fails (2).
- **Do not gate on the 80.11/58.76/69.58/59.66 min lifecycle sums.** Request-to-response, not GPU idle, no 1.5 h breach established, SD 10.01 min at n=4 — they cannot resolve anything under ~30%.
- Do not use self-reported probe `cpu_min` as a cost metric.
- **PTB:** validator-complete, judge-clean cells, same frozen judge; guard only, not below w10's spread (mean .713419, SD .012821, n=4).
- **Leaks:** zero required, with the naming caveat in C4.
- **`c10` is not a comparator** for any verdict-level metric (0 verdicts). Every G/H readout is candidate-vs-w10.
- Four cells is a provisional screen. Promotion still needs the ≥8-cells-per-candidate window and the held-out gate; AIME stays promotion-only.

---

## 6. Launch mechanics — one correction to the packet

I verified these because they determine whether "only the skill hash differs" actually holds:

- `skills/wma/` at HEAD is **byte-identical to `ae46724`** — it does not appear in `git diff --stat ae46724 HEAD -- awm/ skills/ tools/`. Each candidate commit is HEAD's `skills/wma/` plus one edit; v0.2 `176f0a464986` is preserved.
- `WMA_PRIVATE_SHIP = ("awm/__init__.py", "awm/paths.py", "awm/exp_protocol", "awm/wma", "skills/wma")` (`awm/ptb_experiments.py:69-75`). Within that set, **exactly one file diverges from `ae46724` at HEAD: `awm/exp_protocol/lock.py`** (the relock `wma`-history preservation change). `awm/__init__.py`, `awm/paths.py` and `awm/wma/**` are unchanged. Your archive/restore step needs to cover that one file — nothing else.
- `awm/wma_client.py` also diverges (the `say()` flush fix) but is in `PUBLIC_SHIP` only. Public stays pinned at `ae46724`, so it will not enter the cells — which is correct: picking it up would remove the ~8 min of lost turns w10r04 suffered and change the treatment relative to the baseline. **Do not bump the public SHA for this wave.**
- `skills/wma_meta/SKILL.md` diverges at HEAD but is in neither ship list — irrelevant to the cells.

**Queue:** 2 arms × 4 cells = **8 cells**, raising safe PENDING 17 → 25 — one above the hook's 24-job replenishment threshold, far above the hard `>8` floor, allocation unchanged at 16/16. Both arms are independent of `c10r04` and of the w10/c10 extensions and need no wait from either. If you want deeper queue insurance than 25, that is a queue decision, not a science one — the packet's own rule forbids solving it by launching I or J, and I would not add repeats to G or H for capacity either.

One thing I could not settle from the frozen inputs: whether the 4-cell G wave will contain enough manual-§4 blocker findings to read the discipline channel at all. That is why I preregistered it as *inconclusive* rather than *pass* below the stated counts, instead of leaving it as an unmeasurable recall test.

The same review is written to `/home/robtang_google_com/.claude/plans/ultracode-read-only-follow-up-to-wild-sloth.md` in operator-facing form.
