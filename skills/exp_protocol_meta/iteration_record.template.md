# Round NN — <date>

## Variants
| label | commit of skills/exp_protocol | what differs from baseline |
|---|---|---|
| baseline | <sha> | — |
| candidate | <sha> | <one line> |

## Cells
task: <gsm8k> · base model: <id> · scientist: <model, effort> · hours: <N> · seeds per variant: <k>
held-out task this round: <task or "none">

## Results
(paste `awm exp_protocol collect ... --csv`, one row per cell)

| variant | accuracy mean (min–max) | pitfalls_cost_h Σ | n_locked_open Σ | fields_filled mean |
|---|---|---|---|---|

Three cards read by hand per variant: <card paths and one line each>

## Trace review
Link to `<date>-trace-review-round-NN.md`. Per arm: the reviewer header fields (hours to first launch, protocol hours, greedy shipped, RL used, largest eval n, stop reason) and the ranked explanations of the score difference with the number of cells behind each.

## Directions
What changed in `directions-ledger.md` this round: which directions were adopted, screened, deferred, dropped, and why.

## Decision
<promote candidate | keep baseline | inconclusive, rerun with N more seeds>

## Change
<what was edited in skills/exp_protocol, as a diff summary; or "none">

## Evidence
<the specific cards, pitfalls, or numbers the change is traceable to>

## Next round
<what to try, and why>
