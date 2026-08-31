"""Write the study's prompt files into a PostTrainBench checkout.

``get_prompt.py`` loads ``src/eval/general/${POST_TRAIN_BENCH_PROMPT}.txt`` and
fills ``{model} {benchmark} {num_hours} {gpu_info} {setup_other}
{decontamination_tool} {eval_api_note}`` by plain replacement, so every prompt
here is written in those placeholders. Five files (the C1 derivative is
emitted when a PTB checkout is supplied):

    prompt_fulltraj.txt      PTB prompt + a "Prior runs" section          (C1: raw files, no WMA)
    prompt_wm.txt            our instruction.md in PTB placeholders        (C3: WMA, memory only)
    prompt_wm_fulltraj.txt   the same + the "Prior runs" section           (C2: raw files + WMA)
    prompt_wm_smoke.txt      C3 prompt + a one-hour lifecycle directive    (optional smoke)
    prompt_wm_fulltraj_smoke.txt  C2 prompt + that lifecycle directive     (release smoke)

plus copies under rollout/prompts/ so the rendered text is reviewable in this
repo. Run by rollout/setup.sh:

    python rollout/build_prompts.py <ptb checkout>
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PRIOR_RUNS_SECTION = """## Prior runs
- `/home/ben/prior_runs/` holds read-only published trajectory records of previous attempts at this same task (post-training a base model for {benchmark} under these rules) by other autonomous agents. Start with `/home/ben/prior_runs/INDEX.md`: one row per run with its base model, the agent that ran it, its official accuracy, and its path.
- Every run directory has exactly `solve_out.txt` (the complete session trajectory), `metrics.json` (the official score), and `time_taken.txt`. Optional upstream artifacts and `task/` workspace snapshots are deliberately excluded.
- The fetched corpus does not include prior `task/` workspace snapshots, scripts, checkpoints, or private caches. Reimplement any idea or recipe you infer from a trace in your own workspace; do not assume a referenced prior-run file is available.
- You may read anything that is present there and reuse any idea or recipe. Every rule below still applies to what you do with it: a prior run's training data is not exempt from the contamination rules, and a prior run's base model is not yours unless it is `{model}`.
- Reading files under `/home/ben/prior_runs` is an allowed exception to the working-directory rule.

"""

PINNED_BASE_SECTION = """## Pinned base checkpoint
- This study permits only the official `google/gemma-3-4b-pt` base at revision `cc012e0a6d0787b4adcc0fa2c4da74402494554d`. It is already complete at the immutable read-only path `/home/ben/pinned-base/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
- Load that exact snapshot locally. Do not inspect cache credential files, authenticate to Hugging Face, download a substitute, or use any mirror, fork, repack, instruction-tuned model, or other revision (including an `unsloth/*` model).

"""

SMOKE_SECTION = """## One-hour smoke protocol
This is a labelled integration smoke, not a production research cell. Exercise one complete WMA-controlled training lifecycle; do not spend the hour on a broad corpus survey or open a second card. The WMA remains autonomous and chooses which available precedents to inspect and cite.

- By minute 10, create one minimal valid card from permitted train-derived evidence, propose it, read the brief, and explicitly reply `accept` or `override`.
- Use exactly three train-derived watch examples, an official evaluator with `--limit 4`, and one optimizer step from the pinned official base. The smoke tests mechanics, not model quality.
- By minute 35, save a complete loadable checkpoint and call the checkpoint hook with `step >= 1` and `final=True`; when it returns `3`, exit the trainer so the WMA worker can evaluate.
- Read the resulting decision ping and explicitly reply `select:<obs-id>` for that observation. Do not rely on a timeout action.
- By minute 50, fill card sections 5–6 and run `awm wm finalize ...` with decision `adopt`, leaving a complete `final_model/` for the official evaluator.

"""

# instruction.md placeholders -> PTB placeholders / sandbox constants
INSTRUCTION_MAP = {
    "{dir}": "/home/ben/task",
    "{submission}": "/home/ben/task/final_model",
    "{time_limit}": "{num_hours} hours",
    "{gpu}": "one Nvidia H100 GPU",
}


def ptb_fulltraj(ptb_prompt: str) -> str:
    """The PTB prompt with prior runs and the pinned-base rule before ## Rules."""
    anchor = "## Rules"
    if anchor not in ptb_prompt:
        raise SystemExit("PTB prompt.txt has no '## Rules' heading; update build_prompts.py")
    return ptb_prompt.replace(anchor, PRIOR_RUNS_SECTION + PINNED_BASE_SECTION + anchor, 1)


def wm_prompt(instruction: str, *, fulltraj: bool) -> str:
    """Our instruction.md rendered into PTB placeholders."""
    text = instruction
    for k, v in INSTRUCTION_MAP.items():
        text = text.replace(k, v)
    # PTB's per-benchmark fill-ins: the inspect-ai note and the decontamination tool
    env_anchor = "- A copy of the `{benchmark}` test set and a contamination checker are available."
    if env_anchor not in text:
        raise SystemExit("instruction.md changed: the test-set/contamination bullet is missing")
    text = text.replace(env_anchor, "{setup_other}{decontamination_tool}\n" + env_anchor, 1)
    if fulltraj:
        anchor = "## The world-model agent"
        if anchor not in text:
            raise SystemExit("instruction.md changed: '## The world-model agent' heading missing")
        text = text.replace(anchor, PRIOR_RUNS_SECTION + anchor, 1)
    leftover = [p for p in ("{dir}", "{submission}", "{time_limit}", "{gpu}") if p in text]
    if leftover:
        raise SystemExit(f"unrendered placeholders: {leftover}")
    return text


def smoke_prompt(prompt: str) -> str:
    """Add the release-smoke deadlines without changing production prompts."""
    anchor = "## Research"
    if anchor not in prompt:
        raise SystemExit("WMA prompt changed: '## Research' heading missing")
    return prompt.replace(anchor, SMOKE_SECTION + anchor, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptb", nargs="?", type=Path)
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="write only into the supplied private PTB checkout",
    )
    args = parser.parse_args()
    ptb = args.ptb.resolve() if args.ptb else None
    if args.no_review and ptb is None:
        parser.error("--no-review requires a PTB checkout")
    instruction = (ROOT / "input" / "instruction.md").read_text()
    out_review = None if args.no_review else HERE / "prompts"
    if out_review:
        out_review.mkdir(exist_ok=True)
    wm = wm_prompt(instruction, fulltraj=False)
    wm_fulltraj = wm_prompt(instruction, fulltraj=True)
    files = {
        "prompt_wm.txt": wm,
        "prompt_wm_fulltraj.txt": wm_fulltraj,
        "prompt_wm_smoke.txt": smoke_prompt(wm),
        "prompt_wm_fulltraj_smoke.txt": smoke_prompt(wm_fulltraj),
    }
    if ptb:
        ptb_prompt = (ptb / "src" / "eval" / "general" / "prompt.txt").read_text()
        files["prompt_fulltraj.txt"] = ptb_fulltraj(ptb_prompt)
    for name, text in files.items():
        if out_review:
            (out_review / name).write_text(text)
        if ptb:
            (ptb / "src" / "eval" / "general" / name).write_text(text)
    destinations = []
    if out_review:
        destinations.append(str(out_review))
    if ptb:
        destinations.append(str(ptb / "src/eval/general"))
    print(f"wrote {', '.join(files)} to {' and '.join(destinations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
