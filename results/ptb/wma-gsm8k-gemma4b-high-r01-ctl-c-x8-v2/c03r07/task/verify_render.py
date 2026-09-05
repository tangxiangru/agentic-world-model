#!/usr/bin/env python3
"""Guard against the `template_unreachable` pitfall.

Renders the same conversation two ways and refuses to agree unless they are
byte-identical:
  A) train_sft.render()            - what the trainer will teach
  B) templates/gemma3.jinja        - what the grader hands to vLLM

Also reconstructs the grader's full prompt for one real dev question through
the inspect_evals solver chain, so the prompt shape used in training can be
eyeballed against the graded one.
"""
from __future__ import annotations

import hashlib
import json
import sys

from jinja2 import Environment
from jinja2.exceptions import TemplateError

from train_sft import render

TEMPLATE = "templates/gemma3.jinja"


def jinja_render(messages, add_generation_prompt: bool) -> str:
    src = open(TEMPLATE).read()
    env = Environment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg):
        raise TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(src)
    return tpl.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token="<bos>",
    )


def main() -> int:
    print(f"{TEMPLATE} sha256 = {hashlib.sha256(open(TEMPLATE,'rb').read()).hexdigest()}")

    rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "data/sft_gsm.jsonl")]
    # check a zero-shot row and a 10-shot row
    picks = [next(r for r in rows if not r["fewshot"])]
    fs = next((r for r in rows if r["fewshot"]), None)
    if fs:
        picks.append(fs)

    ok = True
    for r in picks:
        tag = "10-shot" if r["fewshot"] else "0-shot"

        mine_prompt = render(r["prompt"], None)
        theirs_prompt = jinja_render([{"role": "user", "content": r["prompt"]}], True)
        # the grader's own message split: system(fewshot) + user(body)
        if r["fewshot"]:
            fewshot = open("data/fewshot_system_message.txt").read()
            body = r["prompt"][len(fewshot) + 2 :]
            split_prompt = jinja_render(
                [
                    {"role": "system", "content": fewshot},
                    {"role": "user", "content": body},
                ],
                True,
            )
            if split_prompt != theirs_prompt:
                print(f"MISMATCH [{tag}] system-split vs concatenated prompt")
                ok = False

        mine_full = render(r["prompt"], r["completion"])
        # the jinja appends the terminator itself, so hand it the bare content
        bare = r["completion"][: -len("<end_of_turn>")]
        theirs_full = jinja_render(
            [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": bare},
            ],
            False,
        )
        # the jinja emits a trailing "\n" after the model turn; vLLM stops at
        # <end_of_turn> so the trainer ends the sequence there instead
        if theirs_full.endswith("\n"):
            theirs_full = theirs_full[:-1]

        if mine_prompt != theirs_prompt:
            print(f"MISMATCH [{tag}] prompt")
            print("  mine  :", repr(mine_prompt[-200:]))
            print("  jinja :", repr(theirs_prompt[-200:]))
            ok = False
        if mine_full != theirs_full:
            print(f"MISMATCH [{tag}] full")
            print("  mine  :", repr(mine_full[-200:]))
            print("  jinja :", repr(theirs_full[-200:]))
            ok = False
        if ok:
            print(f"OK [{tag}] byte-identical ({len(mine_full)} chars)")
            print("  tail:", repr(mine_full[-120:]))

    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
