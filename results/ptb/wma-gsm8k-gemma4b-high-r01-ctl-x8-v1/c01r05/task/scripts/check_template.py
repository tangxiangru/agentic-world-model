"""Pitfall `template_unreachable`: render one conversation with the grader's own
templates/gemma3.jinja and with scripts/common.py, and require byte equality.

Also prints the sha256 of the template and of the gsm8k task file so a later
card can prove they did not move.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer  # noqa: E402

import common  # noqa: E402

TASK = Path("/home/ben/task")
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TPL = TASK / "templates" / "gemma3.jinja"
GSM = Path("/usr/local/lib/python3.10/dist-packages/inspect_evals/gsm8k/gsm8k.py")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(SNAP)
    template = TPL.read_text()

    fewshot = common.fewshot_example(
        "Bob has 3 apples and buys 4 more. How many does he have?",
        "Bob starts with 3 apples.\nHe buys 4 more, so he has 3 + 4 = 7.",
        "7",
    )
    question = "A train goes 60 km/h for 2 hours. How far does it go?"

    ok = True
    for name, fs in (("zero-shot", None), ("few-shot", fewshot)):
        # ---- the grader's path -------------------------------------------
        msgs = []
        if fs:
            msgs.append({"role": "system", "content": fs})
        msgs.append(
            {"role": "user", "content": common.MATH_PROMPT_TEMPLATE.format(prompt=question)}
        )
        theirs = tok.apply_chat_template(
            msgs, chat_template=template, tokenize=False, add_generation_prompt=True
        )
        # ---- our path -----------------------------------------------------
        ours = common.render_prompt(common.user_prompt(question, fs))
        same = theirs == ours
        ok &= same
        print(f"[{name}] match={same}")
        if not same:
            print("  grader:", repr(theirs[:400]))
            print("  ours  :", repr(ours[:400]))

        # ---- full turn, with the assistant message, to check the terminator
        msgs2 = msgs + [{"role": "assistant", "content": "Bob has 7.\n\nANSWER: 7"}]
        theirs2 = tok.apply_chat_template(
            msgs2, chat_template=template, tokenize=False, add_generation_prompt=False
        )
        ours2 = ours + common.render_completion("Bob has 7.\n\nANSWER: 7")
        same2 = theirs2 == ours2
        ok &= same2
        print(f"[{name}] full-turn match={same2}")
        if not same2:
            print("  grader:", repr(theirs2[-200:]))
            print("  ours  :", repr(ours2[-200:]))

    eot_ids = tok.encode(common.EOT, add_special_tokens=False)
    print("eot token ids:", eot_ids, "(generation_config eos_token_id = [1, 106])")
    print("template sha256[:16]:", sha(TPL))
    print("gsm8k task sha256[:16]:", sha(GSM))
    print("RESULT:", "PASS" if ok and eot_ids == [106] else "FAIL")
    return 0 if ok and eot_ids == [106] else 1


if __name__ == "__main__":
    raise SystemExit(main())
