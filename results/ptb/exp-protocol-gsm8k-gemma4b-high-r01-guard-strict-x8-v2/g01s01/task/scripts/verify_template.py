"""Render one example through the grader's real jinja template and through
fmt.render_prompt, and assert they are byte-identical.

Guards pitfall `template_unreachable`.
"""
import sys

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer

sys.path.insert(0, "/home/ben/task/scripts")
import fmt  # noqa: E402

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def jinja_render(messages, template_src, bos_token):
    def raise_exception(msg):
        raise TemplateError(msg)

    env = Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = raise_exception
    tmpl = env.from_string(template_src)
    return tmpl.render(messages=messages, bos_token=bos_token, add_generation_prompt=True)


def main():
    tok = AutoTokenizer.from_pretrained(SNAP)
    src = open(fmt.TEMPLATE_PATH).read()
    print("template sha12:", fmt.template_sha())

    q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    system = "EXEMPLAR ONE\n\nEXEMPLAR TWO"

    for sysmsg in (None, system):
        msgs = []
        if sysmsg:
            msgs.append({"role": "system", "content": sysmsg})
        msgs.append({"role": "user", "content": fmt.user_prompt(q)})
        ref = jinja_render(msgs, src, tok.bos_token)
        ours = fmt.render_prompt(q, sysmsg)
        assert ref == ours, (
            "MISMATCH\n--- jinja ---\n" + repr(ref) + "\n--- ours ---\n" + repr(ours)
        )
        print(f"OK  system={'yes' if sysmsg else 'no '}  len={len(ours)}")

    # terminator check: the token the grader stops on
    eot_ids = tok.encode(fmt.EOT, add_special_tokens=False)
    print("EOT token ids:", eot_ids, "-> expect [106]")
    assert eot_ids == [106], eot_ids
    bos_ids = tok.encode(fmt.BOS, add_special_tokens=False)
    print("BOS token ids:", bos_ids, "-> expect [2]")
    assert bos_ids == [2], bos_ids

    full = fmt.render_prompt(q, None) + fmt.render_completion("Some reasoning.\n\nANSWER: 72")
    ids = tok.encode(full, add_special_tokens=False)
    print("round-trip decode ok:", tok.decode(ids) == full)
    print("last 3 ids:", ids[-3:])
    print("\n--- sample full row ---\n" + full)


if __name__ == "__main__":
    main()
