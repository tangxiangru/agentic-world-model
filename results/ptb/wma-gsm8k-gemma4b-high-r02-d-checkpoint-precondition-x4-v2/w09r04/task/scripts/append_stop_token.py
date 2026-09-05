"""Append the template terminator to sampled targets.

vLLM's returned `text` excludes the stop token, so a jsonl written straight from
rft_sample.py has targets that end at the answer digit. Training on those would
teach the model never to emit <end_of_turn> -- the exact eos_mismatch pitfall.
This step makes the target end where the grader's template ends the turn.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STOP_TOKEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = 0
    with Path(args.out).open("w") as fh:
        for line in Path(args.inp).open():
            r = json.loads(line)
            c = r["completion"].rstrip()
            if not c.endswith(STOP_TOKEN):
                c += STOP_TOKEN
            r["completion"] = c
            fh.write(json.dumps(r) + "\n")
            n += 1
    print(f"wrote {args.out} ({n} rows, all ending in {STOP_TOKEN})")


if __name__ == "__main__":
    main()
