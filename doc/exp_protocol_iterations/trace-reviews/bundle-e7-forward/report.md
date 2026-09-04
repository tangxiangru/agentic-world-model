# E7 independent forward review — model-free, published-guide workflow

Scope: repository `/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo`.
All reviewer artifacts: `/tmp/e7-forward-review.zLMH5o/` (called ROOT below).
No implementation/test sources were read to design fixtures; only public guides,
CLI help, pyproject's console entrypoint, resulting artifacts and error output.
No repository/index/commit changes; no model weights/forwards/training/evaluation,
GPU devices, network, downloads, package installation or Slurm access.

## Identity and reproduction

Full guide/helper hashes are retained in `source-hashes.txt`.
Rendered helper SHA256 stayed `9a30b7f00852315d6816c5adea1f368eadd7cdb2ac3cb34610efbb01597bdd12`.
Guide initially: `e30229cee9015290dcd9d0de50700693b1f6871479dad3c7d654b745e70a600b`.
Guide later: `c91faca74ce7ef1e4ae5f46d7ae42968428591a3f72368f224e7743fe2720167`.
Only its construction-measurement appendix changed during review; reread fully.
The runtime SKILL stayed `c5e57bb1061caf7fdf0d50832cc16f5f96feba4c222d3264eb12c8ad8fd385fa`.
`offline.sh` reproduces the guide's bwrap runtime with read-only host filesystem,
only ROOT writable, isolated network, fresh /dev, offline HF settings and
`TOKENIZERS_PARALLELISM=false`. Native versions observed: Transformers 4.57.3,
tokenizers 0.22.2. Only the supplied local Gemma tokenizer was loaded.
`capture.py` records exact argv, timestamps, return codes, timings and full output
to `transcript.jsonl`; early setup caveats are in `initial-observations.md`.
`awm_cli.py` uses published `awm.cli:main` with `raise SystemExit(main())`.

Recheck retained valid preparation without training or overwriting evidence:

```bash
review_root=/tmp/e7-forward-review.zLMH5o
bash "$review_root/offline.sh" "$review_root/prepare_v2.py" separate_concat raw.jsonl separate-v3 --reuse
bash "$review_root/offline.sh" -c 'from awm.exp_protocol.rendered_training import RenderedTrainingBundle as B; print(B.verify("/tmp/e7-forward-review.zLMH5o/joint-v3/receipt.json"))'
```

For a fresh preparation choose a new output name; preserve existing failed and
successful directories. The three filled cards and `memory/index.md` show the
create/check/lock/run/close sequence. No scientific performance claim is recorded.

## Successful workflow observed

- Synthetic `raw.jsonl`: four source occurrences, including two equivalent
  question/answer occurrences with different IDs, Unicode, two ANSWER markers in
  a prompt's demonstrations, bare raw completion bodies and an overlength prompt.
- Pure source-backed `preprocessing.py` appends EOT plus newline. `prepare_v2.py`
  exercises both separate-concat/pre-rendered and joint-prefix/template-replay.
- Both successful bundles reconcile four raw rows into three kept rows and one
  2,820-token overlength drop. Kept lengths 49, 38, 49; total 136; supervised 40.
  Duplicate occurrences remain separately located. No hidden dedup or truncation.
- Stop fraction 1.0 and marker-bad fraction 0.0: prompt demonstration markers do
  not count against supervised targets; explicit newline tail is accepted.
- Separate report accurately says template-reference-only/equivalence-unverified.
  Joint report says prompt-and-full-replayed, not independently serving-equivalent.
- `exp-02` separate receipt SHA:
  `9e73e770685acc3503f466af9c752656c9637d74b4bb58dcf2d92d8fc9d9eea8`.
  `exp-03` joint receipt SHA:
  `cf30b4cc87de16ff7c03dd6d7ac81a4ae121b68eaa407722bb2fcfb4298c3447`.
- Cards point setup.data at actual tokens.jsonl and retained n=3; raw provenance
  stays inside receipts. CLI check and lock passed without overrides.
- Preflight explicitly SKIPs all three superseded raw heuristics, says "not a raw
  PASS", and leaves model consumption unknown. No comparator was fabricated.
- `consume_v2.py` opens the locked card, accesses all rows, collates and flushes.
  Native Gemma left-padding rounds width 49 to 56. Using attention masks, actual
  input and label arrays are preserved; padding labels are -100 and attention 0.
  Each row has one BOS; supervised counts are 13, 14, 13. Both commands exited 0.
- Attempts: exp02 `02850aa28e7d4fe1b6de84e4cf66075b`; exp03
  `1dc06a7627114dd7bf472e8ea73d87f0` under `memory/attempts/`.
  stdout.txt records assertions/rejections; finish.json distinguishes child exit,
  unverified artifact namespace and scientific validation not performed.
- `memory/rendered-consumers/exp-02/` and `/exp-03/` distinguish loader binding,
  row access and collation, lower-bound counts and model_consumption unknown.
  Unchanged `reuse=True` returned the same receipt hash successfully.

## Independent adverse cases

1. Missing supervised EOT (`bad-stop.jsonl`) fails preparation with explicit stop
   threshold error; retained `bad-stop-output/` permits inspection.
2. Malformed raw second line fails with exact `malformed.jsonl:2` locator rather
   than silently disappearing; retained partial output remains.
3. My initial Jinja fixture trimmed the newline after the role. Native output was
   `...userOne plus one?...`; template replay correctly refused it. Original
   `chat-initial.jinja` and failed `joint/` remain; corrected template passed.
4. Opening the filled-but-unlocked exp02 card refused missing successful lock.
5. Modified feature label and missing bundle identity each refused collation.
6. `mixed_probe.py` mixed rows from the two valid bundles; collation refused their
   different receipt identities even though their token arrays were equivalent.
7. `flush_probe.py` opened exp03, collated and explicitly flushed, then paused.
   A reviewer apply_patch changed its locked hypothesis. The next explicit flush
   refused `consumer card/lock/receipt changed after binding` (`flush-probe.json`).
   The original plan was restored exactly; the probe is documented in the card.

## False-block assessment and guide/claim gaps

No demonstrated E7 false block in these supported cases. Two reviewer mistakes
were preserved rather than blamed on the utility:

- My initial renderer used settings.mode; runtime supplies a dict, requiring
  settings['mode']. The guide does not explain this callback argument shape;
  preparation's RenderedSettings object makes that distinction easy to miss.
- Initial consume.py assumed right padding. Actual arrays show consistent native
  left-padding. Exp01 records the failed assertion, was closed inconclusive, and
  a new source/preparation/card supplies mask-aware assertions. It is not an E7 bug.
  The early CLI shim also omitted SystemExit(main()); its captured zero status
  was my shim error, not a product exit-propagation defect. Actual finish retained 1.

Runtime SKILL currently links save/sampling guides but not rendered-training.md;
without the review brief's explicit path, this optional workflow is hard to discover.
The guide's actual-model-consumption and serving-equivalence limitations match the
records. This audit confirms neither arbitrary script coverage nor optimizer use.
Packed/multispan data, worker processes, backend mutations and large-volume behavior
were not independently exercised here. High drop rates require scientific judgment;
the fixture deliberately allowed 25% through explicit max_drop_fraction=0.30.

Local observed costs, including Python/Gemma-tokenizer startup: preparation about
20–21 seconds for four raw rows; unchanged reuse 22.4 seconds; CLI lock 2.7–2.8 seconds;
guarded CPU consumer about 6 seconds. These are measured local costs, not scale or
training-cost predictions, and do not reproduce the builder's separate benchmark.
All three cards are closed, with inconclusive scientific verdicts and no checkpoint.
