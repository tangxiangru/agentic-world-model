# Initial bootstrap observations

- Read AGENTS.md, runtime SKILL.md, rendered-training.md, save-safety.md,
  card.template.yaml, execution-records.md and pitfalls.yaml; did not read implementation/tests.
- `python -m awm --help` is unavailable (`No module named awm.__main__`).
  Read only pyproject.toml's published console entrypoint `awm.cli:main` and
  added a local shim for the pinned Python interpreter. `exp_protocol --help` works.
- First synthetic separate-mode prepare failed at renderer `settings.mode`:
  `AttributeError: 'dict' object has no attribute 'mode'`. The guide names
  explicit settings but does not specify that the callback receives a dict,
  although preparation accepts a RenderedSettings instance. Updated only the
  reviewer formatter to use settings['mode']; preserved failed output `separate/`.
- Initial native template replay correctly refused my mismatched formatter:
  Jinja trimming removed the newline after the role, producing `userOne...`.
  Preserved chat-initial.jinja and failed joint/; corrected only my template.
- First consume.py assertion assumed right padding; Gemma's collator correctly
  left-padded to56. Preserved exp01 failed attempt and source; new consume_v2.py
  uses the returned attention mask. This is not an E7 defect.
- First CLI shim omitted the installed-entrypoint SystemExit wrapper, losing
  main()'s nonzero return. Fixed my shim to raise SystemExit(main()); initial
  capture return0 is not evidence of a product exit-propagation defect.
