# Frozen forward cases, selected from guides only

No implementation or test source was read to select these cases.
This is native-type CPU integration with inert engine returns, not inference.

1. Prepare already-rendered Gemma prompts, including identical prompt text and
   repeated typed IDs (integer 7 and string "7"), preserve n=2 draws with explicit
   EOT stop and an additional string stop, chosen temperature/top_p/seed.
   Include a native EOS completion whose stop_reason is null.
2. A developer numeric parser fails on one overflow/non-finite answer, then a
   corrected parser reprocesses the original raw recording without rewriting it.
3. Interrupt a parsing pass with KeyboardInterrupt; investigate durable state,
   then recover with a new parsing pass.
4. A returning native batch has fewer than the requested draws: inspect preservation
   of all returned raw evidence and whether parsing incorrectly certifies it.
5. A returning batch exchanges two nonidentical prompts: check identity detection
   and preservation, separately from the admitted identical-prompt order boundary.
6. A native completion reports abort: inspect raw preservation and success state.
7. Static preparation uses accidental immediate double-BOS or an unknown/multitoken
   required stop. Record usability and whether a real tokenizer catches it.
8. A prepared recording directory is accidentally reused; verify first evidence
   remains intact. Confirm the lock and native source/type checks stay active.
