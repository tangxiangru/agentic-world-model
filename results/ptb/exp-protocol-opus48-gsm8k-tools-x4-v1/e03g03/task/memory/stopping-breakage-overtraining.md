---
name: stopping-breakage-overtraining
description: Full-FT SFT of a base LLM can silently lose EOT stopping, tanking a last-number grader
metadata:
  type: feedback
---

When SFT-ing a base model (gemma-3-4b-pt) to answer then stop at `<end_of_turn>`, over-training (too many steps / large heterogeneous data) makes the model an over-fluent generator: it emits the correct first `ANSWER: N` but keeps going with new hallucinated problems. A last-number scorer then reads a wrong later number → scored accuracy collapses even though FIRST-answer accuracy is high.

**Why:** more update steps on clean single-block targets paradoxically weakened the relative probability of EOT after the answer, and the eval's 10-shot multi-block context primed continuation.

**How to apply:** always diagnose eval logs with BOTH first-answer and last-number accuracy plus stop_reasons (`stop` vs `max_tokens`). If first >> last, it's a stopping bug, not reasoning. Observed thresholds for this model (eff batch 16): ~1400-4000 steps = clean stopping; ~4200+ steps or heterogeneous data (MetaMath) = broken. Prefer moderate steps, homogeneous clean-EOT data, and consider training in the eval's few-shot format to teach stop-after-final-block. See [[gsm8k-gemma-rft]].
