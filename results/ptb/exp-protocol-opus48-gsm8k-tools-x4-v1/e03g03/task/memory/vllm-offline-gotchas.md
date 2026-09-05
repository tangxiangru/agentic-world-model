---
name: vllm-offline-gotchas
description: vLLM 0.11 offline sampling API + engine-hang traps that cost ~1.4h
metadata:
  type: reference
---

vLLM 0.11.0 offline sampling for RFT/self-sampling:
- `LLM.generate()` has NO `prompt_token_ids` kwarg. Pass `from vllm import TokensPrompt` → `llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in ...], sampling_params=params)`. Passing pre-rendered strings would add a SECOND `<bos>` (Gemma template already emits one), so pass token ids (or add_special_tokens=False).
- Set `SamplingParams(stop_token_ids=[1,106])` for gemma3 (<eos>, <end_of_turn>).
- A crash after the engine loads leaves the vLLM EngineCore subprocess HANGING at interpreter shutdown; it held the GPU for ~1.4h until killed (`kill -KILL -<pgid>`; `pkill -9 -f EngineCore`). Verify the API on the class signature BEFORE launching a model run (you can't cheaply smoke-test under the protocol's lock rule).
- `os._exit(0)` at end of a sampling script avoids the normal-exit hang but ORPHANS the EngineCore subprocess (keeps ~66GB GPU) — `pkill -9 -f EngineCore` afterward before the next GPU job.
See [[gsm8k-gemma-rft]].
