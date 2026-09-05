# Opus4.8 E wave1: independent trace review

Four canonical completed results: e03g01=.637604,e03g02=.542835(flagged),e03g03=.427597,e03g04=.583017. Mean all4=.547763; clean3=.5494061157442507. These are descriptive; compare only matched Opus4.8 controls, not historic Opus5.

Most consequential correction: e02/e03 falsely claim greedy while final JSON contains only do_sample:false. Explicit temperature0 exists only e01/e04. Actual final-eval request fields, submitted generation JSON and primary pinned-vLLM source are retained. No same-weight correction run has been performed, so score gain unknown.

E02 failed because single-card RFT declared a future training file; _live_plan's hard hash requirement rejected both run and record_vllm despite lock override. E04 explicitly used two cards and succeeded. Documentation says existing input hashes are required, so this is a stage/workflow incompatibility and misleading override/readiness surface, not proof hashes should be relaxed. Minimal CPU predicate reproduction is retained (earlier plan/lock checks stubbed; not GPU end-to-end).

E01 and E03 also lose~.9h/~1.4h to unsupported vLLM prompt_token_ids keyword and delayed failure observation. Candidate components therefore have independent cell support: pre-engine readiness+native API example; explicit sample/persist/train stage pattern; actual served-decoder evidence. Preserve scientist strategy freedom and strict real training inputs. No automatic temperature rewrite, no loose hash bypass, no global process kill.

E features actually adopted differ by cell. RenderedTrainingBundle appears in e01 only; SaveSafeTrainer in e01/e03, GenerationSaveContract also e04; guarded-run attempts cover e02 first4, e03 all12, e04 nine; record_vllm reaches no successful capture across4. e02 later bypasses the wrapper/helper after its refusal. Do not equate availability with use, or annotate the classifier's9.18h as protocol paperwork: wrapper-run compute is misclassified.

No executed RL in these4; all try RFT. This disproves a claim that E universally prevents RFT, but does not establish RFT superiority. Stopping-related score regressions accompany multiple data branches; e04's few-shot branch recovers, while e01's teacher-data augmentation helps and later RFT hurts. First-answer correctness is diagnostic and not a substitute official metric. No BF16 update measurement exists, so precision-related low-score causation remains a hypothesis.

Reports e03g01.md–e03g04.md include recipes, runtime, actual decode, failure/success evidence, cards and limits. Source facts/timelines retain auto-tool errors for audit; corrected launch evidence is in execution-and-decode.json. Future-data failure/anomaly source is in e03g02 report plus raw judgement.

Claude session: none. Default launch failed EROFS; two allowed elevated attempts hit automatic-approval timeout. No actual helper trace reads were observed, and no session was falsely marked started. Independent Codex analysis proceeded; Claude cross-review remains an explicit incomplete contractual step.
