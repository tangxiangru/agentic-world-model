# g01r03 exp08: stored sample-array prefix is not the dataset prefix

Planner read-only audit,2026-09-04. This corrects the scientist card and first synthesis; no evaluation, model, score, validator, queue or frozen runtime was changed. Exact raw log paths/hashes and derived counts are in `g01r03-prefix-audit.json`; source receipt/manifest/spec remain in `roster.json`.

The exp08 card explicitly claims a150-item comparator extracted from the earlier800-item soup evaluation. It is **not** simply calling0.84 the entire n800 score. However its referenced scalar file `task/eval/exp-07_soup_dev800.json` actually contains accuracy0.83875 and stderr0.013010462599243267, with no n. The card's0.84 is not that file's scalar value.

The trace pinpoints the extraction error: at02:45:11 UTC, original decompressed L9026 executes `s=d["samples"][:150]`; the following output is labelled “soup accuracy on the first150 of the dev800 run” and prints0.84. Structured parsing reproduces126/150 correct in this stored-array slice, but **122 of its150 IDs are outside the actual dataset-first150**.

Using `eval.dataset.sample_ids[:150]` instead, and joining typed `[id,epoch]` records (epoch1), the old soup log has **128/150** correct. The fresh default evaluation has **127/150**, with2 old-only and1 new-only correct. Its declared150 IDs equal the old dataset prefix in both order and set; aligned original inputs and targets have0 mismatches. The actual matched-subset contrast is therefore−1/150, not the card's+1/150. This is not a demonstrated accuracy regression: the evaluation size, concurrency32→2 and GPU-memory fraction0.8→0.3 also change. No effect is isolated or attributed to floating-point batching.

The official PTB score remains1092/1319=0.8278999241849886, validator-complete/eligible and judge-clean. The large-n selection and final path loading evidence remain; what fails is the recorded comparator identity/subset derivation and its claimed one-item sign. This audit does not establish training contamination or quantify a score effect of the protocol violation.

For synthesis/candidate design: replace the filename-only classification with this actual artifact/card/trace evidence. A scientist-declared comparator n=150 alone could pass while pointing to the wrong subset and scalar file. Such a field can improve declared-n consistency but must not be called independent artifact/count verification. Proper subset evidence requires declared dataset IDs and a binding to the source measurement; do not infer sample order from a JSON array. No candidate is selected by this audit.
