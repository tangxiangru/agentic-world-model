export const meta = {
  name: 'wm-x-reverify-r0-29',
  description: 'Third-round independent verification of cell r0-29 after the orchestrator applied the round-2 finding on r0-29-exp-05 by hand',
  phases: [{ title: 'Re-verify', detail: 'one Opus skeptic on r0-29', model: 'opus' }],
}
const REPO = '/home/kalorona/agentic-world-model'
const DATA = '/home/kalorona/awm-data'
const VERIFY_SCHEMA = {
  type: 'object',
  properties: { results: { type: 'array', items: { type: 'object', properties: { checkpoint_id: { type: 'string' }, verdict: { type: 'string', enum: ['confirmed', 'needs_fix', 'cannot_verify'] }, high: { type: 'integer' }, medium: { type: 'integer' }, note: { type: 'string' } }, required: ['checkpoint_id', 'verdict', 'high', 'medium'] } } },
  required: ['results'],
}
const cell = 'r0-29'
const prompt = `You are the verifier agent for one scientist session. Cell: \`${cell}\`. This is the THIRD round: after your second-round finding on r0-29-exp-05 (second soup input recorded only as free text in other_inputs), the reviewing orchestrator moved that input into inputs.additional_parents with produced_by_step = s07_select_out_sft2_checkpoint-2109 (see the record's "revisions" array). Check every record of the cell from scratch, with particular attention to that step's inputs against the trace.

First read, in full:
1. ${REPO}/tools/wm_benchmark/verify_prompt.md  (your instructions)
2. ${REPO}/tools/wm_benchmark/extract_prompt.md  (what the extractor was told; describes the timeline layout)
3. ${REPO}/tools/wm_benchmark/launch_record.schema.json
4. ${DATA}/benchmark/targets/${cell}.json

Then, for every record in ${DATA}/benchmark/x_raw/${cell}/*.json, do the checks against the timeline at ${DATA}/timeline/${cell}/ and write your verdict to ${DATA}/benchmark/x_verify/${cell}/<checkpoint_id>.json (write with the Write tool; never rm anything, never edit the extractor's records). Be adversarial: open every cited seq yourself. Return the structured per-checkpoint results.`
const v = await agent(prompt, { label: `reverify3:${cell}`, phase: 'Re-verify', agentType: 'general-purpose', model: 'opus', effort: 'high', schema: VERIFY_SCHEMA })
return v