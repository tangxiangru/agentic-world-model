export const meta = {
  name: 'wm-x-extraction-r2',
  description: 'Extract X (launch chains) for the 100 remaining cells with Opus agents, verify all 124 cells with independent Opus skeptics, repair and re-verify where the verifier objects',
  phases: [
    { title: 'Extract', detail: 'one Opus extractor per remaining cell → x_raw/<cell>/<ckpt>.json', model: 'opus' },
    { title: 'Verify', detail: 'one Opus skeptic per cell (incl. the 24 already extracted) → x_verify/<cell>/<ckpt>.json', model: 'opus' },
    { title: 'Repair', detail: 'only cells with needs_fix verdicts', model: 'opus' },
    { title: 'Re-verify', detail: 'verifier again on repaired cells', model: 'opus' },
  ],
}

const REPO = '/home/kalorona/agentic-world-model'
const DATA = '/home/kalorona/awm-data'
const already = new Set(args.already_extracted)
const cells = [...args.already_extracted, ...args.remaining]
const MODEL = 'opus'
const EFFORT = 'high'

const EXTRACT_SCHEMA = {
  type: 'object',
  properties: {
    checkpoints: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          checkpoint_id: { type: 'string' },
          archived_from_dir: { type: ['string', 'null'] },
          archive_submit_seq: { type: ['integer', 'null'] },
          producing_launch_seq: { type: ['integer', 'null'] },
          steps: { type: 'integer' },
          confidence: { type: 'string' },
          open_issues: { type: 'integer' },
        },
        required: ['checkpoint_id', 'archived_from_dir', 'steps', 'confidence'],
      },
    },
  },
  required: ['checkpoints'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          checkpoint_id: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'needs_fix', 'cannot_verify'] },
          high: { type: 'integer' },
          medium: { type: 'integer' },
          note: { type: 'string' },
        },
        required: ['checkpoint_id', 'verdict', 'high', 'medium'],
      },
    },
  },
  required: ['results'],
}

function extractPrompt(cell) {
  return `You are the extractor agent for one scientist session. Cell: \`${cell}\`.

First read, in full:
1. ${REPO}/tools/wm_benchmark/extract_prompt.md  (your instructions — follow every rule, especially "Never delete anything")
2. ${REPO}/tools/wm_benchmark/launch_record.schema.json  (the output schema; every required field must be present, including archive_evidence.archive_submit_seq)
3. ${DATA}/benchmark/targets/${cell}.json  (the checkpoints to reconstruct, with hints; note archive_submit_seq and later_submits_after_archive)

Then work through the timeline at ${DATA}/timeline/${cell}/ exactly as the instructions say, and write one JSON per target checkpoint to ${DATA}/benchmark/x_raw/${cell}/<checkpoint_id>.json (mkdir -p the directory if needed; write files with the Write tool; never rm/mv anything). Save heredoc-extracted file contents under ${DATA}/benchmark/x_raw/${cell}/files/.

Be exhaustive about launch-time file versions and the chain back to the base model, including data-build steps and any weight averaging / merging / checkpoint selection / config edits that happened BEFORE the archiving submit. Cite seqs everywhere. Never record scores, accuracies or conclusions; redact card-yaml heredocs inside quoted commands.

Before returning, validate each record you wrote against the schema (python3 -c with jsonschema if available, else check the required keys by hand) and fix any violation.

When finished, return the structured summary (one entry per checkpoint).`
}

function verifyPrompt(cell, round) {
  return `You are the verifier agent for one scientist session. Cell: \`${cell}\`.${round === 2 ? ' This is the SECOND round: the extractor has revised the records after your earlier findings; check the revised records from scratch.' : ''}

First read, in full:
1. ${REPO}/tools/wm_benchmark/verify_prompt.md  (your instructions)
2. ${REPO}/tools/wm_benchmark/extract_prompt.md  (what the extractor was told; describes the timeline layout)
3. ${REPO}/tools/wm_benchmark/launch_record.schema.json
4. ${DATA}/benchmark/targets/${cell}.json

Then, for every record in ${DATA}/benchmark/x_raw/${cell}/*.json, do the checks against the timeline at ${DATA}/timeline/${cell}/ and write your verdict to ${DATA}/benchmark/x_verify/${cell}/<checkpoint_id>.json (mkdir -p the directory; write with the Write tool; never rm anything, never edit the extractor's records). Be adversarial: open every cited seq yourself. A record with a wrong archive mapping, wrong producing launch, a missing or wrong-version file, a chain that does not end at archived_from_dir, a score leak, or a missing schema-required field (e.g. archive_evidence.archive_submit_seq) is needs_fix. A target checkpoint listed in targets/${cell}.json that has NO record file at all is also a needs_fix result (checkpoint_id, verdict needs_fix, note "record missing"). Return the structured per-checkpoint results.`
}

function repairPrompt(cell, bad) {
  return `You are the extractor agent for cell \`${cell}\`, returning to REPAIR records that an independent verifier rejected.

Read ${REPO}/tools/wm_benchmark/extract_prompt.md and ${REPO}/tools/wm_benchmark/launch_record.schema.json, then for each checkpoint below read the verifier's findings in ${DATA}/benchmark/x_verify/${cell}/<checkpoint_id>.json, re-open the timeline at ${DATA}/timeline/${cell}/ at the cited seqs, and correct the record ${DATA}/benchmark/x_raw/${cell}/<checkpoint_id>.json in place (Write tool; never rm anything). If the record is missing entirely, extract it now from ${DATA}/benchmark/targets/${cell}.json. Where the verifier is wrong, keep the record and explain why in its open_issues. Add a top-level "revisions" array to each touched record listing what changed and which finding it answers.

Rejected checkpoints: ${JSON.stringify(bad)}

Return the structured summary (one entry per repaired checkpoint).`
}

log(`verify ${args.already_extracted.length} already-extracted cells; extract+verify ${args.remaining.length} remaining cells; model=${MODEL}`)

const results = await pipeline(
  cells,
  cell => {
    if (already.has(cell)) return Promise.resolve({ cell, extract: { cached: true } })
    return agent(extractPrompt(cell), { label: `extract:${cell}`, phase: 'Extract', agentType: 'general-purpose', model: MODEL, effort: EFFORT, schema: EXTRACT_SCHEMA })
      .then(x => (x ? { cell, extract: x } : null))
  },
  (r, cell) => {
    if (!r) return null
    return agent(verifyPrompt(cell, 1), { label: `verify:${cell}`, phase: 'Verify', agentType: 'general-purpose', model: MODEL, effort: EFFORT, schema: VERIFY_SCHEMA })
      .then(v => ({ ...r, verify: v }))
  },
  (r, cell) => {
    if (!r) return null
    const bad = ((r.verify && r.verify.results) || []).filter(x => x.verdict === 'needs_fix').map(x => ({ checkpoint_id: x.checkpoint_id, high: x.high, medium: x.medium, note: x.note }))
    if (!bad.length) return r
    log(`${cell}: ${bad.length} record(s) need repair`)
    return agent(repairPrompt(cell, bad), { label: `repair:${cell}`, phase: 'Repair', agentType: 'general-purpose', model: MODEL, effort: EFFORT, schema: EXTRACT_SCHEMA })
      .then(x => ({ ...r, repair: x, repaired: bad.map(b => b.checkpoint_id) }))
  },
  (r, cell) => {
    if (!r || !r.repair) return r
    return agent(verifyPrompt(cell, 2), { label: `reverify:${cell}`, phase: 'Re-verify', agentType: 'general-purpose', model: MODEL, effort: EFFORT, schema: VERIFY_SCHEMA })
      .then(v => ({ ...r, reverify: v }))
  },
)

const rows = results.filter(Boolean)
const tally = { cells: rows.length, extract_failed_cells: cells.length - rows.length, unverified_cells: 0, checkpoints_extracted: 0, confirmed: 0, needs_fix: 0, cannot_verify: 0, repaired_cells: 0 }
const failedCells = cells.filter((c, i) => !results[i])
for (const r of rows) {
  const final = (r.reverify || r.verify || {}).results
  if (!final) tally.unverified_cells += 1
  if (r.extract && r.extract.checkpoints) tally.checkpoints_extracted += r.extract.checkpoints.length
  for (const x of (final || [])) tally[x.verdict] = (tally[x.verdict] || 0) + 1
  if (r.repair) tally.repaired_cells += 1
}
log(`done: ${JSON.stringify(tally)}; failed cells: ${failedCells.join(',') || 'none'}`)
return {
  tally,
  failed_cells: failedCells,
  cells: rows.map(r => ({
    cell: r.cell,
    extracted: r.extract && r.extract.checkpoints ? r.extract.checkpoints.map(c => `${c.checkpoint_id}:${c.confidence}:${c.steps}`) : (r.extract && r.extract.cached ? 'cached' : []),
    verdicts: ((r.reverify || r.verify || {}).results || []).map(x => `${x.checkpoint_id}:${x.verdict}:${x.high}/${x.medium}`),
    repaired: r.repaired || [],
  })),
}