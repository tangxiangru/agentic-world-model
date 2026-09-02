# Online WMA sidecar boundary

An online PTB scientist reads `skills/exp_protocol` before any task action and
interacts with the WMA through only two commands:

```bash
awm wma review --dir . exp-01 [exp-02 ...] --background
awm wma status --dir .
```

The scientist-visible checkout contains `awm/wma_client.py` but not `awm/wma`,
`skills/wma`, `skills/wma_meta`, WMA history, or WMA transcripts. The client
atomically writes batch requests below `<session>/.wma/requests` and returns
immediately. It does not select the backend, model, effort, budget, history, or
skill version.

`third_party/PostTrainBench/src/run_task.sh` starts a separate CPU-only
Apptainer sidecar before the scientist. Its mounts are:

- scientist session at `/session`, read-only;
- `/session/memory/cards` and `/session/.wma`, writable;
- private WMA checkout at `/opt/awm`, read-only;
- historical corpus at `/history`, read-only;
- private scratch and transcript directories, not mounted into the scientist.

The sidecar accepts only cards with `exp-NN.lock.json`, runs the frozen WMA
backend/model/effort/budget from the manifest, and writes only verdict and
request-status files into the scientist session. It launches Claude with empty
setting sources so the scientist's `CLAUDE.md`, project skills, hooks, and
personas cannot become WMA instructions. Complete WMA transcripts live under
the result directory's `wma_private/` for later iteration-agent analysis.

The launcher records two distinct checkout digests in the receipt:

- `awm_checkouts`: public scientist-visible protocol/client checkout;
- `wma_private_checkouts`: private WMA runtime/skill checkout.

`audit_receipt` rejects a WMA-aware result when the frozen runtime identity
differs, the sidecar did not finish, no verdict exists, or the harvested
scientist task exposes a WMA skill or transcript.
