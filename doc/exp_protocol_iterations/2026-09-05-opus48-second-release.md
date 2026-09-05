# Opus4.8 GSM8K second release — direct pending-first instruction

On2026-09-05 the user corrected the operator's overly rigid interpretation:
scientifically valid pending work should be put onto idle GPUs rather than kept
held solely to preserve a numeric held floor. This supersedes the previous
decision to leave process-knowledge4 and guard4 idle after the first wave.

Release the two complete immutable receipts, not selected cells:

- process-knowledge k03g01–04, jobs92129–92132;
- old-guard bridge g03g01–04, jobs92137–92140.

Both manifests passed the full site/source/provider checks before registration.
The user instruction supplies a new per-receipt shared-reservation authorization;
ownership, live PENDING(JobHeldUser), and exact frozen ondem0–1 ReqNodeList checks
remain mandatory. Do not touch legacy mixed-receipt jobs90826–90830 or any WMA/
external queue. Run the operator dry reconciliation before apply and persist all
release evidence.

This direction changes utilization policy, not experimental interpretation:
the four sessions per arm remain fixed, failed/flagged first-wave outcomes are
not hidden, and no extra filler repeat is added. Replenish future pending work
asynchronously from scientifically useful admitted benchmark blocks. A lack of
ready future tasks should be reported as a task-admission dependency, not used
to stop already-valid work.
