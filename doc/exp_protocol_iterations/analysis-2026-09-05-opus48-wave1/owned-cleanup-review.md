# Independent owned-container cleanup review

Scope: read-only review of `src/commit_utils/slurm/humaneval_environment_acceptance.py` and `src/run_task.sh` in `/tmp/exp-protocol-nextwave-ckg3t0gc/repo/third_party/PostTrainBench`. No GPU, cluster, model, or provider execution.

The original adopted-zombie defect was reproduced: a grandchild exited while its parent remained alive; the wrapper marked its record terminal, failed to reap it before adoption, then incorrectly reported success with the zombie still present. The revised terminal-record retry and final empty descendant graph check fix this defect. The same real CPU repro now reports `passed: true`, `cleanup_complete: true`, and `remaining_descendants: []`.

The revised observer now checks ancestry after opening the pidfd and rechecks process start ticks, addressing stale scanned PID ownership. The final direct-root exception now requires both `pid == process.pid` and `process.returncode is None`; this was independently confirmed by reading PTB commit `1674150`. The ancestry bypass is therefore restricted to the unreaped direct child. No requested corrections remain.

No additional blocking production-path regression was found. The shell wiring retains the scientist deadline, evaluator existing outer deadline, output flow, and evaluation attempt numbering. Exact owned pidfds authorize TERM/KILL; no user-wide or name-based process cleanup was added. FD handles close when terminal, and detached child cleanup is shared by admission and production.

Verification: the earlier targeted pytest module completed with exit 0; collection may have preceded the final appended zombie regression. The revised zombie case was then independently rerun directly with real fork/wait behavior and passed. Main agent reports the final targeted suite passed (17 tests, 9.63 seconds); no additional suite was launched by this reviewer. Real node re-admission of both images remains required; CPU review does not establish that acceptance.
