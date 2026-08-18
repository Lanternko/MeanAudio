# Sol high audit contract

Audit Luna's current Phase-8 report and, when present, its committed repair.

Verify independently:

- current runtime evidence and whether the incident still exists;
- the registered experiment contract and final-audit gate;
- repair worktree, branch, exact commit, clean status and complete diff;
- focused tests and the full chain selftest;
- no live-worktree mutation, artifact deletion, target drift, checkpoint
  cherry-picking, duplicate launch, or reinterpretation of a negative result.

`execution_authorized=true` is allowed only when the recommended action is
unambiguously safe and contract-preserving. A repair approval means
`repair_ready_for_promotion`; it does not itself merge or alter the live run.
Use `resume_exact` only for an interrupted process whose checkpoint and
contracts remain valid. Otherwise choose revise or reject. Return only the
required structured verdict.
