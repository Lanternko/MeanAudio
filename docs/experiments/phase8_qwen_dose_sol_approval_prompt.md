# Sol high approval review — Phase-8 Qwen caption-dose chain

Review the proposed 50k/100k paired caption-dose experiment under the live
repository `/home/kojiek/MeanAudio`.

This is the fourth review after three fail-closed rejections. Confirm that the
paired bootstrap now validates its real schema (including n=5521), every
reviewed implementation file is hash-bound by authorization and execution
manifest, resume requires a distinct `resume_identical_contract` Sol verdict
plus Codex authorization, and the scheduler waits for the passed parent final
step and complete parent process exit. Also confirm the expanded regression
tests cover those gates.

Also confirm the bootstrap fields are typed and finite and bind seed,
replicates, paths, paired-ID digest, and mean/delta consistency. Confirm every
fresh or resumed preflight re-hashes both arm TSVs, cache lists, canonical cache
manifests, TSV ID order, the control validation manifest, the Qwen mapper and
boundary artifact, and the MusicCaps TSV; arm audits re-check the same declared
input hashes. Confirm negative tests reject malformed bootstrap values,
implementation drift, data provenance drift, and invalid resume semantics.

The nested control validation/canonical-manifest and Qwen mapper/boundary
checks now live in one shared hash-bound module used by both preflight and arm
audit. Confirm its negative tests individually reject every nested artifact
drift plus bootstrap seed, replicate, path, digest, CI-order, and mean/delta
inconsistency.

Read the dose contract, queue, audit, scheduler, monitor, Luna loop/prompt,
paired-report change, self-test, exact plan output, current immutable 20k
contract/runtime hashes, current live experiment state, disk capacity, and git
diff for these files. Verify that:

- the predecessor 20k paired report and both arm audits are hard gates;
- 50k means it620000 -> it650000 and 100k means it650000 -> it700000;
- Control and Qwen arms differ only in caption TSV/NPZ/cache source;
- optimizer/scheduler are resumed, not reset, at dose milestones;
- NoQ, NoMask, seed, batch, LR, model, eval, and row order remain matched;
- each arm has fixed-checkpoint metrics/audit and paired report;
- 100k cannot start before the 50k paired report passes;
- metric values never trigger retry, stopping, or hyperparameter changes;
- duplicate, disk, source-iteration, provenance, partial-eval, and resume gates
  fail closed;
- scheduler cannot launch before the current parent queue completes successfully;
- monitor/Luna have proposal-only authority and Sol/Codex control interventions.

Return only the supplied schema JSON. Do not edit, launch, stop, resume, or
otherwise mutate any experiment. `approve_predeclared_dose_chain` authorizes
Codex only to bind this verdict into an authorization artifact and arm the
waiting scheduler; actual launch must still pass the live predecessor and
preflight gates.
