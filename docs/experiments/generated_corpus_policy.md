# Generated-corpus policy

This is the canonical quality and provenance contract for every generated
caption corpus used by MeanAudio. Generation, repair, regeneration, launch,
feature extraction, and training code must implement this policy. A prompt,
length-distribution summary, or small pilot is not evidence that a corpus is
clean.

## Generation and stop behavior

The generator must pass explicit end-of-sequence (EOS) and padding IDs to the
model/tokenizer APIs. Before decoding, truncate each generated token-ID
sequence at its first stop token, including EOS. A bounded runtime test must
assert both that stopping occurs and that no token after the first stop is
decoded. The test must fail if the stop token is missing, duplicated in the
decoded output, or followed by decoded content.

## One structural gate

Maintain one reusable row-level structural classifier. Repair, regeneration,
the launch gate, and the pre-training gate must call that same classifier and
the same defect taxonomy; they must not maintain divergent ad hoc checks. For
the corpus contract, the classifier must detect at least:

- null rows or fields;
- duplicate, missing, or extra IDs;
- ID and row ordering errors;
- invalid role turns or role sequencing;
- CJK text when the contract is English-only;
- JSON or Markdown wrappers and other formatting wrappers;
- lead-ins or other unrequested prefatory text;
- URLs, code, or LaTeX;
- repeated or unexpected symbols;
- multiline output;
- metadata drift from the registered schema or source values; and
- missing terminal punctuation.

Benign short captions remain in the corpus. They may be filtered only when a
length-selection rule was preregistered for every comparison arm; a healthy
length distribution is not a substitute for this rule or for row-level
validation.

Every classifier result must identify the row/ID, defect category, observed
value or bounded evidence, and expected contract. Summary counts and aggregate
statistics never replace row-level checks.

## Fail-closed gates and contract integration

The experiment contract must register the corpus artifact, schema, classifier
version, defect taxonomy, stop-behavior test, and machine-readable gate report
as immutable inputs/evidence for each downstream phase. The gate is independent
of generation completion markers and must run on the complete corpus:

New experiment contracts must select the `generated` corpus branch in
`harn-schema-v1`; the `non_generated` branch cannot carry or substitute for a
generated-corpus gate. The offline validator checks document bindings, while the
runtime must verify the referenced artifact bytes and freshness.

1. before every launch, including a resume;
2. again immediately before training; and
3. after any repair, regeneration, upstream change, or expanded defect
   taxonomy.

A `.done` marker is only a progress hint and is never gate evidence by itself.
The launch or training phase must stop when the full-corpus gate is absent,
stale, invalid, or failing. It may proceed only on a passing report whose
corpus hash, classifier/taxonomy identity, and contract/run identity match the
current inputs. A newly observed defect category creates a new incident
fingerprint and invalidates prior corpus approval until the expanded full
corpus gate passes.

Validators must write the gate report and defect list atomically in
machine-readable form and return non-zero for every failed invariant. The
repository must include adversarial fixtures covering every defect family in
the classifier, including stop behavior and metadata drift. A gate that cannot
produce its report, defect list, or fixture-backed validation is invalid and
fails closed.

## Corpus, TSV, and feature-cache provenance

Bind the artifact chain `corpus -> TSV -> feature cache` with SHA-256 hashes.
The TSV manifest must record the exact corpus hash and row/ID mapping. The
feature-cache completion report must record the exact TSV hash and cache
identity for every derived artifact.

Cache identity must include all of the following:

- caption hash;
- pinned encoder model revision;
- encoder checkpoint hash;
- tokenization and windowing semantics; and
- stored feature shape.

A caption hash alone is insufficient for resume. Whenever an upstream hash,
schema, encoder fingerprint, tokenization/windowing rule, or stored shape
changes, rebuild or independently verify every affected derived artifact and
write a new completion report. Training may start only when the current
corpus gate, TSV manifest, and feature-cache completion report agree on hashes,
row/ID coverage, encoder identity, and contract/run identity. Any missing,
stale, mismatched, or unverifiable link fails closed.

## Immutable audio store and switchable caption overlays

For every new experiment, audio latents and caption-derived features are
separate artifacts. The canonical audio NPZ store is immutable and contains
the audio latent distribution plus a stable `clip_id`; it must not be rewritten
to switch caption variants. Each caption corpus has its own text-feature
overlay and manifest, selected by configuration at launch time.

The loader must bind `TSV id -> cache-list filename -> audio clip_id -> text
overlay clip_id` and fail closed on any missing file, duplicate ID, count
mismatch, or ID mismatch. Switching captions means changing the TSV and text
overlay paths together; it never means mutating or restoring the audio store.
New experiment contracts must set the equivalent of
`require_text_overlay=true`. The embedded-text path remains available only for
explicitly labeled historical reproduction and may not be used as the source
of a new primary comparison.

A multi-caption overlay stores all registered text features for one audio ID
inside that ID's overlay record. Caption choice is a loader/sampler decision;
it must not create duplicate audio-latent records or rely on positional row
alignment.
