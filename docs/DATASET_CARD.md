# Dataset card

## Summary

ClaimScope v1.0 evaluates audits of already-stated mathematical claims. A
response must classify a claim, state its justified scope, identify a failure
mode, provide a machine-checkable counterexample when the claim is false, and
select a repair.

The benchmark is diagnostic rather than encyclopedic. It uses small exact
domains so that submitted witnesses can be checked without subjective grading.

## Data files

| File | Purpose |
| --- | --- |
| `artifacts/benchmark/extended_v1_public.jsonl` | 72 model-facing tasks |
| `artifacts/reference/extended_v1_gold.jsonl` | Reference labels, source specifications, and witnesses |
| `artifacts/benchmark/extended_v1_manifest.json` | Counts, seeds, and SHA-256 bindings |
| `artifacts/benchmark/pilot_public.jsonl` | Frozen 48-task construction base |
| `artifacts/reference/pilot_gold.jsonl` | Reference data for the construction base |
| `artifacts/benchmark/pilot_manifest.json` | Hash manifest for the construction base |

The names `pilot` and `extended_v1` are frozen artifact identifiers. The
complete 72-task `extended_v1` split is the ClaimScope v1.0 dataset.

## Task schema

Each model-facing JSONL record contains:

- `task_id`: stable task identifier;
- `family`: one of `optimizer_transport`, `affine_recurrence`, or
  `linear_algebra`;
- `split`: frozen artifact identifier;
- `prompt`: complete task and output instructions; and
- `response_schema`: field-level response requirements.

Reference records contain a matching `task_id`, the exact source
specification, and the reference audit under `gold`.

## Families

### Optimizer-state transport

Tasks audit one-step identities connecting reconstructed Adam-style directions
with Muon-style transitions. Mutations target learning-rate scaling, bias
correction, epsilon placement, second-moment normalization, decay terms, and
the distinction between direction equality and full-transition equality.

### Affine recurrence

Tasks audit finite-horizon unrollings of an affine recurrence. Mutations cover
input and initial-state exponents, singular geometric-series cases, domain
restrictions, and formula mismatches. Exact rational arithmetic avoids
floating-point ambiguity.

### Linear algebra

Tasks audit identities involving transpose, determinant, inverse order, and
inverse additivity for rational 2 x 2 matrices. Eight source identities appear
in three wording variants each. Inverse claims explicitly state their
invertibility conditions.

## Distribution

Each family contains 12 valid and 12 invalid tasks. The full release therefore
contains 36 valid and 36 invalid claims. Surface variants are correlated; the
included analysis reports both task-level counts and equal-weight summaries
over 48 distinct source specifications.

## Intended use

ClaimScope supports research on:

- mathematical claim verification;
- executable counterexample generation;
- scope and repair prediction;
- verifier-assisted revision; and
- evaluation methods that separate answer labels from supporting evidence.

The release can also serve as a small integration test for structured
mathematical-agent outputs.

## Evaluation isolation

For a blind model evaluation, provide only the records in
`artifacts/benchmark/extended_v1_public.jsonl`. Do not place reference answers,
source specifications, score rows, or validator internals in the model's
context or retrieval index. The reported runs follow this separation.

The reference files are published for reproducibility; they do not constitute
an unseen test set after release.

## Limitations

- The release contains 72 tasks and 48 distinct source specifications.
- The domains are deliberately narrow and exactly computable.
- Linear-algebra surface variants are not statistically independent.
- All prompts are in English and require a fixed JSON response schema.
- A verified counterexample establishes that a universal claim is false; it
  does not by itself establish that every diagnostic label or repair is ideal.
- Results from one model and deterministic decoding do not establish broad
  model or domain generality.

Future versions should add independently authored domains, more models,
stochastic replications, and a separately maintained hidden evaluation set.

## Relationship to prior work

ClaimScope has a narrower structured-audit objective than broad
counterexample, proof, or mathematical error-detection benchmarks. It does
not claim priority for counterexample reasoning or verifier-assisted
revision. See `docs/RELATED_WORK.md` for the closest benchmarks and the exact
scope of the contribution.

## Version and integrity

- Release name: ClaimScope v1.0
- Frozen split identifier: `extended_v1`
- Public-task SHA-256:
  `87bd5117f13a4335a950d858343ca02dc7d9390a765d2495772e7ef827ed13ec`
- Reference-answer SHA-256:
  `071fa625f6e12648d438e8a2afc65475c27f6c5224db848a24955bc0c681d121`
- Generation seed: `20260902`
