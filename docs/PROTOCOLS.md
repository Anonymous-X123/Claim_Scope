# Evaluation protocols

All reported conditions use the same model revision, quantization, task order,
temperature, token limit, and model-facing benchmark. Every two-pass condition
starts with the same draft produced by the direct condition.

## Direct

The model receives the system instruction and task prompt and produces one
response.

## Self-reflection

The direct draft is returned to the model with an instruction to recheck
algebraic factors, scope, boundary cases, and witness validity. No external
checker is called.

## Counterexample-guided revision

The direct draft is returned with an instruction to search explicitly over
small exact values, verify assumptions, and recompute both sides. No external
checker is called.

## Generic feedback

The direct draft is returned with a matched negative-feedback message stating
that the witness may be invalid. The message explicitly says that no checker
has inspected the draft and supplies no task-specific signal.

## Verifier feedback

The direct draft is parsed and only its proposed counterexample is submitted
to an exact family-specific validator. The validator returns one of:

- `verified`: the witness satisfies the conditions and refutes the claim;
- `rejected`: the witness fails validation, together with a local validation
  reason; or
- `not_applicable`: no invalid-claim witness is available to check.

An unparseable draft receives `unavailable`. The validator does not return the
reference verdict, scope, failure label, repair, proof, or a corrected witness.
The model receives the status message and produces a second response.

The validator loads only the family and source specification needed to test a
witness. Its `WitnessVerifier` object discards the reference labels during
construction.

## Frozen generation settings

- Model: `Qwen/Qwen3-8B`
- Revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Quantization: NF4 4-bit
- Temperature: `0`
- Maximum new tokens per generation: `512`
- Thinking mode: off
- Hardware: NVIDIA RTX A4500

The run files preserve token counts, elapsed generation time, parser status,
drafts, final outputs, and verifier feedback where applicable.
