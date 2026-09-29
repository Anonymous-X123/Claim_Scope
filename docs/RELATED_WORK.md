# Relationship to prior benchmarks

ClaimScope does not claim to be the first benchmark involving mathematical
counterexamples, error detection, or verifier feedback. Its contribution is
the particular audit contract and controlled evaluation packaged here.

## Closely related work

- **COUNTERMATH** evaluates proof or disproof of university-level statements
  through counterexamples. ClaimScope instead begins with a proposed local
  identity and requires a structured audit containing a verdict, justified
  scope, diagnostic label, executable witness, and repair.
  https://countermath.github.io/
- **MathDebugger** evaluates detection and classification of errors in
  synthetic mathematical questions and solutions. ClaimScope focuses on
  small exact claims whose submitted counterexamples can be executed by
  family-specific validators.
  https://arxiv.org/abs/2502.19058
- **CRAFT** is a counterexample-guided workflow for falsifying and repairing
  claims in machine learning and optimization. It is the closest prior work:
  its verifier searches for witnesses and can expose a found witness to a
  model. ClaimScope instead checks the model's own submitted witness and, in
  the verifier-feedback condition, returns only witness status and a local
  validation reason. ClaimScope also uses matched two-pass controls that begin
  from the same direct draft and reports composite audit completion.
  https://github.com/yiruiliu/Craft_CounterExample

## Scope of the contribution

The release supports the following specific contribution claim:

1. a six-field audit task combining claim classification, scope, diagnosis,
   an executable counterexample, and repair;
2. exact family-specific validation across optimizer-state transport,
   finite-horizon affine recurrences, and rational 2 x 2 linear algebra;
3. a label-blind protocol that validates a proposed witness without returning
   a gold verdict, repair, proof, or replacement witness; and
4. a paired comparison with self-reflection, counterexample-guided prompting,
   and a matched checker-free negative-feedback control whose first-pass
   drafts are identical to the direct condition.

These elements distinguish ClaimScope's evaluation design. They do not imply
that counterexample reasoning, mathematical error diagnosis, or
verifier-assisted revision originated with ClaimScope.
