# Evaluation metrics

ClaimScope reports each metric as a binary value per task and as a mean or
count over tasks.

## Structural metrics

- **JSON valid:** the response contains all required fields with admissible
  structural values.
- **Task ID correct:** the reported task identifier exactly matches the task
  routed to the scorer.

## Audit-component metrics

- **Verdict correct:** `valid` or `invalid` matches the reference verdict.
- **Scope correct:** the selected scope matches the reference scope.
- **Failure mode correct:** the selected diagnostic label matches the
  reference label. This is diagnostic and is not part of safe completion.
- **Repair correct:** the selected repair matches the reference repair.
- **Counterexample valid:** valid claims use `counterexample=null`; invalid
  claims provide a state that satisfies the task conditions and makes the
  candidate differ from the exact target.

## Composite metrics

Safe completion requires:

1. valid JSON;
2. the correct task identifier;
3. the correct verdict;
4. the correct scope;
5. a valid counterexample policy or witness; and
6. the correct repair.

Strict completion additionally requires the exact reference failure-mode
label. Keeping the two composites separate prevents a plausible alternative
diagnostic label from invalidating an otherwise correct and executable audit.

## Aggregation

Task-level summaries give every one of the 72 prompts equal weight.
Template-weighted summaries first average over wording variants of each source
specification and then give each of the 48 specifications equal weight.
Paired comparisons use identical task identifiers across conditions.
