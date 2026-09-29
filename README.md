# ClaimScope (Accepted at NeurIPS 2026 workshop on Mathematical Reasoning and AI)

ClaimScope is a benchmark for auditing mathematical claims. Each task asks a
model to determine whether a proposed identity is valid at its stated scope,
identify the failure mode when it is invalid, provide an executable
counterexample, and select a repair.

This repository contains the complete ClaimScope v1.0 review artifact:

- 72 benchmark tasks across three mathematical families;
- reference answers and exact computational validators;
- the scoring and local-inference code used in the evaluation;
- frozen outputs for all five reported conditions; and
- scripts for integrity checks and matched-control analysis.

During the reported evaluation, models received only the task prompts. The
reference answers and validator implementation were not included in model
context. They are included here so that the results can be inspected and
reproduced.

## Benchmark at a glance

| Family | Tasks | Valid | Invalid |
| --- | ---: | ---: | ---: |
| Optimizer-state transport | 24 | 12 | 12 |
| Finite-horizon affine recurrence | 24 | 12 | 12 |
| Exact 2 x 2 linear algebra | 24 | 12 | 12 |
| **Total** | **72** | **36** | **36** |

The 24 linear-algebra tasks instantiate eight identities with three surface
forms each. Across all families, the benchmark contains 48 distinct source
specifications. This dependence is handled by the template-weighted analysis.

The frozen split identifier inside the artifact is `extended_v1`; it denotes
the 72-task ClaimScope v1.0 release and is retained to preserve the reported
file hashes.

ClaimScope is related to prior benchmarks for counterexample reasoning and
mathematical error diagnosis, but it evaluates a different output contract:
verdict, scope, failure mode, executable witness, and repair under exact
family-specific validation. See
[docs/RELATED_WORK.md](docs/RELATED_WORK.md) for a concise comparison and the
limits of the novelty claim.

## Repository layout

```text
artifacts/
  benchmark/     task files and hash manifests
  reference/     reference answers and validator inputs
  environment/   pinned model revisions
  results/       reproduced aggregate analysis
docs/            benchmark, metric, and protocol documentation
results/
  runs/          frozen model generations
  scores/        frozen per-task scores
scripts/         integrity and analysis entry points
src/claimscope/  generators, validators, runner, scorer, and analysis code
tests/           unit and regression tests
```

## Quick start

ClaimScope's benchmark generation, validation, scoring, and analysis use only
the Python standard library. Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/verify_artifact.py
python -m unittest discover -s tests -v
```

The integrity check verifies the published hashes, task balance, task-ID
alignment, oracle answers, frozen run coverage, and stored per-task scores.

## Score predictions

Each prediction must identify the task and provide these six fields:

```json
{
  "task_id": "PILOT-REC-001",
  "verdict": "valid",
  "scope": "finite_horizon_value",
  "failure_mode": "none",
  "counterexample": null,
  "repair": "none"
}
```

Predictions may be bare objects or records containing a nested `prediction`
object plus `model_id` and `protocol`. Score a complete run with:

```bash
claimscope-score results/runs/direct.jsonl \
  --output reproduced/direct-score.json \
  --require-complete
```

The default reference and manifest paths point to the complete 72-task
release. Explicit `--gold` and `--manifest` options remain available.

## Reproduce the reported analysis

```bash
python scripts/reproduce_analysis.py
```

This command checks the stored scores, then regenerates:

- `artifacts/results/matched_controls.json`
- `artifacts/results/matched_controls.md`

The main paired comparison uses checker-free generic negative feedback as the
matched control for label-blind witness verification.

## Run a local model

GPU inference is optional and has additional dependencies:

```bash
python -m pip install -r requirements-gpu.txt
claimscope-run \
  --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --output results/new-direct-run.jsonl \
  --protocol direct \
  --quantization 4bit \
  --thinking off \
  --max-new-tokens 512 \
  --temperature 0
```

For `verifier_feedback`, also pass:

```bash
--verifier-key artifacts/reference/extended_v1_gold.jsonl
```

The runner rejects any attempt to load a file under `artifacts/reference` as
model input.

## Reported conditions

| Condition | Generations per task | External witness check |
| --- | ---: | --- |
| `direct` | 1 | No |
| `self_reflect` | 2 | No |
| `counterexample_guided` | 2 | No |
| `generic_feedback` | 2 | No |
| `verifier_feedback` | 2 | Yes, witness validity only |

All two-pass conditions start from drafts that match the corresponding direct
outputs. See [docs/PROTOCOLS.md](docs/PROTOCOLS.md) for the exact intervention
contract and [docs/DATASET_CARD.md](docs/DATASET_CARD.md) for task construction,
intended use, and limitations.

The frozen task-level results are:

| Condition | Verdict | Valid witness | Safe | Strict |
| --- | ---: | ---: | ---: | ---: |
| `direct` | 43/72 | 21/72 | 18/72 | 15/72 |
| `self_reflect` | 41/72 | 18/72 | 15/72 | 13/72 |
| `counterexample_guided` | 43/72 | 20/72 | 17/72 | 14/72 |
| `generic_feedback` | 44/72 | 23/72 | 20/72 | 17/72 |
| `verifier_feedback` | 43/72 | 30/72 | 25/72 | 17/72 |

These counts describe one model revision and one deterministic decoding
setting. They are not estimates of performance across models or prompts.

## License and citation

Source code is available under the MIT License. Benchmark and result data are
available under CC BY 4.0; see [DATA_LICENSE.md](DATA_LICENSE.md). Model and
software attributions appear in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

