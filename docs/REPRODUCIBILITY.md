# Reproducibility guide

## 1. Verify the artifact

From the repository root:

```bash
python -m pip install -e .
python scripts/verify_artifact.py
python -m unittest discover -s tests -v
```

The verifier checks the published public and reference hashes before using the
answers. It also reconstructs oracle predictions and requires perfect safe and
strict completion.

## 2. Regenerate the benchmark

The 72-task release extends the frozen 48-task construction base with 24
linear-algebra tasks:

```bash
python -m claimscope.generate_extended --force
python scripts/verify_artifact.py
```

Generation is deterministic. The command must reproduce the SHA-256 values in
`artifacts/benchmark/extended_v1_manifest.json`.

## 3. Recompute frozen scores

```bash
claimscope-score results/runs/direct.jsonl \
  --output reproduced/direct.json \
  --require-complete

claimscope-score results/runs/verifier_feedback.jsonl \
  --output reproduced/verifier_feedback.json \
  --require-complete
```

The same command applies to `generic_feedback.jsonl`. The combined
`prompt_controls.jsonl` file contains complete `self_reflect` and
`counterexample_guided` groups.

## 4. Recompute paired analysis

```bash
python scripts/reproduce_analysis.py
```

This produces task-level, family-level, template-weighted, cost, and paired
transition summaries. Cluster bootstrap resampling operates over the 48 source
specifications with 10,000 samples and seed `20260904`.

## 5. Repeat model inference

Install the pinned GPU environment:

```bash
python -m pip install -r requirements-gpu.txt
```

The model revisions used in the project appear in
`artifacts/environment/model_revisions.json`. Hardware, CUDA, and library
versions are saved in each run record. Exact wall-clock time may vary by
hardware even when decoded outputs match.

Hugging Face model weights are not redistributed in this repository.
