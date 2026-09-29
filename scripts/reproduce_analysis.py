"""Reproduce the matched-control analysis shipped with ClaimScope."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimscope.analyze_controls import analyze_files, render_markdown  # noqa: E402


def main() -> int:
    os.chdir(ROOT)
    prompt_run = Path("results/runs/prompt_controls.jsonl")
    prompt_score = Path("results/scores/prompt_controls.json")
    conditions = [
        (
            "direct",
            Path("results/runs/direct.jsonl"),
            Path("results/scores/direct.json"),
        ),
        ("self_reflect", prompt_run, prompt_score),
        ("counterexample_guided", prompt_run, prompt_score),
        (
            "generic_feedback",
            Path("results/runs/generic_feedback.jsonl"),
            Path("results/scores/generic_feedback.json"),
        ),
        (
            "verifier_feedback",
            Path("results/runs/verifier_feedback.jsonl"),
            Path("results/scores/verifier_feedback.json"),
        ),
    ]

    analysis = analyze_files(
        gold_path=Path("artifacts/reference/extended_v1_gold.jsonl"),
        conditions=conditions,
        baseline_name="direct",
        require_complete=True,
        bootstrap_samples=10_000,
        seed=20260904,
        comparison_pairs=[
            ("counterexample_guided", "verifier_feedback"),
            ("generic_feedback", "verifier_feedback"),
        ],
    )

    output_dir = Path("artifacts/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "matched_controls.json"
    output_markdown = output_dir / "matched_controls.md"
    output_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output_markdown.write_text(
        render_markdown(analysis),
        encoding="utf-8",
        newline="\n",
    )

    print(render_markdown(analysis))
    print(f"\nJSON: {output_json}")
    print(f"Markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
