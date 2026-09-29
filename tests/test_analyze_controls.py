import json
import tempfile
import unittest
from pathlib import Path

from claimscope.analyze_controls import analyze_files, render_markdown


class ControlAnalysisTests(unittest.TestCase):
    def _write_fixture(self, root: Path):
        gold = root / "gold.jsonl"
        direct_run = root / "direct.jsonl"
        direct_score = root / "direct-score.json"
        generic_run = root / "generic.jsonl"
        generic_score = root / "generic-score.json"

        gold_rows = [
            {
                "task_id": "T1",
                "family": "linear_algebra",
                "source": {
                    "formula_id": "identity_a",
                    "surface_variant": 0,
                },
                "gold": {"verdict": "invalid"},
            },
            {
                "task_id": "T2",
                "family": "linear_algebra",
                "source": {
                    "formula_id": "identity_a",
                    "surface_variant": 1,
                },
                "gold": {"verdict": "invalid"},
            },
            {
                "task_id": "T3",
                "family": "affine_recurrence",
                "source": {"formula_id": "identity_b"},
                "gold": {"verdict": "invalid"},
            },
        ]

        def run_rows(protocol: str):
            return [
                {
                    "task_id": task_id,
                    "family": family,
                    "model_id": "fake/model",
                    "protocol": protocol,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "seconds": 1.0,
                    "draft_output": None if protocol == "direct" else "draft",
                }
                for task_id, family in (
                    ("T1", "linear_algebra"),
                    ("T2", "linear_algebra"),
                    ("T3", "affine_recurrence"),
                )
            ]

        def score_rows(protocol: str, safe: tuple[int, int, int]):
            return [
                {
                    "task_id": task_id,
                    "family": family,
                    "model_id": "fake/model",
                    "protocol": protocol,
                    "json_valid": 1,
                    "verdict_correct": 1,
                    "scope_correct": 1,
                    "repair_correct": value,
                    "counterexample_valid": value,
                    "safe_completion": value,
                    "strict_completion": value,
                }
                for (task_id, family), value in zip(
                    (
                        ("T1", "linear_algebra"),
                        ("T2", "linear_algebra"),
                        ("T3", "affine_recurrence"),
                    ),
                    safe,
                )
            ]

        def write_jsonl(path: Path, rows):
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

        write_jsonl(gold, gold_rows)
        write_jsonl(direct_run, run_rows("direct"))
        write_jsonl(generic_run, run_rows("generic_feedback"))
        direct_score.write_text(
            json.dumps({"rows": score_rows("direct", (0, 0, 0))}),
            encoding="utf-8",
        )
        generic_score.write_text(
            json.dumps(
                {"rows": score_rows("generic_feedback", (1, 0, 1))}
            ),
            encoding="utf-8",
        )
        return gold, direct_run, direct_score, generic_run, generic_score

    def test_paired_template_and_cost_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_fixture(Path(directory))
            analysis = analyze_files(
                gold_path=paths[0],
                conditions=[
                    ("direct", paths[1], paths[2]),
                    ("generic_feedback", paths[3], paths[4]),
                ],
                baseline_name="direct",
                require_complete=True,
                bootstrap_samples=100,
            )

        self.assertEqual(analysis["gold_task_count"], 3)
        self.assertEqual(analysis["unique_template_count"], 2)
        generic = analysis["conditions"]["generic_feedback"]
        self.assertEqual(generic["cost"]["generation_count"], 6)
        self.assertEqual(generic["metrics"]["safe_completion"]["count"], 2)
        self.assertAlmostEqual(
            generic["template_macro"]["safe_completion"],
            0.75,
        )

        paired = analysis["paired_comparisons"][
            "direct_to_generic_feedback"
        ]
        self.assertEqual(
            paired["metrics"]["safe_completion"]["delta"], 2
        )
        self.assertEqual(
            paired["template_comparisons"]["safe_completion"][
                "templates_improved"
            ],
            2,
        )

        markdown = render_markdown(analysis)
        self.assertIn("Template-weighted results", markdown)
        self.assertIn("direct_to_generic_feedback", markdown)

    def test_complete_analysis_rejects_missing_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_fixture(Path(directory))
            score = json.loads(paths[4].read_text(encoding="utf-8"))
            score["rows"].pop()
            paths[4].write_text(json.dumps(score), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "task sets do not match"):
                analyze_files(
                    gold_path=paths[0],
                    conditions=[
                        ("direct", paths[1], paths[2]),
                        ("generic_feedback", paths[3], paths[4]),
                    ],
                    baseline_name="direct",
                    require_complete=True,
                    bootstrap_samples=10,
                )


if __name__ == "__main__":
    unittest.main()
