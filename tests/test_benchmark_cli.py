from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from claimscope.benchmark_cli import ALL_TASK_IDS
from claimscope.recurrence import TASKS as RECURRENCE_TASKS
from claimscope.tasks import TASKS as OPTIMIZER_TASKS


class UnifiedBenchmarkCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.environment = os.environ.copy()
        self.environment["PYTHONPATH"] = str(
            self.project_root / "src"
        )

    def test_combined_task_count(self) -> None:
        self.assertEqual(len(ALL_TASK_IDS), 22)
        self.assertEqual(len(set(ALL_TASK_IDS)), 22)

    def test_export_contains_all_tasks_without_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prompts.jsonl"

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "claimscope.benchmark_cli",
                    "export",
                    "--output",
                    str(output),
                ],
                cwd=self.project_root,
                env=self.environment,
                check=True,
                capture_output=True,
                text=True,
            )

            rows = [
                json.loads(line)
                for line in output.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertEqual(len(rows), 22)
            self.assertTrue(
                all(
                    set(row) == {
                        "task_id",
                        "family",
                        "prompt",
                    }
                    for row in rows
                )
            )

    def test_gold_predictions_reach_safe_completion_one(self) -> None:
        rows = []

        for task in OPTIMIZER_TASKS:
            rows.append(
                {
                    "task_id": task.task_id,
                    "verdict": task.expected_verdict,
                    "scope": task.expected_scope,
                    "failure_mode": task.failure_mode,
                    "repair": task.repair,
                }
            )

        for task in RECURRENCE_TASKS:
            rows.append(
                {
                    "task_id": task.task_id,
                    "verdict": task.expected_verdict,
                    "scope": "finite_trajectory",
                    "failure_mode": task.failure_mode,
                    "repair": task.repair,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text(
                "\n".join(json.dumps(row) for row in rows)
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "claimscope.benchmark_cli",
                    "score",
                    str(predictions),
                ],
                cwd=self.project_root,
                env=self.environment,
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(completed.stdout)

            self.assertEqual(summary["tasks"], 22)
            self.assertEqual(
                summary["safe_completion_rate"],
                1.0,
            )
            self.assertEqual(
                summary["by_family"][
                    "optimizer_transport"
                ]["safe_completion_rate"],
                1.0,
            )
            self.assertEqual(
                summary["by_family"][
                    "affine_recurrence"
                ]["safe_completion_rate"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
