from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from claimscope.core import State, audit_candidate, reconstructed_direction
from claimscope.tasks import TASKS


class TransportIdentityTests(unittest.TestCase):
    def test_full_transport_matches_complete_adamw_transition(self) -> None:
        audit = audit_candidate(
            "full_transport", "adamw_decay", "full_transition"
        )
        self.assertTrue(audit.valid_on_grid)

    def test_direction_transport_matches_with_muon_transition_decay(self) -> None:
        audit = audit_candidate(
            "direction_transport", "muon_decay", "full_transition"
        )
        self.assertTrue(audit.valid_on_grid)

    def test_direction_transport_does_not_generally_match_adamw_decay(self) -> None:
        audit = audit_candidate(
            "direction_transport", "adamw_decay", "full_transition"
        )
        self.assertFalse(audit.valid_on_grid)
        self.assertIsNotNone(audit.counterexample)

    def test_each_invalid_development_task_has_a_witness(self) -> None:
        for task in TASKS:
            with self.subTest(task=task.task_id):
                audit = audit_candidate(
                    task.formula_id,
                    task.protocol,
                    task.quantity,
                    condition=task.condition,
                )
                self.assertEqual(
                    audit.valid_on_grid,
                    task.expected_verdict == "valid",
                )

    def test_zero_second_moment_is_finite_when_epsilon_is_positive(self) -> None:
        state = State(
            w=1.0,
            d=0.5,
            v=0.0,
            alpha=0.02,
            eta=0.01,
            lambda_m=0.0,
            lambda_a=0.0,
            beta1=0.9,
            beta2=0.99,
            step=5,
            epsilon=1e-8,
        )
        self.assertAlmostEqual(
            reconstructed_direction(state, "direction_transport"),
            state.target_direction,
        )

    def test_invalid_state_is_rejected(self) -> None:
        state = State(1, 1, -1, 0.1, 0.1, 0, 0, 0.9, 0.99, 1, 1e-8)
        with self.assertRaises(ValueError):
            reconstructed_direction(state, "direction_transport")


class CliTests(unittest.TestCase):
    def test_export_contains_no_answer_key(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root / "src")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prompts.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "claimscope.cli",
                    "export",
                    "--output",
                    str(output),
                ],
                cwd=project_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), len(TASKS))
            self.assertTrue(all(set(row) == {"task_id", "prompt"} for row in rows))

    def test_safe_completion_is_not_tied_to_one_failure_label(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root / "src")
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            rows = []
            for task in TASKS:
                mode = task.failure_mode
                if task.task_id == "CS010":
                    mode = "epsilon_placement"  # deliberately non-gold
                rows.append(
                    {
                        "task_id": task.task_id,
                        "verdict": task.expected_verdict,
                        "scope": task.expected_scope,
                        "failure_mode": mode,
                        "repair": task.repair,
                    }
                )
            predictions.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "claimscope.cli",
                    "score",
                    str(predictions),
                ],
                cwd=project_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["safe_completion_rate"], 1.0)
            self.assertLess(summary["failure_mode_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
