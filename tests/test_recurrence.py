from __future__ import annotations

import unittest

from claimscope.recurrence import (
    TASKS,
    audit_recurrence,
    audit_task,
    diagnostic_states,
    exact_unroll,
)


class RecurrenceOracleTests(unittest.TestCase):
    def test_exact_unroll_matches_iterative_recurrence(self) -> None:
        for state in diagnostic_states():
            with self.subTest(state=state):
                self.assertEqual(exact_unroll(state), state.iterative_value())

    def test_all_development_tasks_match_expected_labels(self) -> None:
        for task in TASKS:
            with self.subTest(task=task.task_id):
                audit = audit_task(task.task_id)
                self.assertEqual(
                    audit.valid_on_grid,
                    task.expected_verdict == "valid",
                )

    def test_input_off_by_one_has_a_witness(self) -> None:
        audit = audit_recurrence("input_off_by_one")
        self.assertFalse(audit.valid_on_grid)
        self.assertIsNotNone(audit.counterexample)
        self.assertEqual(audit.counterexample.reason, "value_mismatch")

    def test_piecewise_formula_handles_rho_one(self) -> None:
        audit = audit_recurrence(
            "constant_piecewise",
            condition="constant_input",
        )
        self.assertTrue(audit.valid_on_grid)

    def test_geometric_formula_requires_singular_case_handling(self) -> None:
        audit = audit_recurrence(
            "constant_geometric",
            condition="constant_input",
        )
        self.assertFalse(audit.valid_on_grid)
        self.assertIsNotNone(audit.counterexample)
        self.assertEqual(
            audit.counterexample.reason,
            "candidate_undefined",
        )


if __name__ == "__main__":
    unittest.main()
