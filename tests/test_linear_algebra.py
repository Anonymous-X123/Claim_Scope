import unittest
from fractions import Fraction

from claimscope.linear_algebra import (
    FORMULA_SPECS,
    Matrix2,
    MatrixState,
    claim_holds,
    condition_holds,
    diagnostic_states,
    expected_and_actual,
    find_counterexample,
)


class LinearAlgebraOracleTests(unittest.TestCase):
    def test_valid_specs_hold_on_diagnostic_states(self) -> None:
        for spec in FORMULA_SPECS:
            if spec.verdict != "valid":
                continue

            for state in diagnostic_states():
                if condition_holds(state, spec.condition):
                    self.assertTrue(
                        claim_holds(
                            state,
                            spec.formula_id,
                            spec.condition,
                        )
                    )

    def test_each_invalid_spec_has_an_exact_witness(self) -> None:
        for spec in FORMULA_SPECS:
            if spec.verdict != "invalid":
                continue

            witness = find_counterexample(
                spec.formula_id,
                spec.condition,
            )

            self.assertIsNotNone(witness, spec.formula_id)

    def test_inverse_conditions_reject_singular_states(self) -> None:
        singular = Matrix2(
            Fraction(1),
            Fraction(0),
            Fraction(0),
            Fraction(0),
        )
        identity = Matrix2(
            Fraction(1),
            Fraction(0),
            Fraction(0),
            Fraction(1),
        )
        state = MatrixState(singular, identity)

        self.assertFalse(
            condition_holds(state, "invertible_pair")
        )

        with self.assertRaises(ValueError):
            expected_and_actual(
                state,
                "inverse_reverse",
                "invertible_pair",
            )


if __name__ == "__main__":
    unittest.main()