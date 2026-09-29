"""Exact 2x2 rational-matrix identity oracle for ClaimScope."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable


FORMULA_IDS = (
    "transpose_reverse",
    "transpose_same_order",
    "trace_cyclic",
    "determinant_product",
    "determinant_additive",
    "inverse_reverse",
    "inverse_same_order",
    "inverse_additive",
)

CONDITIONS = (
    "all",
    "invertible_pair",
    "invertible_pair_and_sum",
)

FORMULA_TEXT = {
    "transpose_reverse": "(A*B)^T = B^T*A^T",
    "transpose_same_order": "(A*B)^T = A^T*B^T",
    "trace_cyclic": "tr(A*B) = tr(B*A)",
    "determinant_product": "det(A*B) = det(A)*det(B)",
    "determinant_additive": "det(A+B) = det(A)+det(B)",
    "inverse_reverse": "(A*B)^(-1) = B^(-1)*A^(-1)",
    "inverse_same_order": "(A*B)^(-1) = A^(-1)*B^(-1)",
    "inverse_additive": "(A+B)^(-1) = A^(-1)+B^(-1)",
}

CONDITION_TEXT = {
    "all": (
        "A and B are arbitrary 2x2 matrices with rational entries."
    ),
    "invertible_pair": (
        "A and B are invertible 2x2 matrices with rational entries."
    ),
    "invertible_pair_and_sum": (
        "A, B, and A+B are invertible 2x2 matrices with rational entries."
    ),
}

REPAIR_TEXT = {
    "none": "No repair is required.",
    "reverse_transpose_order": (
        "Use (A*B)^T = B^T*A^T."
    ),
    "expand_2x2_determinant": (
        "Replace determinant additivity with the explicit 2x2 expansion."
    ),
    "reverse_inverse_order": (
        "Use (A*B)^(-1) = B^(-1)*A^(-1)."
    ),
    "drop_inverse_additivity": (
        "There is no universal identity (A+B)^(-1)=A^(-1)+B^(-1)."
    ),
}


@dataclass(frozen=True)
class Matrix2:
    a: Fraction
    b: Fraction
    c: Fraction
    d: Fraction

    @classmethod
    def from_rows(
        cls,
        rows: tuple[
            tuple[Fraction, Fraction],
            tuple[Fraction, Fraction],
        ],
    ) -> "Matrix2":
        return cls(
            a=rows[0][0],
            b=rows[0][1],
            c=rows[1][0],
            d=rows[1][1],
        )

    def to_rows(self) -> list[list[str]]:
        return [
            [str(self.a), str(self.b)],
            [str(self.c), str(self.d)],
        ]

    def transpose(self) -> "Matrix2":
        return Matrix2(self.a, self.c, self.b, self.d)

    def add(self, other: "Matrix2") -> "Matrix2":
        return Matrix2(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c,
            self.d + other.d,
        )

    def multiply(self, other: "Matrix2") -> "Matrix2":
        return Matrix2(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    def determinant(self) -> Fraction:
        return self.a * self.d - self.b * self.c

    def trace(self) -> Fraction:
        return self.a + self.d

    def inverse(self) -> "Matrix2":
        determinant = self.determinant()
        if determinant == 0:
            raise ValueError("Matrix must be invertible")

        return Matrix2(
            self.d / determinant,
            -self.b / determinant,
            -self.c / determinant,
            self.a / determinant,
        )


@dataclass(frozen=True)
class MatrixState:
    A: Matrix2
    B: Matrix2

    def validate(self) -> None:
        if not isinstance(self.A, Matrix2) or not isinstance(
            self.B,
            Matrix2,
        ):
            raise ValueError("A and B must be 2x2 matrices")

    def to_dict(self) -> dict[str, object]:
        return {
            "A": self.A.to_rows(),
            "B": self.B.to_rows(),
        }


@dataclass(frozen=True)
class LinearAlgebraCounterexample:
    state: MatrixState
    expected: Matrix2 | Fraction
    actual: Matrix2 | Fraction

    def to_dict(self) -> dict[str, object]:
        def serialize(value: Matrix2 | Fraction) -> object:
            if isinstance(value, Matrix2):
                return value.to_rows()
            return str(value)

        return {
            "state": self.state.to_dict(),
            "expected": serialize(self.expected),
            "actual": serialize(self.actual),
        }


@dataclass(frozen=True)
class FormulaSpec:
    formula_id: str
    condition: str
    verdict: str
    failure_mode: str
    repair: str


FORMULA_SPECS = (
    FormulaSpec(
        "transpose_reverse",
        "all",
        "valid",
        "none",
        "none",
    ),
    FormulaSpec(
        "transpose_same_order",
        "all",
        "invalid",
        "transpose_order",
        "reverse_transpose_order",
    ),
    FormulaSpec(
        "trace_cyclic",
        "all",
        "valid",
        "none",
        "none",
    ),
    FormulaSpec(
        "determinant_product",
        "all",
        "valid",
        "none",
        "none",
    ),
    FormulaSpec(
        "determinant_additive",
        "all",
        "invalid",
        "determinant_nonlinearity",
        "expand_2x2_determinant",
    ),
    FormulaSpec(
        "inverse_reverse",
        "invertible_pair",
        "valid",
        "none",
        "none",
    ),
    FormulaSpec(
        "inverse_same_order",
        "invertible_pair",
        "invalid",
        "inverse_order",
        "reverse_inverse_order",
    ),
    FormulaSpec(
        "inverse_additive",
        "invertible_pair_and_sum",
        "invalid",
        "inverse_additivity",
        "drop_inverse_additivity",
    ),
)

SPEC_BY_ID = {
    spec.formula_id: spec for spec in FORMULA_SPECS
}


def _matrix(
    a: int,
    b: int,
    c: int,
    d: int,
) -> Matrix2:
    return Matrix2(
        Fraction(a),
        Fraction(b),
        Fraction(c),
        Fraction(d),
    )


def diagnostic_states() -> tuple[MatrixState, ...]:
    identity = _matrix(1, 0, 0, 1)
    upper = _matrix(1, 1, 0, 1)
    lower = _matrix(1, 0, 1, 1)
    diagonal = _matrix(2, 0, 0, 3)
    swap = _matrix(0, 1, 1, 0)

    return (
        MatrixState(identity, diagonal),
        MatrixState(upper, lower),
        MatrixState(diagonal, upper),
        MatrixState(swap, upper),
    )


def condition_holds(state: MatrixState, condition: str) -> bool:
    state.validate()

    if condition == "all":
        return True

    A_invertible = state.A.determinant() != 0
    B_invertible = state.B.determinant() != 0

    if condition == "invertible_pair":
        return A_invertible and B_invertible

    if condition == "invertible_pair_and_sum":
        return (
            A_invertible
            and B_invertible
            and state.A.add(state.B).determinant() != 0
        )

    raise ValueError(f"Unknown linear-algebra condition: {condition}")


def expected_and_actual(
    state: MatrixState,
    formula_id: str,
    condition: str,
) -> tuple[Matrix2 | Fraction, Matrix2 | Fraction]:
    if formula_id not in FORMULA_IDS:
        raise ValueError(f"Unknown linear-algebra formula: {formula_id}")
    if not condition_holds(state, condition):
        raise ValueError("State does not satisfy the stated condition")

    A = state.A
    B = state.B
    product = A.multiply(B)

    if formula_id == "transpose_reverse":
        return product.transpose(), B.transpose().multiply(A.transpose())

    if formula_id == "transpose_same_order":
        return product.transpose(), A.transpose().multiply(B.transpose())

    if formula_id == "trace_cyclic":
        return product.trace(), B.multiply(A).trace()

    if formula_id == "determinant_product":
        return product.determinant(), A.determinant() * B.determinant()

    if formula_id == "determinant_additive":
        return A.add(B).determinant(), (
            A.determinant() + B.determinant()
        )

    if formula_id == "inverse_reverse":
        return product.inverse(), B.inverse().multiply(A.inverse())

    if formula_id == "inverse_same_order":
        return product.inverse(), A.inverse().multiply(B.inverse())

    if formula_id == "inverse_additive":
        return A.add(B).inverse(), A.inverse().add(B.inverse())

    raise AssertionError("Formula validation should have returned earlier")


def claim_holds(
    state: MatrixState,
    formula_id: str,
    condition: str,
) -> bool:
    expected, actual = expected_and_actual(
        state,
        formula_id,
        condition,
    )
    return expected == actual


def find_counterexample(
    formula_id: str,
    condition: str,
    *,
    states: Iterable[MatrixState] | None = None,
) -> LinearAlgebraCounterexample | None:
    candidates = diagnostic_states() if states is None else states

    for state in candidates:
        if not condition_holds(state, condition):
            continue

        expected, actual = expected_and_actual(
            state,
            formula_id,
            condition,
        )
        if expected != actual:
            return LinearAlgebraCounterexample(
                state=state,
                expected=expected,
                actual=actual,
            )

    return None