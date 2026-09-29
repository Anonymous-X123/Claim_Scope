"""Exact oracle for finite-horizon affine recurrence claims."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable


FORMULA_IDS = (
    "unroll_exact",
    "input_off_by_one",
    "initial_off_by_one",
    "constant_geometric",
    "constant_piecewise",
    "rho_one_linear",
)


class UndefinedCandidate(ValueError):
    """Raised when a candidate formula is undefined under the stated domain."""


@dataclass(frozen=True)
class RecurrenceState:
    """State for x_(k+1) = rho*x_k + u_k."""

    rho: Fraction
    x0: Fraction
    inputs: tuple[Fraction, ...]

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("at least one input is required")

    @property
    def horizon(self) -> int:
        return len(self.inputs)

    @property
    def is_constant_input(self) -> bool:
        return all(value == self.inputs[0] for value in self.inputs)

    @property
    def constant_input(self) -> Fraction:
        if not self.is_constant_input:
            raise UndefinedCandidate("the input sequence is not constant")
        return self.inputs[0]

    def iterative_value(self) -> Fraction:
        self.validate()
        value = self.x0
        for forcing in self.inputs:
            value = self.rho * value + forcing
        return value


@dataclass(frozen=True)
class RecurrenceCounterexample:
    state: RecurrenceState
    expected: Fraction
    actual: Fraction | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "state": {
                "rho": str(self.state.rho),
                "x0": str(self.state.x0),
                "inputs": [str(value) for value in self.state.inputs],
                "horizon": self.state.horizon,
            },
            "expected": str(self.expected),
            "actual": None if self.actual is None else str(self.actual),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RecurrenceAudit:
    formula_id: str
    condition: str
    cases_checked: int
    valid_on_grid: bool
    counterexample: RecurrenceCounterexample | None

    def to_dict(self) -> dict[str, object]:
        return {
            "formula_id": self.formula_id,
            "condition": self.condition,
            "cases_checked": self.cases_checked,
            "valid_on_grid": self.valid_on_grid,
            "counterexample": (
                None
                if self.counterexample is None
                else self.counterexample.to_dict()
            ),
        }


def diagnostic_states() -> tuple[RecurrenceState, ...]:
    """Fixed rational states in a reproducible diagnostic order."""

    return (
        RecurrenceState(
            Fraction(1, 2),
            Fraction(3),
            (Fraction(1), Fraction(-2), Fraction(4)),
        ),
        RecurrenceState(
            Fraction(-1, 2),
            Fraction(-1),
            (Fraction(2), Fraction(2), Fraction(2)),
        ),
        RecurrenceState(
            Fraction(2),
            Fraction(1),
            (Fraction(1), Fraction(1), Fraction(1)),
        ),
        RecurrenceState(
            Fraction(1),
            Fraction(5),
            (Fraction(-2), Fraction(-2), Fraction(-2)),
        ),
        RecurrenceState(
            Fraction(0),
            Fraction(7),
            (Fraction(3), Fraction(3)),
        ),
        RecurrenceState(
            Fraction(3, 2),
            Fraction(-2),
            (Fraction(1), Fraction(0), Fraction(-1)),
        ),
    )


def exact_unroll(state: RecurrenceState) -> Fraction:
    state.validate()
    horizon = state.horizon
    forcing_sum = sum(
        (
            state.rho ** (horizon - 1 - index)
            * forcing
            for index, forcing in enumerate(state.inputs)
        ),
        Fraction(0),
    )
    return state.rho**horizon * state.x0 + forcing_sum


def candidate_value(state: RecurrenceState, formula_id: str) -> Fraction:
    state.validate()
    horizon = state.horizon

    exact_forcing = sum(
        (
            state.rho ** (horizon - 1 - index)
            * forcing
            for index, forcing in enumerate(state.inputs)
        ),
        Fraction(0),
    )

    if formula_id == "unroll_exact":
        return state.rho**horizon * state.x0 + exact_forcing

    if formula_id == "input_off_by_one":
        wrong_forcing = sum(
            (
                state.rho ** (horizon - index)
                * forcing
                for index, forcing in enumerate(state.inputs)
            ),
            Fraction(0),
        )
        return state.rho**horizon * state.x0 + wrong_forcing

    if formula_id == "initial_off_by_one":
        return state.rho ** (horizon - 1) * state.x0 + exact_forcing

    if formula_id == "constant_geometric":
        forcing = state.constant_input
        if state.rho == 1:
            raise UndefinedCandidate("division by zero at rho=1")
        return (
            state.rho**horizon * state.x0
            + forcing
            * (1 - state.rho**horizon)
            / (1 - state.rho)
        )

    if formula_id == "constant_piecewise":
        forcing = state.constant_input
        if state.rho == 1:
            return state.x0 + horizon * forcing
        return (
            state.rho**horizon * state.x0
            + forcing
            * (1 - state.rho**horizon)
            / (1 - state.rho)
        )

    if formula_id == "rho_one_linear":
        forcing = state.constant_input
        return state.x0 + horizon * forcing

    raise KeyError(f"unknown formula_id: {formula_id}")


def _condition(name: str) -> Callable[[RecurrenceState], bool]:
    conditions: dict[str, Callable[[RecurrenceState], bool]] = {
        "all": lambda _: True,
        "constant_input": lambda state: state.is_constant_input,
        "rho_not_one_constant": lambda state: (
            state.rho != 1 and state.is_constant_input
        ),
        "rho_one_constant": lambda state: (
            state.rho == 1 and state.is_constant_input
        ),
    }
    try:
        return conditions[name]
    except KeyError as error:
        raise KeyError(f"unknown condition: {name}") from error


def find_counterexample(
    formula_id: str,
    *,
    condition: str = "all",
    states: Iterable[RecurrenceState] | None = None,
) -> RecurrenceCounterexample | None:
    if formula_id not in FORMULA_IDS:
        raise KeyError(f"unknown formula_id: {formula_id}")

    predicate = _condition(condition)
    eligible = [
        state
        for state in (states or diagnostic_states())
        if predicate(state)
    ]
    if not eligible:
        raise ValueError(f"condition {condition!r} selects no states")

    for state in eligible:
        expected = state.iterative_value()
        try:
            actual = candidate_value(state, formula_id)
        except UndefinedCandidate:
            return RecurrenceCounterexample(
                state=state,
                expected=expected,
                actual=None,
                reason="candidate_undefined",
            )

        if actual != expected:
            return RecurrenceCounterexample(
                state=state,
                expected=expected,
                actual=actual,
                reason="value_mismatch",
            )

    return None


def audit_recurrence(
    formula_id: str,
    *,
    condition: str = "all",
) -> RecurrenceAudit:
    predicate = _condition(condition)
    eligible = [
        state for state in diagnostic_states() if predicate(state)
    ]
    counterexample = find_counterexample(
        formula_id,
        condition=condition,
        states=eligible,
    )
    return RecurrenceAudit(
        formula_id=formula_id,
        condition=condition,
        cases_checked=len(eligible),
        valid_on_grid=counterexample is None,
        counterexample=counterexample,
    )


@dataclass(frozen=True)
class RecurrenceTask:
    task_id: str
    title: str
    formula_id: str
    condition: str
    expected_verdict: str
    failure_mode: str
    repair: str


TASKS = (
    RecurrenceTask(
        "RC001",
        "Exact finite-horizon unrolling",
        "unroll_exact",
        "all",
        "valid",
        "none",
        "none",
    ),
    RecurrenceTask(
        "RC002",
        "Off-by-one forcing exponent",
        "input_off_by_one",
        "all",
        "invalid",
        "input_exponent",
        "unroll_exact",
    ),
    RecurrenceTask(
        "RC003",
        "Off-by-one initial-state exponent",
        "initial_off_by_one",
        "all",
        "invalid",
        "initial_exponent",
        "unroll_exact",
    ),
    RecurrenceTask(
        "RC004",
        "Geometric formula away from rho one",
        "constant_geometric",
        "rho_not_one_constant",
        "valid",
        "none",
        "none",
    ),
    RecurrenceTask(
        "RC005",
        "Geometric formula without singular-case handling",
        "constant_geometric",
        "constant_input",
        "invalid",
        "singular_case",
        "constant_piecewise",
    ),
    RecurrenceTask(
        "RC006",
        "Piecewise constant-input formula",
        "constant_piecewise",
        "constant_input",
        "valid",
        "none",
        "none",
    ),
    RecurrenceTask(
        "RC007",
        "Linear formula at rho one",
        "rho_one_linear",
        "rho_one_constant",
        "valid",
        "none",
        "none",
    ),
    RecurrenceTask(
        "RC008",
        "Linear formula outside rho one",
        "rho_one_linear",
        "constant_input",
        "invalid",
        "domain_restriction",
        "constant_piecewise",
    ),
)


TASK_BY_ID = {task.task_id: task for task in TASKS}


def audit_task(task_id: str) -> RecurrenceAudit:
    task = TASK_BY_ID[task_id]
    return audit_recurrence(
        task.formula_id,
        condition=task.condition,
    )
