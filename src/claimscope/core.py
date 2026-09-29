"""Exact-by-construction numerical oracle for the ClaimScope development set.

The transport identities are coordinatewise, so scalar witnesses suffice to
disprove a purported universal tensor identity. The checker uses a fixed,
non-degenerate diagnostic grid to keep feedback deterministic and auditable.
It is a testing oracle, not a symbolic proof assistant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, isfinite, sqrt
from typing import Callable, Iterable


FORMULA_IDS = (
    "full_transport",
    "direction_transport",
    "missing_lr_ratio",
    "first_step_bias",
    "drop_epsilon",
    "wrong_v_bias",
    "decay_sign_reversed",
    "off_by_one_bias",
)

PROTOCOLS = ("adamw_decay", "muon_decay")
QUANTITIES = ("full_transition", "optimizer_direction")


@dataclass(frozen=True)
class State:
    """One scalar coordinate of a transition state."""

    w: float
    d: float
    v: float
    alpha: float
    eta: float
    lambda_m: float
    lambda_a: float
    beta1: float
    beta2: float
    step: int
    epsilon: float

    def validate(self) -> None:
        values = (
            self.w,
            self.d,
            self.v,
            self.alpha,
            self.eta,
            self.lambda_m,
            self.lambda_a,
            self.beta1,
            self.beta2,
            self.epsilon,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("state contains a non-finite value")
        if self.v < 0:
            raise ValueError("v must be nonnegative")
        if self.alpha <= 0 or self.eta <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.beta1 < 1:
            raise ValueError("beta1 must lie in (0, 1)")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must lie in [0, 1)")
        if self.step < 1:
            raise ValueError("step must be a positive integer")
        if self.epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if self.q == 0:
            raise ValueError("the adaptive denominator must be positive")

    @property
    def q(self) -> float:
        return sqrt(self.v / (1 - self.beta2**self.step)) + self.epsilon

    @property
    def target_direction(self) -> float:
        """Direction Adam must take so eta * A equals alpha * D."""

        return (self.alpha / self.eta) * self.d

    @property
    def target_update(self) -> float:
        return (1 - self.alpha * self.lambda_m) * self.w - self.alpha * self.d


@dataclass(frozen=True)
class Counterexample:
    state: State
    expected: float
    actual: float
    absolute_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "state": asdict(self.state),
            "expected": self.expected,
            "actual": self.actual,
            "absolute_error": self.absolute_error,
        }


@dataclass(frozen=True)
class CandidateAudit:
    formula_id: str
    protocol: str
    quantity: str
    cases_checked: int
    valid_on_grid: bool
    counterexample: Counterexample | None

    def to_dict(self) -> dict[str, object]:
        return {
            "formula_id": self.formula_id,
            "protocol": self.protocol,
            "quantity": self.quantity,
            "cases_checked": self.cases_checked,
            "valid_on_grid": self.valid_on_grid,
            "counterexample": (
                None if self.counterexample is None else self.counterexample.to_dict()
            ),
        }


def candidate_moment(state: State, formula_id: str) -> float:
    """Construct the candidate post-transition first moment."""

    state.validate()
    first_bias = 1 - state.beta1**state.step
    full_residual = (
        state.alpha * state.lambda_m / state.eta - state.lambda_a
    ) * state.w

    if formula_id == "full_transport":
        return first_bias * (state.target_direction + full_residual) * state.q
    if formula_id == "direction_transport":
        return first_bias * state.target_direction * state.q
    if formula_id == "missing_lr_ratio":
        return first_bias * state.d * state.q
    if formula_id == "first_step_bias":
        return (1 - state.beta1) * state.target_direction * state.q
    if formula_id == "drop_epsilon":
        q_without_epsilon = sqrt(state.v / (1 - state.beta2**state.step))
        return first_bias * state.target_direction * q_without_epsilon
    if formula_id == "wrong_v_bias":
        # Deliberately omit the square root on the bias-correction factor.
        # This isolates second-moment normalization from epsilon placement.
        wrong_q = sqrt(state.v) / (1 - state.beta2**state.step) + state.epsilon
        return first_bias * state.target_direction * wrong_q
    if formula_id == "decay_sign_reversed":
        reversed_residual = (
            state.lambda_a - state.alpha * state.lambda_m / state.eta
        ) * state.w
        return first_bias * (state.target_direction + reversed_residual) * state.q
    if formula_id == "off_by_one_bias":
        return (1 - state.beta1 ** (state.step + 1)) * state.target_direction * state.q
    raise KeyError(f"unknown formula_id: {formula_id}")


def reconstructed_direction(state: State, formula_id: str) -> float:
    moment = candidate_moment(state, formula_id)
    return (moment / (1 - state.beta1**state.step)) / state.q


def candidate_update(state: State, formula_id: str, protocol: str) -> float:
    direction = reconstructed_direction(state, formula_id)
    if protocol == "adamw_decay":
        decay_factor = 1 - state.eta * state.lambda_a
    elif protocol == "muon_decay":
        decay_factor = 1 - state.alpha * state.lambda_m
    else:
        raise KeyError(f"unknown protocol: {protocol}")
    return decay_factor * state.w - state.eta * direction


def diagnostic_states() -> tuple[State, ...]:
    """Fixed ordering makes the first returned witness reproducible."""

    return (
        State(2.0, -0.7, 0.16, 0.03, 0.01, 0.02, 0.10, 0.9, 0.99, 7, 1e-3),
        State(-1.5, 0.4, 1.21, 0.02, 0.02, 0.00, 0.05, 0.8, 0.95, 1, 2e-2),
        State(0.75, 1.2, 0.0, 0.05, 0.01, 0.04, 0.00, 0.7, 0.90, 3, 1e-2),
        State(-3.0, -0.2, 2.5, 0.01, 0.04, 0.01, 0.03, 0.95, 0.999, 12, 1e-6),
        State(1.0, 0.8, 0.64, 0.04, 0.04, 0.02, 0.02, 0.85, 0.97, 4, 0.0),
    )


def _condition(name: str) -> Callable[[State], bool]:
    conditions: dict[str, Callable[[State], bool]] = {
        "all": lambda _: True,
        "equal_learning_rates": lambda s: isclose(s.alpha, s.eta),
        "first_step": lambda s: s.step == 1,
        "zero_epsilon": lambda s: s.epsilon == 0 and s.v > 0,
        "matched_decay": lambda s: isclose(
            s.alpha * s.lambda_m, s.eta * s.lambda_a
        ),
    }
    try:
        return conditions[name]
    except KeyError as error:
        raise KeyError(f"unknown condition: {name}") from error


def expected_and_actual(
    state: State,
    formula_id: str,
    protocol: str,
    quantity: str,
) -> tuple[float, float]:
    if quantity == "optimizer_direction":
        return state.target_direction, reconstructed_direction(state, formula_id)
    if quantity == "full_transition":
        return state.target_update, candidate_update(state, formula_id, protocol)
    raise KeyError(f"unknown quantity: {quantity}")


def find_counterexample(
    formula_id: str,
    protocol: str,
    quantity: str,
    *,
    condition: str = "all",
    states: Iterable[State] | None = None,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> Counterexample | None:
    if formula_id not in FORMULA_IDS:
        raise KeyError(f"unknown formula_id: {formula_id}")
    if protocol not in PROTOCOLS:
        raise KeyError(f"unknown protocol: {protocol}")
    if quantity not in QUANTITIES:
        raise KeyError(f"unknown quantity: {quantity}")

    predicate = _condition(condition)
    eligible = [state for state in (states or diagnostic_states()) if predicate(state)]
    if not eligible:
        raise ValueError(f"condition {condition!r} selects no diagnostic states")

    for state in eligible:
        expected, actual = expected_and_actual(state, formula_id, protocol, quantity)
        if not isclose(
            expected,
            actual,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            return Counterexample(
                state=state,
                expected=expected,
                actual=actual,
                absolute_error=abs(expected - actual),
            )
    return None


def audit_candidate(
    formula_id: str,
    protocol: str,
    quantity: str,
    *,
    condition: str = "all",
) -> CandidateAudit:
    states = [state for state in diagnostic_states() if _condition(condition)(state)]
    counterexample = find_counterexample(
        formula_id,
        protocol,
        quantity,
        condition=condition,
        states=states,
    )
    return CandidateAudit(
        formula_id=formula_id,
        protocol=protocol,
        quantity=quantity,
        cases_checked=len(states),
        valid_on_grid=counterexample is None,
        counterexample=counterexample,
    )
