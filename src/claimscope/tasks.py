"""Transparent development tasks for ClaimScope.

These examples are for prompt and evaluator development. They must not be used
as the final held-out evaluation split.
"""

from __future__ import annotations

from dataclasses import dataclass


FORMULA_TEXT = {
    "full_transport": (
        "M=(1-beta1^t)*[(alpha/eta)*D + "
        "((alpha*lambda_M/eta)-lambda_A)*W]*Q"
    ),
    "direction_transport": "M=(1-beta1^t)*(alpha/eta)*D*Q",
    "missing_lr_ratio": "M=(1-beta1^t)*D*Q",
    "first_step_bias": "M=(1-beta1)*(alpha/eta)*D*Q",
    "drop_epsilon": (
        "M=(1-beta1^t)*(alpha/eta)*D*sqrt(V/(1-beta2^t))"
    ),
    "wrong_v_bias": (
        "M=(1-beta1^t)*(alpha/eta)*D*"
        "[sqrt(V)/(1-beta2^t)+epsilon]"
    ),
    "decay_sign_reversed": (
        "M=(1-beta1^t)*[(alpha/eta)*D + "
        "(lambda_A-(alpha*lambda_M/eta))*W]*Q"
    ),
    "off_by_one_bias": "M=(1-beta1^(t+1))*(alpha/eta)*D*Q",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    formula_id: str
    protocol: str
    quantity: str
    condition: str
    expected_verdict: str
    expected_scope: str
    failure_mode: str
    repair: str
    alternative_failure_modes: tuple[str, ...] = ()

    def prompt(self) -> str:
        protocol_text = {
            "adamw_decay": (
                "The transition step is executed by AdamW, including its own "
                "decay factor (1-eta*lambda_A)."
            ),
            "muon_decay": (
                "The transition step applies Muon's decay factor "
                "(1-alpha*lambda_M), then uses Adam's reconstructed direction."
            ),
        }[self.protocol]
        quantity_text = {
            "full_transition": (
                "The claim is that the complete parameter value after this one "
                "transition step equals Muon's update for every admissible state."
            ),
            "optimizer_direction": (
                "The claim is that eta times the reconstructed adaptive direction "
                "equals alpha*D for every admissible state."
            ),
        }[self.quantity]
        condition_text = {
            "all": "No additional equality assumptions are allowed.",
            "equal_learning_rates": "Assume alpha=eta.",
            "first_step": "Assume t=1.",
            "zero_epsilon": "Assume epsilon=0 and V>0.",
            "matched_decay": "Assume alpha*lambda_M=eta*lambda_A.",
        }[self.condition]
        return (
            f"Task ID: {self.task_id}\n\n"
            "Let Q=sqrt(V/(1-beta2^t))+epsilon and let Adam reconstruct "
            "A=[M/(1-beta1^t)]/Q. Muon's reference step is "
            "W_next=(1-alpha*lambda_M)*W-alpha*D.\n\n"
            f"Candidate: {FORMULA_TEXT[self.formula_id]}\n"
            f"Protocol: {protocol_text}\n"
            f"Claim: {quantity_text}\n"
            f"Assumptions: {condition_text}\n\n"
            "Return JSON with exactly these keys: task_id, verdict, scope, "
            "failure_mode, and repair.\n"
            "Allowed verdicts: valid, invalid.\n"
            "Allowed scopes: full_transition, optimizer_direction.\n"
            "Allowed failure modes: none, decay_mismatch, "
            "learning_rate_scaling, bias_correction, epsilon_placement, "
            "second_moment_bias, decay_compensation, guarantee_scope.\n"
            "Allowed repairs: none; direction_transport, meaning "
            "M=(1-beta1^t)*(alpha/eta)*D*Q; full_transport, meaning "
            "M=(1-beta1^t)*[(alpha/eta)*D+"
            "((alpha*lambda_M/eta)-lambda_A)*W]*Q.\n"
            "Use repair=none if and only if the stated claim is valid."
        )


TASKS = (
    Task(
        "CS001",
        "Full decay-compensated transport",
        "full_transport",
        "adamw_decay",
        "full_transition",
        "all",
        "valid",
        "full_transition",
        "none",
        "none",
    ),
    Task(
        "CS002",
        "Direction transport under AdamW decay",
        "direction_transport",
        "adamw_decay",
        "full_transition",
        "all",
        "invalid",
        "full_transition",
        "decay_mismatch",
        "full_transport",
    ),
    Task(
        "CS003",
        "Direction transport under Muon transition decay",
        "direction_transport",
        "muon_decay",
        "full_transition",
        "all",
        "valid",
        "full_transition",
        "none",
        "none",
    ),
    Task(
        "CS004",
        "Missing learning-rate ratio",
        "missing_lr_ratio",
        "muon_decay",
        "optimizer_direction",
        "all",
        "invalid",
        "optimizer_direction",
        "learning_rate_scaling",
        "direction_transport",
    ),
    Task(
        "CS005",
        "Missing ratio with equal learning rates",
        "missing_lr_ratio",
        "muon_decay",
        "optimizer_direction",
        "equal_learning_rates",
        "valid",
        "optimizer_direction",
        "none",
        "none",
    ),
    Task(
        "CS006",
        "Step-one bias factor at arbitrary time",
        "first_step_bias",
        "muon_decay",
        "optimizer_direction",
        "all",
        "invalid",
        "optimizer_direction",
        "bias_correction",
        "direction_transport",
    ),
    Task(
        "CS007",
        "Step-one bias factor at the first step",
        "first_step_bias",
        "muon_decay",
        "optimizer_direction",
        "first_step",
        "valid",
        "optimizer_direction",
        "none",
        "none",
    ),
    Task(
        "CS008",
        "Dropped stabilizer",
        "drop_epsilon",
        "muon_decay",
        "optimizer_direction",
        "all",
        "invalid",
        "optimizer_direction",
        "epsilon_placement",
        "direction_transport",
    ),
    Task(
        "CS009",
        "Dropped stabilizer when epsilon is zero",
        "drop_epsilon",
        "muon_decay",
        "optimizer_direction",
        "zero_epsilon",
        "valid",
        "optimizer_direction",
        "none",
        "none",
    ),
    Task(
        "CS010",
        "Second-moment bias moved across a sum",
        "wrong_v_bias",
        "muon_decay",
        "optimizer_direction",
        "all",
        "invalid",
        "optimizer_direction",
        "second_moment_bias",
        "direction_transport",
    ),
    Task(
        "CS011",
        "Reversed decay compensation",
        "decay_sign_reversed",
        "adamw_decay",
        "full_transition",
        "all",
        "invalid",
        "full_transition",
        "decay_compensation",
        "full_transport",
    ),
    Task(
        "CS012",
        "Off-by-one first-moment counter",
        "off_by_one_bias",
        "muon_decay",
        "optimizer_direction",
        "all",
        "invalid",
        "optimizer_direction",
        "bias_correction",
        "direction_transport",
    ),
    Task(
        "CS013",
        "Full-step repair is not direction preservation",
        "full_transport",
        "adamw_decay",
        "optimizer_direction",
        "all",
        "invalid",
        "optimizer_direction",
        "guarantee_scope",
        "direction_transport",
        ("decay_compensation",),
    ),
    Task(
        "CS014",
        "Direction transport with matched total decay",
        "direction_transport",
        "adamw_decay",
        "full_transition",
        "matched_decay",
        "valid",
        "full_transition",
        "none",
        "none",
    ),
)


TASK_BY_ID = {task.task_id: task for task in TASKS}
