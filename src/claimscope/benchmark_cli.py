"""Unified CLI for all ClaimScope mathematical families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import (
    FORMULA_IDS as OPTIMIZER_FORMULA_IDS,
    audit_candidate,
)
from .recurrence import (
    FORMULA_IDS as RECURRENCE_FORMULA_IDS,
    TASKS as RECURRENCE_TASKS,
    TASK_BY_ID as RECURRENCE_BY_ID,
    audit_recurrence,
)
from .tasks import (
    TASKS as OPTIMIZER_TASKS,
    TASK_BY_ID as OPTIMIZER_BY_ID,
)


RECURRENCE_FORMULA_TEXT = {
    "unroll_exact": (
        "x_T=rho^T*x_0 + "
        "sum_{k=0}^{T-1} rho^(T-1-k)*u_k"
    ),
    "input_off_by_one": (
        "x_T=rho^T*x_0 + "
        "sum_{k=0}^{T-1} rho^(T-k)*u_k"
    ),
    "initial_off_by_one": (
        "x_T=rho^(T-1)*x_0 + "
        "sum_{k=0}^{T-1} rho^(T-1-k)*u_k"
    ),
    "constant_geometric": (
        "x_T=rho^T*x_0 + u*(1-rho^T)/(1-rho)"
    ),
    "constant_piecewise": (
        "if rho=1, x_T=x_0+T*u; otherwise "
        "x_T=rho^T*x_0+u*(1-rho^T)/(1-rho)"
    ),
    "rho_one_linear": "x_T=x_0+T*u",
}


RECURRENCE_CONDITION_TEXT = {
    "all": "No additional restrictions are allowed.",
    "constant_input": "Assume u_k=u for every k.",
    "rho_not_one_constant": (
        "Assume u_k=u for every k and rho is not equal to one."
    ),
    "rho_one_constant": (
        "Assume u_k=u for every k and rho equals one."
    ),
}


ALL_TASK_IDS = tuple(
    [task.task_id for task in OPTIMIZER_TASKS]
    + [task.task_id for task in RECURRENCE_TASKS]
)


def _write_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def task_family(task_id: str) -> str:
    if task_id in OPTIMIZER_BY_ID:
        return "optimizer_transport"
    if task_id in RECURRENCE_BY_ID:
        return "affine_recurrence"
    raise KeyError(f"unknown task_id: {task_id}")


def recurrence_prompt(task_id: str) -> str:
    task = RECURRENCE_BY_ID[task_id]
    candidate = RECURRENCE_FORMULA_TEXT[task.formula_id]
    assumptions = RECURRENCE_CONDITION_TEXT[task.condition]

    return (
        f"Task ID: {task.task_id}\n\n"
        "Consider the scalar recurrence x_(k+1)=rho*x_k+u_k "
        "for k=0,...,T-1.\n\n"
        f"Candidate: {candidate}\n"
        "Claim: the candidate equals the exact finite-horizon state x_T "
        "for every admissible state.\n"
        f"Assumptions: {assumptions}\n\n"
        "Return JSON with exactly these keys: task_id, verdict, scope, "
        "failure_mode, and repair.\n"
        "Allowed verdicts: valid, invalid.\n"
        "The scope must be finite_trajectory.\n"
        "Allowed failure modes: none, input_exponent, initial_exponent, "
        "singular_case, domain_restriction.\n"
        "Allowed repairs: none; unroll_exact, meaning "
        "x_T=rho^T*x_0+sum rho^(T-1-k)*u_k; "
        "constant_piecewise, meaning use x_0+T*u at rho=1 and the "
        "geometric formula otherwise.\n"
        "Use repair=none if and only if the stated claim is valid."
    )


def task_prompt(task_id: str) -> str:
    if task_id in OPTIMIZER_BY_ID:
        return OPTIMIZER_BY_ID[task_id].prompt()
    if task_id in RECURRENCE_BY_ID:
        return recurrence_prompt(task_id)
    raise KeyError(f"unknown task_id: {task_id}")


def command_list(_: argparse.Namespace) -> int:
    for task in OPTIMIZER_TASKS:
        print(f"{task.task_id}\toptimizer_transport\t{task.title}")
    for task in RECURRENCE_TASKS:
        print(f"{task.task_id}\taffine_recurrence\t{task.title}")
    return 0


def command_show(args: argparse.Namespace) -> int:
    print(task_prompt(args.task_id))
    return 0


def command_check(args: argparse.Namespace) -> int:
    if args.task_id in OPTIMIZER_BY_ID:
        task = OPTIMIZER_BY_ID[args.task_id]
        if args.formula_id not in OPTIMIZER_FORMULA_IDS:
            raise ValueError(
                f"{args.formula_id!r} is not an optimizer formula"
            )
        audit = audit_candidate(
            args.formula_id,
            task.protocol,
            task.quantity,
            condition=task.condition,
        )
    elif args.task_id in RECURRENCE_BY_ID:
        task = RECURRENCE_BY_ID[args.task_id]
        if args.formula_id not in RECURRENCE_FORMULA_IDS:
            raise ValueError(
                f"{args.formula_id!r} is not a recurrence formula"
            )
        audit = audit_recurrence(
            args.formula_id,
            condition=task.condition,
        )
    else:
        raise KeyError(f"unknown task_id: {args.task_id}")

    _write_json(audit.to_dict())
    return 0 if audit.valid_on_grid else 1


def command_export(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for task_id in ALL_TASK_IDS:
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "family": task_family(task_id),
                        "prompt": task_prompt(task_id),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    print(output.resolve())
    return 0


def _read_predictions(path: Path) -> dict[str, dict[str, object]]:
    predictions: dict[str, dict[str, object]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            task_id = row.get("task_id")

            if task_id not in ALL_TASK_IDS:
                raise ValueError(
                    f"line {line_number}: unknown task_id {task_id!r}"
                )
            if task_id in predictions:
                raise ValueError(
                    f"line {line_number}: duplicate task_id {task_id!r}"
                )

            predictions[str(task_id)] = row

    return predictions


def _score_optimizer(
    task_id: str,
    prediction: dict[str, object],
) -> dict[str, object]:
    task = OPTIMIZER_BY_ID[task_id]

    verdict_ok = prediction.get("verdict") == task.expected_verdict
    scope_ok = prediction.get("scope") == task.expected_scope

    accepted_modes = {
        task.failure_mode,
        *task.alternative_failure_modes,
    }
    mode_ok = prediction.get("failure_mode") in accepted_modes

    repair_id = prediction.get("repair")
    repair_ok = repair_id == task.repair

    if (
        task.expected_verdict == "invalid"
        and repair_id in OPTIMIZER_FORMULA_IDS
    ):
        repair_ok = audit_candidate(
            str(repair_id),
            task.protocol,
            task.quantity,
            condition=task.condition,
        ).valid_on_grid

    safe = verdict_ok and scope_ok and repair_ok

    return {
        "task_id": task_id,
        "family": "optimizer_transport",
        "verdict_ok": verdict_ok,
        "scope_ok": scope_ok,
        "failure_mode_ok": mode_ok,
        "repair_ok": repair_ok,
        "safe_completion": safe,
    }


def _score_recurrence(
    task_id: str,
    prediction: dict[str, object],
) -> dict[str, object]:
    task = RECURRENCE_BY_ID[task_id]

    verdict_ok = prediction.get("verdict") == task.expected_verdict
    scope_ok = prediction.get("scope") == "finite_trajectory"
    mode_ok = prediction.get("failure_mode") == task.failure_mode

    repair_id = prediction.get("repair")
    repair_ok = repair_id == task.repair

    if (
        task.expected_verdict == "invalid"
        and repair_id in RECURRENCE_FORMULA_IDS
    ):
        repair_ok = audit_recurrence(
            str(repair_id),
            condition=task.condition,
        ).valid_on_grid

    safe = verdict_ok and scope_ok and repair_ok

    return {
        "task_id": task_id,
        "family": "affine_recurrence",
        "verdict_ok": verdict_ok,
        "scope_ok": scope_ok,
        "failure_mode_ok": mode_ok,
        "repair_ok": repair_ok,
        "safe_completion": safe,
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row[key]) for row in rows) / len(rows)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "tasks": len(rows),
        "audit_accuracy": _mean(rows, "verdict_ok"),
        "scope_accuracy": _mean(rows, "scope_ok"),
        "failure_mode_accuracy": _mean(
            rows,
            "failure_mode_ok",
        ),
        "repair_validity": _mean(rows, "repair_ok"),
        "safe_completion_rate": _mean(
            rows,
            "safe_completion",
        ),
    }


def command_score(args: argparse.Namespace) -> int:
    predictions = _read_predictions(Path(args.predictions))
    rows: list[dict[str, object]] = []

    for task_id in ALL_TASK_IDS:
        prediction = predictions.get(task_id, {})

        if task_id in OPTIMIZER_BY_ID:
            rows.append(_score_optimizer(task_id, prediction))
        else:
            rows.append(_score_recurrence(task_id, prediction))

    by_family = {}
    for family in ("optimizer_transport", "affine_recurrence"):
        family_rows = [
            row for row in rows if row["family"] == family
        ]
        by_family[family] = _summarize(family_rows)

    summary = _summarize(rows)
    summary["submitted"] = len(predictions)
    summary["by_family"] = by_family
    summary["per_task"] = rows

    _write_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claimscope")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    list_parser = subparsers.add_parser(
        "list",
        help="list all tasks",
    )
    list_parser.set_defaults(function=command_list)

    show_parser = subparsers.add_parser(
        "show",
        help="show one task",
    )
    show_parser.add_argument("task_id", choices=ALL_TASK_IDS)
    show_parser.set_defaults(function=command_show)

    check_parser = subparsers.add_parser(
        "check",
        help="check a candidate formula",
    )
    check_parser.add_argument("task_id", choices=ALL_TASK_IDS)
    check_parser.add_argument("formula_id")
    check_parser.set_defaults(function=command_check)

    export_parser = subparsers.add_parser(
        "export",
        help="export all prompts",
    )
    export_parser.add_argument("--output", required=True)
    export_parser.set_defaults(function=command_export)

    score_parser = subparsers.add_parser(
        "score",
        help="score prediction JSONL",
    )
    score_parser.add_argument("predictions")
    score_parser.set_defaults(function=command_score)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
