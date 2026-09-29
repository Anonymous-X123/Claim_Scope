"""Command-line interface for development and agent experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import FORMULA_IDS, audit_candidate
from .tasks import TASKS, TASK_BY_ID


def _write_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def command_list(_: argparse.Namespace) -> int:
    for task in TASKS:
        print(f"{task.task_id}\t{task.title}")
    return 0


def command_show(args: argparse.Namespace) -> int:
    task = TASK_BY_ID[args.task_id]
    print(task.prompt())
    if args.include_answer:
        _write_json(
            {
                "task_id": task.task_id,
                "verdict": task.expected_verdict,
                "scope": task.expected_scope,
                "failure_mode": task.failure_mode,
                "repair": task.repair,
            }
        )
    return 0


def command_check(args: argparse.Namespace) -> int:
    task = TASK_BY_ID[args.task_id]
    audit = audit_candidate(
        args.formula_id,
        task.protocol,
        task.quantity,
        condition=task.condition,
    )
    _write_json(audit.to_dict())
    return 0 if audit.valid_on_grid else 1


def command_export(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in TASKS:
            handle.write(
                json.dumps(
                    {"task_id": task.task_id, "prompt": task.prompt()},
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
            if task_id not in TASK_BY_ID:
                raise ValueError(f"line {line_number}: unknown task_id {task_id!r}")
            if task_id in predictions:
                raise ValueError(f"line {line_number}: duplicate task_id {task_id!r}")
            predictions[str(task_id)] = row
    return predictions


def command_score(args: argparse.Namespace) -> int:
    predictions = _read_predictions(Path(args.predictions))
    rows: list[dict[str, object]] = []
    for task in TASKS:
        prediction = predictions.get(task.task_id, {})
        verdict_ok = prediction.get("verdict") == task.expected_verdict
        scope_ok = prediction.get("scope") == task.expected_scope
        accepted_modes = {task.failure_mode, *task.alternative_failure_modes}
        mode_ok = prediction.get("failure_mode") in accepted_modes
        repair_id = prediction.get("repair")
        repair_ok = repair_id == task.repair
        if task.expected_verdict == "invalid" and repair_id in FORMULA_IDS:
            repair_ok = audit_candidate(
                str(repair_id),
                task.protocol,
                task.quantity,
                condition=task.condition,
            ).valid_on_grid
        # Failure-mode labels are useful diagnostics but can admit multiple
        # mathematically defensible descriptions. Keep them out of the strict
        # primary endpoint.
        safe = verdict_ok and scope_ok and repair_ok
        rows.append(
            {
                "task_id": task.task_id,
                "verdict_ok": verdict_ok,
                "scope_ok": scope_ok,
                "failure_mode_ok": mode_ok,
                "repair_ok": repair_ok,
                "safe_completion": safe,
            }
        )

    count = len(rows)
    summary = {
        "tasks": count,
        "submitted": len(predictions),
        "audit_accuracy": sum(bool(row["verdict_ok"]) for row in rows) / count,
        "scope_accuracy": sum(bool(row["scope_ok"]) for row in rows) / count,
        "failure_mode_accuracy": sum(
            bool(row["failure_mode_ok"]) for row in rows
        )
        / count,
        "repair_validity": sum(bool(row["repair_ok"]) for row in rows) / count,
        "safe_completion_rate": sum(
            bool(row["safe_completion"]) for row in rows
        )
        / count,
        "per_task": rows,
    }
    _write_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claimscope")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list development tasks")
    list_parser.set_defaults(function=command_list)

    show_parser = subparsers.add_parser("show", help="show one task prompt")
    show_parser.add_argument("task_id", choices=sorted(TASK_BY_ID))
    show_parser.add_argument("--include-answer", action="store_true")
    show_parser.set_defaults(function=command_show)

    check_parser = subparsers.add_parser(
        "check", help="check a candidate formula and return a witness on failure"
    )
    check_parser.add_argument("task_id", choices=sorted(TASK_BY_ID))
    check_parser.add_argument("formula_id", choices=FORMULA_IDS)
    check_parser.set_defaults(function=command_check)

    export_parser = subparsers.add_parser("export", help="export prompts as JSONL")
    export_parser.add_argument("--output", required=True)
    export_parser.set_defaults(function=command_export)

    score_parser = subparsers.add_parser("score", help="score prediction JSONL")
    score_parser.add_argument("predictions")
    score_parser.set_defaults(function=command_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
