"""Score ClaimScope predictions using computational witness validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from .core import State, find_counterexample
from .linear_algebra import Matrix2, MatrixState
from .linear_algebra import find_counterexample as find_linear_algebra_counterexample
from .recurrence import RecurrenceState
from .recurrence import find_counterexample as find_recurrence_counterexample


SCORER_VERSION = "0.3.0"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT

DEFAULT_GOLD_PATH = (
    PROJECT_ROOT / "artifacts" / "reference" / "extended_v1_gold.jsonl"
)
DEFAULT_MANIFEST_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "extended_v1_manifest.json"
)

REQUIRED_FIELDS = {
    "verdict",
    "scope",
    "failure_mode",
    "counterexample",
    "repair",
}

METRIC_FIELDS = (
    "json_valid",
    "task_id_correct",
    "verdict_correct",
    "scope_correct",
    "counterexample_valid",
    "repair_correct",
    "failure_mode_correct",
    "safe_completion",
    "strict_completion",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: each JSONL row must be an object"
                )
            rows.append(value)

    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _normalize_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _finite_float(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric state value")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError("State values must be finite")
    return result


def _integer(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not an integer")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("Expected an integer")
        return int(value)

    text = str(value).strip()
    result = int(text)
    return result


def _fraction(value: object) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a rational number")

    if isinstance(value, Fraction):
        return value

    if isinstance(value, int):
        return Fraction(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Rational values must be finite")
        return Fraction(str(value))

    return Fraction(str(value).strip())


def _state_object(counterexample: object) -> dict[str, Any]:
    if not isinstance(counterexample, dict):
        raise ValueError("Counterexample must be a JSON object")

    state = counterexample.get("state")
    if not isinstance(state, dict):
        raise ValueError("Counterexample must contain a state object")

    return state


def validate_optimizer_counterexample(
    source: dict[str, Any],
    counterexample: object,
) -> tuple[bool, str]:
    required = {
        "w",
        "d",
        "v",
        "alpha",
        "eta",
        "lambda_m",
        "lambda_a",
        "beta1",
        "beta2",
        "step",
        "epsilon",
    }

    try:
        state_data = _state_object(counterexample)
        missing = sorted(required.difference(state_data))
        if missing:
            raise ValueError(
                "Missing optimizer state fields: " + ", ".join(missing)
            )

        state = State(
            w=_finite_float(state_data["w"]),
            d=_finite_float(state_data["d"]),
            v=_finite_float(state_data["v"]),
            alpha=_finite_float(state_data["alpha"]),
            eta=_finite_float(state_data["eta"]),
            lambda_m=_finite_float(state_data["lambda_m"]),
            lambda_a=_finite_float(state_data["lambda_a"]),
            beta1=_finite_float(state_data["beta1"]),
            beta2=_finite_float(state_data["beta2"]),
            step=_integer(state_data["step"]),
            epsilon=_finite_float(state_data["epsilon"]),
        )
        state.validate()

        witness = find_counterexample(
            source["formula_id"],
            source["protocol"],
            source["quantity"],
            condition=source["condition"],
            states=(state,),
        )

        if witness is None:
            return (
                False,
                "The proposed state does not refute the claim under its condition.",
            )

        return True, "Counterexample verified by the optimizer oracle."

    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return False, f"Invalid optimizer counterexample: {error}"


def validate_recurrence_counterexample(
    source: dict[str, Any],
    counterexample: object,
) -> tuple[bool, str]:
    try:
        state_data = _state_object(counterexample)

        for field in ("rho", "x0", "inputs", "horizon"):
            if field not in state_data:
                raise ValueError(
                    f"Missing recurrence state field: {field}"
                )

        inputs_data = state_data["inputs"]
        if not isinstance(inputs_data, list):
            raise ValueError("inputs must be a JSON list")

        horizon = _integer(state_data["horizon"])
        if horizon != len(inputs_data):
            raise ValueError(
                "horizon must equal the number of supplied inputs"
            )

        state = RecurrenceState(
            rho=_fraction(state_data["rho"]),
            x0=_fraction(state_data["x0"]),
            inputs=tuple(_fraction(value) for value in inputs_data),
        )
        state.validate()

        witness = find_recurrence_counterexample(
            source["formula_id"],
            condition=source["condition"],
            states=(state,),
        )

        if witness is None:
            return (
                False,
                "The proposed state does not refute the claim under its condition.",
            )

        return True, "Counterexample verified by the recurrence oracle."

    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return False, f"Invalid recurrence counterexample: {error}"


def _matrix2(value: object, name: str) -> Matrix2:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a 2x2 JSON list")

    rows: list[tuple[Fraction, Fraction]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{name} must be a 2x2 JSON list")
        rows.append((_fraction(row[0]), _fraction(row[1])))

    return Matrix2.from_rows((rows[0], rows[1]))


def validate_linear_algebra_counterexample(
    source: dict[str, Any],
    counterexample: object,
) -> tuple[bool, str]:
    try:
        state_data = _state_object(counterexample)

        for field in ("A", "B"):
            if field not in state_data:
                raise ValueError(
                    f"Missing linear algebra state field: {field}"
                )

        state = MatrixState(
            A=_matrix2(state_data["A"], "A"),
            B=_matrix2(state_data["B"], "B"),
        )
        state.validate()

        witness = find_linear_algebra_counterexample(
            source["formula_id"],
            condition=source["condition"],
            states=(state,),
        )

        if witness is None:
            return (
                False,
                "The proposed state does not refute the claim under its condition.",
            )

        return (
            True,
            "Counterexample verified by the linear algebra oracle.",
        )

    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return False, f"Invalid linear algebra counterexample: {error}"


def validate_counterexample(
    family: str,
    source: dict[str, Any],
    counterexample: object,
) -> tuple[bool, str]:
    if family == "optimizer_transport":
        return validate_optimizer_counterexample(source, counterexample)

    if family == "affine_recurrence":
        return validate_recurrence_counterexample(source, counterexample)

    if family == "linear_algebra":
        return validate_linear_algebra_counterexample(
            source,
            counterexample,
        )

    return False, f"Unknown task family: {family}"


def _prediction_payload(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    if "prediction" in record:
        value = record["prediction"]
        return value if isinstance(value, dict) else None

    if "parsed" in record:
        value = record["parsed"]
        return value if isinstance(value, dict) else None

    return record


def _reported_task_id(
    record: dict[str, Any],
    payload: dict[str, Any] | None,
) -> str:
    if payload is not None and isinstance(payload.get("task_id"), str):
        return payload["task_id"].strip()

    value = record.get("task_id")
    return value.strip() if isinstance(value, str) else ""


def _routing_task_id(
    record: dict[str, Any],
    payload: dict[str, Any] | None,
) -> str:
    """Return the benchmark task assigned by a trusted runner envelope."""
    if "prediction" in record or "parsed" in record:
        value = record.get("task_id")
        return value.strip() if isinstance(value, str) else ""
    return _reported_task_id(record, payload)


def score_prediction(
    gold_record: dict[str, Any],
    prediction_record: dict[str, Any],
) -> dict[str, Any]:
    payload = _prediction_payload(prediction_record)
    expected_task_id = gold_record["task_id"]
    reported_task_id = _reported_task_id(prediction_record, payload)

    task_id_correct = reported_task_id == expected_task_id
    json_valid = (
        payload is not None
        and task_id_correct
        and REQUIRED_FIELDS.issubset(payload)
    )

    if payload is None:
        payload = {}

    gold = gold_record["gold"]
    expected_verdict = _normalize_label(gold["verdict"])

    verdict_correct = (
        _normalize_label(payload.get("verdict"))
        == expected_verdict
    )
    scope_correct = (
        _normalize_label(payload.get("scope"))
        == _normalize_label(gold["scope"])
    )
    repair_correct = (
        _normalize_label(payload.get("repair"))
        == _normalize_label(gold["repair"])
    )
    failure_mode_correct = (
        _normalize_label(payload.get("failure_mode"))
        == _normalize_label(gold["failure_mode"])
    )

    proposed_counterexample = payload.get("counterexample")

    if expected_verdict == "valid":
        counterexample_valid = proposed_counterexample is None
        counterexample_reason = (
            "Valid claims correctly use a null counterexample."
            if counterexample_valid
            else "Valid claims must use a null counterexample."
        )
    else:
        counterexample_valid, counterexample_reason = (
            validate_counterexample(
                gold_record["family"],
                gold_record["source"],
                proposed_counterexample,
            )
        )

    safe_completion = all(
        (
            json_valid,
            verdict_correct,
            scope_correct,
            counterexample_valid,
            repair_correct,
        )
    )
    strict_completion = (
        safe_completion and failure_mode_correct
    )

    return {
        "task_id": expected_task_id,
        "family": gold_record["family"],
        "model_id": str(
            prediction_record.get("model_id", "unspecified")
        ),
        "protocol": str(
            prediction_record.get("protocol", "unspecified")
        ),
        "json_valid": int(json_valid),
        "task_id_correct": int(task_id_correct),
        "verdict_correct": int(verdict_correct),
        "scope_correct": int(scope_correct),
        "counterexample_valid": int(counterexample_valid),
        "repair_correct": int(repair_correct),
        "failure_mode_correct": int(failure_mode_correct),
        "safe_completion": int(safe_completion),
        "strict_completion": int(strict_completion),
        "counterexample_reason": counterexample_reason,
    }


def _metric_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "prediction_count": 0,
            "unique_task_count": 0,
            **{metric: 0.0 for metric in METRIC_FIELDS},
        }

    result: dict[str, Any] = {
        "prediction_count": len(rows),
        "unique_task_count": len(
            {row["task_id"] for row in rows}
        ),
    }

    for metric in METRIC_FIELDS:
        result[metric] = (
            sum(float(row[metric]) for row in rows) / len(rows)
        )

    return result


def _group_summaries(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        groups[key].append(row)

    return {
        "|".join(key): _metric_summary(group)
        for key, group in sorted(groups.items())
    }


def score_records(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = {
        record["task_id"]: record
        for record in gold_records
    }

    if len(gold_by_id) != len(gold_records):
        raise ValueError("Private gold contains duplicate task IDs")

    scored: list[dict[str, Any]] = []

    for prediction in prediction_records:
        payload = _prediction_payload(prediction)
        task_id = _routing_task_id(prediction, payload)

        if not task_id:
            raise ValueError(
                "Every prediction record must identify its task_id"
            )
        if task_id not in gold_by_id:
            raise ValueError(f"Unknown prediction task_id: {task_id}")

        scored.append(
            score_prediction(gold_by_id[task_id], prediction)
        )

    if not scored:
        raise ValueError("No predictions were supplied")

    by_model_protocol = _group_summaries(
        scored,
        ("model_id", "protocol"),
    )

    for group in by_model_protocol.values():
        group["coverage"] = (
            group["unique_task_count"] / len(gold_records)
        )

    summary = {
        "gold_task_count": len(gold_records),
        "prediction_count": len(scored),
        "overall": _metric_summary(scored),
        "by_family": _group_summaries(scored, ("family",)),
        "by_protocol": _group_summaries(scored, ("protocol",)),
        "by_model": _group_summaries(scored, ("model_id",)),
        "by_model_protocol": by_model_protocol,
    }

    return {
        "summary": summary,
        "rows": scored,
    }


def score_files(
    *,
    predictions_path: Path,
    gold_path: Path = DEFAULT_GOLD_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    expected_gold_hash = manifest.get("private_sha256")
    actual_gold_hash = _sha256(gold_path)

    if expected_gold_hash != actual_gold_hash:
        raise ValueError(
            "Private gold SHA-256 does not match the public manifest"
        )

    report = score_records(
        read_jsonl(gold_path),
        read_jsonl(predictions_path),
    )

    if require_complete:
        incomplete = {
            key: value["coverage"]
            for key, value in report["summary"][
                "by_model_protocol"
            ].items()
            if value["coverage"] != 1.0
        }
        if incomplete:
            raise ValueError(
                f"Incomplete model/protocol groups: {incomplete}"
            )

    report["scorer_version"] = SCORER_VERSION
    report["benchmark"] = {
        "split": manifest.get("split"),
        "seed": manifest.get("seed"),
        "generator_version": manifest.get("generator_version"),
        "public_sha256": manifest.get("public_sha256"),
        "private_sha256": actual_gold_hash,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    output_path.write_text(
        text.rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score ClaimScope pilot predictions using exact "
            "counterexample validation."
        )
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = score_files(
        predictions_path=args.predictions,
        gold_path=args.gold,
        manifest_path=args.manifest,
        output_path=args.output,
        require_complete=args.require_complete,
    )

    print(
        json.dumps(
            report["summary"],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
