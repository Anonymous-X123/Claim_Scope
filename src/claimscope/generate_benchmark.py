"""Deterministic generation of public ClaimScope tasks and private gold keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from .core import (
    FORMULA_IDS as OPTIMIZER_FORMULA_IDS,
    PROTOCOLS,
    QUANTITIES,
    audit_candidate,
)
from .tasks import TASKS as OPTIMIZER_TASKS
from .tasks import Task as OptimizerTask
from .recurrence import (
    FORMULA_IDS as RECURRENCE_FORMULA_IDS,
    TASKS as RECURRENCE_TASKS,
    audit_recurrence,
)
from .benchmark_cli import (
    RECURRENCE_CONDITION_TEXT,
    RECURRENCE_FORMULA_TEXT,
)


GENERATOR_VERSION = "0.3.0"
DEFAULT_SEED = 20260830
TASKS_PER_FAMILY = 24
TASKS_PER_VERDICT = 12
SURFACE_VARIANTS = 4

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT

DEFAULT_PUBLIC_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "pilot_public.jsonl"
)
DEFAULT_PRIVATE_PATH = (
    PROJECT_ROOT / "artifacts" / "reference" / "pilot_gold.jsonl"
)
DEFAULT_MANIFEST_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "pilot_manifest.json"
)

OPTIMIZER_CONDITIONS = tuple(
    sorted({task.condition for task in OPTIMIZER_TASKS})
)
RECURRENCE_CONDITIONS = tuple(
    sorted({task.condition for task in RECURRENCE_TASKS})
)

RESPONSE_SCHEMA = {
    "task_id": "copy the benchmark task identifier exactly",
    "verdict": "valid or invalid",
    "scope": "the strongest scope justified by the derivation",
    "failure_mode": "none or a concise diagnostic label",
    "counterexample": "null when valid; an exact witness when invalid",
    "repair": "one of the allowed repair identifiers listed in the prompt",
}

OPTIMIZER_COUNTEREXAMPLE_INSTRUCTION = (
    'For an invalid optimizer claim, counterexample must have the form '
    '{"state":{"w":number,"d":number,"v":number,"alpha":number,'
    '"eta":number,"lambda_m":number,"lambda_a":number,"beta1":number,'
    '"beta2":number,"step":integer,"epsilon":number}}. '
    'The expected, actual, and absolute_error fields are optional.'
)

RECURRENCE_COUNTEREXAMPLE_INSTRUCTION = (
    'For an invalid recurrence claim, counterexample must have the form '
    '{"state":{"rho":"p/q","x0":"p/q","inputs":["p/q",...],'
    '"horizon":integer}}. Integers and terminating decimals may replace '
    'fraction strings. The expected, actual, and reason fields are optional.'
)


def _json_line(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(path, "".join(_json_line(row) + "\n" for row in rows))


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    _atomic_write(path, text.rstrip() + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decorate_prompt(
    base: str,
    variant: int,
    counterexample_instruction: str,
) -> str:
    prefixes = (
        "Audit the following mathematical research claim.",
        (
            "Act as an independent mathematical referee. "
            "Check the proposed identity rather than trusting its derivation."
        ),
        (
            "A collaborator proposes the claim below. Determine its exact "
            "scope and test it against edge cases."
        ),
        (
            "Perform a counterexample-guided audit of the following claim. "
            "Distinguish a universal identity from accidental agreement."
        ),
    )

    suffix = (
        "\n\nReturn a JSON object with exactly these fields: task_id, verdict, "
        "scope, failure_mode, counterexample, and repair. Copy the task_id "
        "exactly. If the claim is invalid, give a concrete witness and a "
        "non-vacuous repair."
    )

    return (
        f"{prefixes[variant]}\n\n{base.strip()}{suffix}\n\n"
        f"{counterexample_instruction}"
    )


def _optimizer_failure_mode(
    formula_id: str,
    protocol: str,
    quantity: str,
) -> str:
    if formula_id == "full_transport":
        if quantity == "optimizer_direction":
            return "guarantee_scope"
        return "decay_compensation"

    mapping = {
        "direction_transport": "decay_mismatch",
        "missing_lr_ratio": "learning_rate_scaling",
        "first_step_bias": "bias_correction",
        "drop_epsilon": "epsilon_placement",
        "wrong_v_bias": "second_moment_bias",
        "decay_sign_reversed": "decay_compensation",
        "off_by_one_bias": "bias_correction",
    }
    return mapping[formula_id]


def _optimizer_repair(protocol: str, quantity: str) -> str:
    if quantity == "optimizer_direction":
        return "direction_transport"
    if protocol == "adamw_decay":
        return "full_transport"
    return "direction_transport"


def _recurrence_failure_mode(formula_id: str, condition: str) -> str:
    if formula_id == "input_off_by_one":
        return "input_exponent"
    if formula_id == "initial_off_by_one":
        return "initial_exponent"
    if formula_id == "constant_geometric":
        if condition in {"constant_input", "rho_one_constant"}:
            return "singular_case"
        return "domain_restriction"
    if formula_id == "constant_piecewise":
        return "domain_restriction"
    if formula_id == "rho_one_linear":
        return "domain_restriction"
    return "formula_mismatch"


def _recurrence_repair(formula_id: str, condition: str) -> str:
    if (
        formula_id in {"constant_geometric", "rho_one_linear"}
        and condition != "all"
    ):
        return "constant_piecewise"
    return "unroll_exact"


def _select_diverse(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(
            f"Only {len(rows)} candidates are available; {count} are required."
        )

    pool = list(rows)
    random.Random(seed).shuffle(pool)

    semantic_counts: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()
    quantity_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []

    while len(selected) < count:
        chosen = min(
            pool,
            key=lambda row: (
                semantic_counts[row["semantic"]],
                formula_counts[row["source"]["formula_id"]],
                condition_counts[row["source"]["condition"]],
                protocol_counts[row["source"].get("protocol", "")],
                quantity_counts[row["source"].get("quantity", "")],
            ),
        )

        pool.remove(chosen)
        selected.append(chosen)

        source = chosen["source"]
        semantic_counts[chosen["semantic"]] += 1
        formula_counts[source["formula_id"]] += 1
        condition_counts[source["condition"]] += 1
        protocol_counts[source.get("protocol", "")] += 1
        quantity_counts[source.get("quantity", "")] += 1

    return selected


def _optimizer_candidates() -> list[dict[str, Any]]:
    development = {
        (
            task.formula_id,
            task.protocol,
            task.quantity,
            task.condition,
        )
        for task in OPTIMIZER_TASKS
    }

    candidates: list[dict[str, Any]] = []

    for formula_id, protocol, quantity, condition in product(
        OPTIMIZER_FORMULA_IDS,
        PROTOCOLS,
        QUANTITIES,
        OPTIMIZER_CONDITIONS,
    ):
        signature = (formula_id, protocol, quantity, condition)
        if signature in development:
            continue

        audit = audit_candidate(
            formula_id,
            protocol,
            quantity,
            condition=condition,
        )

        if audit.cases_checked == 0:
            continue

        verdict = "valid" if audit.valid_on_grid else "invalid"
        failure_mode = (
            "none"
            if audit.valid_on_grid
            else _optimizer_failure_mode(formula_id, protocol, quantity)
        )
        repair = (
            "none"
            if audit.valid_on_grid
            else _optimizer_repair(protocol, quantity)
        )

        source = {
            "formula_id": formula_id,
            "protocol": protocol,
            "quantity": quantity,
            "condition": condition,
        }

        candidates.append(
            {
                "family": "optimizer_transport",
                "semantic": "|".join(signature),
                "source": source,
                "gold": {
                    "verdict": verdict,
                    "scope": quantity,
                    "failure_mode": failure_mode,
                    "repair": repair,
                    "counterexample": (
                        None
                        if audit.counterexample is None
                        else audit.counterexample.to_dict()
                    ),
                    "cases_checked": audit.cases_checked,
                },
            }
        )

    return candidates


def _recurrence_candidates() -> list[dict[str, Any]]:
    development = {
        (task.formula_id, task.condition)
        for task in RECURRENCE_TASKS
    }

    candidates: list[dict[str, Any]] = []

    for formula_id, condition in product(
        RECURRENCE_FORMULA_IDS,
        RECURRENCE_CONDITIONS,
    ):
        signature = (formula_id, condition)
        if signature in development:
            continue

        audit = audit_recurrence(formula_id, condition=condition)
        if audit.cases_checked == 0:
            continue

        verdict = "valid" if audit.valid_on_grid else "invalid"
        failure_mode = (
            "none"
            if audit.valid_on_grid
            else _recurrence_failure_mode(formula_id, condition)
        )
        repair = (
            "none"
            if audit.valid_on_grid
            else _recurrence_repair(formula_id, condition)
        )

        for variant in range(SURFACE_VARIANTS):
            source = {
                "formula_id": formula_id,
                "condition": condition,
                "surface_variant": variant,
            }

            candidates.append(
                {
                    "family": "affine_recurrence",
                    "semantic": "|".join(signature),
                    "source": source,
                    "gold": {
                        "verdict": verdict,
                        "scope": "finite_horizon_value",
                        "failure_mode": failure_mode,
                        "repair": repair,
                        "counterexample": (
                            None
                            if audit.counterexample is None
                            else audit.counterexample.to_dict()
                        ),
                        "cases_checked": audit.cases_checked,
                    },
                }
            )

    return candidates


def _select_family(
    candidates: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    valid = [
        row for row in candidates
        if row["gold"]["verdict"] == "valid"
    ]
    invalid = [
        row for row in candidates
        if row["gold"]["verdict"] == "invalid"
    ]

    selected = _select_diverse(
        valid,
        TASKS_PER_VERDICT,
        seed + 101,
    )
    selected.extend(
        _select_diverse(
            invalid,
            TASKS_PER_VERDICT,
            seed + 202,
        )
    )

    random.Random(seed + 303).shuffle(selected)
    return selected


def _optimizer_prompt(
    task_id: str,
    row: dict[str, Any],
    variant: int,
) -> str:
    source = row["source"]
    gold = row["gold"]

    task = OptimizerTask(
        task_id=task_id,
        title="Generated optimizer-transport audit",
        formula_id=source["formula_id"],
        protocol=source["protocol"],
        quantity=source["quantity"],
        condition=source["condition"],
        expected_verdict=gold["verdict"],
        expected_scope=gold["scope"],
        failure_mode=gold["failure_mode"],
        repair=gold["repair"],
        alternative_failure_modes=(),
    )

    base = task.prompt()
    old_schema_instruction = (
        "Return JSON with exactly these keys: task_id, verdict, scope, "
        "failure_mode, and repair."
    )
    if old_schema_instruction not in base:
        raise RuntimeError("Optimizer prompt schema instruction changed.")

    base = base.replace(
        old_schema_instruction,
        "Use the allowed labels and repairs listed below.",
    )
    return _decorate_prompt(
        base,
        variant,
        OPTIMIZER_COUNTEREXAMPLE_INSTRUCTION,
    )


def _recurrence_prompt(
    task_id: str,
    row: dict[str, Any],
) -> str:
    source = row["source"]
    formula = RECURRENCE_FORMULA_TEXT[source["formula_id"]]
    condition = RECURRENCE_CONDITION_TEXT[source["condition"]]

    base = f"""Benchmark item: {task_id}

Consider the scalar affine recurrence

x_(k+1) = rho*x_k + u_k, for k = 0,...,T-1.

A derivation claims the following expression for x_T:

{formula}

Permitted assumptions:
{condition}

Required label vocabulary:
- Use scope=finite_horizon_value.
- Allowed failure modes are none, input_exponent, initial_exponent,
  singular_case, domain_restriction, and formula_mismatch.
- Allowed repairs are:
  * none
  * unroll_exact, meaning
    x_T=rho^T*x_0 + sum_(k=0)^(T-1) rho^(T-1-k)*u_k
  * constant_piecewise, meaning x_T=x_0+T*u when rho=1 and
    x_T=rho^T*x_0+u*(1-rho^T)/(1-rho) otherwise.
- Use repair=none if and only if the stated claim is valid.

Decide whether the claim is valid for every state and horizon satisfying
the permitted assumptions. Check boundary and singular cases explicitly."""

    return _decorate_prompt(
        base,
        source["surface_variant"],
        RECURRENCE_COUNTEREXAMPLE_INSTRUCTION,
    )


def _materialize(
    optimizer_rows: list[dict[str, Any]],
    recurrence_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []

    variant_rng = random.Random(seed + 404)

    for index, row in enumerate(optimizer_rows, start=1):
        task_id = f"PILOT-OPT-{index:03d}"
        prompt = _optimizer_prompt(
            task_id,
            row,
            variant_rng.randrange(SURFACE_VARIANTS),
        )

        public.append(
            {
                "task_id": task_id,
                "family": row["family"],
                "split": "pilot",
                "prompt": prompt,
                "response_schema": RESPONSE_SCHEMA,
            }
        )
        private.append(
            {
                "task_id": task_id,
                "family": row["family"],
                "source": row["source"],
                "gold": row["gold"],
            }
        )

    for index, row in enumerate(recurrence_rows, start=1):
        task_id = f"PILOT-REC-{index:03d}"
        prompt = _recurrence_prompt(task_id, row)

        public.append(
            {
                "task_id": task_id,
                "family": row["family"],
                "split": "pilot",
                "prompt": prompt,
                "response_schema": RESPONSE_SCHEMA,
            }
        )
        private.append(
            {
                "task_id": task_id,
                "family": row["family"],
                "source": row["source"],
                "gold": row["gold"],
            }
        )

    return public, private


def generate_benchmark(
    *,
    seed: int = DEFAULT_SEED,
    public_path: Path = DEFAULT_PUBLIC_PATH,
    private_path: Path = DEFAULT_PRIVATE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = (public_path, private_path, manifest_path)

    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing benchmark files: "
                + ", ".join(existing)
            )

    optimizer_rows = _select_family(
        _optimizer_candidates(),
        seed + 1000,
    )
    recurrence_rows = _select_family(
        _recurrence_candidates(),
        seed + 2000,
    )

    public, private = _materialize(
        optimizer_rows,
        recurrence_rows,
        seed,
    )

    _write_jsonl(public_path, public)
    _write_jsonl(private_path, private)

    distribution = {
        family: {
            verdict: sum(
                1
                for row in private
                if row["family"] == family
                and row["gold"]["verdict"] == verdict
            )
            for verdict in ("valid", "invalid")
        }
        for family in ("optimizer_transport", "affine_recurrence")
    }

    task_id_digest = hashlib.sha256(
        "\n".join(row["task_id"] for row in public).encode("utf-8")
    ).hexdigest()

    manifest = {
        "generator": "claimscope.generate_benchmark",
        "generator_version": GENERATOR_VERSION,
        "split": "pilot",
        "seed": seed,
        "task_count": len(public),
        "tasks_per_family": TASKS_PER_FAMILY,
        "development_signatures_excluded": True,
        "distribution": distribution,
        "task_id_sha256": task_id_digest,
        "public_sha256": _sha256(public_path),
        "private_sha256": _sha256(private_path),
    }

    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic ClaimScope pilot benchmark."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--public",
        type=Path,
        default=DEFAULT_PUBLIC_PATH,
    )
    parser.add_argument(
        "--private",
        type=Path,
        default=DEFAULT_PRIVATE_PATH,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = generate_benchmark(
            seed=args.seed,
            public_path=args.public,
            private_path=args.private,
            manifest_path=args.manifest,
            overwrite=args.force,
        )
    except FileExistsError as error:
        parser.error(str(error))

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
