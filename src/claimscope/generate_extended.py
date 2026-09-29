"""Generate a hash-bound ClaimScope extension with linear algebra."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .generate_benchmark import RESPONSE_SCHEMA
from .linear_algebra import (
    CONDITION_TEXT,
    FORMULA_SPECS,
    FORMULA_TEXT,
)
from .linear_algebra import (
    find_counterexample as find_linear_algebra_counterexample,
)


GENERATOR_VERSION = "0.1.0"
DEFAULT_SEED = 20260902
BASE_TASK_COUNT = 48
LINEAR_ALGEBRA_TASK_COUNT = 24
LINEAR_ALGEBRA_SURFACE_VARIANTS = 3

FROZEN_PILOT_PUBLIC_SHA256 = (
    "a522de6ca1b9190e10c13f93d7e0fd2a18abcc33164d65dc1ae01c643a5ad42e"
)
FROZEN_PILOT_PRIVATE_SHA256 = (
    "5f4a2006649be14773cf9060dac8930a65a2dedb7ca17600d9d5240d1309f6d6"
)

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT

DEFAULT_BASE_PUBLIC_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "pilot_public.jsonl"
)
DEFAULT_BASE_PRIVATE_PATH = (
    PROJECT_ROOT / "artifacts" / "reference" / "pilot_gold.jsonl"
)
DEFAULT_PUBLIC_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "extended_v1_public.jsonl"
)
DEFAULT_PRIVATE_PATH = (
    PROJECT_ROOT / "artifacts" / "reference" / "extended_v1_gold.jsonl"
)
DEFAULT_MANIFEST_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "extended_v1_manifest.json"
)

FORBIDDEN_PUBLIC_FIELDS = {
    "source",
    "gold",
    "answer",
    "answer_key",
    "expected_verdict",
}

COUNTEREXAMPLE_INSTRUCTION = (
    'For an invalid linear-algebra claim, counterexample must have the form '
    '{"state":{"A":[["p/q","p/q"],["p/q","p/q"]],'
    '"B":[["p/q","p/q"],["p/q","p/q"]]}}. Integers and terminating '
    'decimals may replace fraction strings. Expected and actual are optional.'
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
    _atomic_write(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).rstrip()
        + "\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_frozen_pilot(
    public_path: Path,
    private_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_hash = _sha256(public_path)
    private_hash = _sha256(private_path)

    if public_hash != FROZEN_PILOT_PUBLIC_SHA256:
        raise ValueError(
            "The base public file is not the frozen ClaimScope pilot: "
            f"{public_hash}"
        )
    if private_hash != FROZEN_PILOT_PRIVATE_SHA256:
        raise ValueError(
            "The base private file is not the frozen ClaimScope pilot: "
            f"{private_hash}"
        )

    public = _read_jsonl(public_path)
    private = _read_jsonl(private_path)

    if len(public) != BASE_TASK_COUNT or len(private) != BASE_TASK_COUNT:
        raise ValueError("The frozen pilot must contain exactly 48 tasks")

    public_ids = [row.get("task_id") for row in public]
    private_ids = [row.get("task_id") for row in private]

    if public_ids != private_ids or len(set(public_ids)) != BASE_TASK_COUNT:
        raise ValueError("Frozen public and private task IDs disagree")

    for row in public:
        if FORBIDDEN_PUBLIC_FIELDS.intersection(row):
            raise ValueError("The frozen public pilot exposes an answer field")

    return public, private


def _prompt(
    task_id: str,
    formula_id: str,
    condition: str,
    variant: int,
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
    )

    base = f"""Task ID: {task_id}

Let A and B be 2x2 matrices with rational entries.

Candidate identity: {FORMULA_TEXT[formula_id]}
Assumptions: {CONDITION_TEXT[condition]}
Claim: The candidate identity holds for every pair of matrices satisfying the assumptions.

Allowed verdicts: valid, invalid.
Allowed scope: matrix_identity.
Allowed failure modes: none, transpose_order, determinant_nonlinearity, inverse_order, inverse_additivity.
Allowed repair identifiers: none, reverse_transpose_order, expand_2x2_determinant, reverse_inverse_order, drop_inverse_additivity.

Use verdict=valid, failure_mode=none, counterexample=null, and repair=none if and only if the universal claim is valid. For an invalid claim, provide one exact rational 2x2 witness and a non-vacuous repair."""

    suffix = (
        "\n\nReturn a JSON object with exactly these fields: task_id, verdict, "
        "scope, failure_mode, counterexample, and repair. Copy task_id exactly."
    )
    return f"{prefixes[variant]}\n\n{base.strip()}{suffix}\n\n{COUNTEREXAMPLE_INSTRUCTION}"


def _linear_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    index = 0

    for variant in range(LINEAR_ALGEBRA_SURFACE_VARIANTS):
        for spec in FORMULA_SPECS:
            index += 1
            task_id = f"EXT-LA-{index:03d}"
            witness = find_linear_algebra_counterexample(
                spec.formula_id,
                condition=spec.condition,
            )

            if spec.verdict == "valid":
                if witness is not None:
                    raise RuntimeError(
                        f"Valid formula unexpectedly has a witness: "
                        f"{spec.formula_id}"
                    )
                counterexample = None
            else:
                if witness is None:
                    raise RuntimeError(
                        f"Invalid formula lacks a witness: "
                        f"{spec.formula_id}"
                    )
                counterexample = witness.to_dict()

            public.append(
                {
                    "task_id": task_id,
                    "family": "linear_algebra",
                    "split": "extended_v1",
                    "prompt": _prompt(
                        task_id,
                        spec.formula_id,
                        spec.condition,
                        variant,
                    ),
                    "response_schema": RESPONSE_SCHEMA,
                }
            )
            private.append(
                {
                    "task_id": task_id,
                    "family": "linear_algebra",
                    "source": {
                        "formula_id": spec.formula_id,
                        "condition": spec.condition,
                        "semantic_id": spec.formula_id,
                        "surface_variant": variant,
                    },
                    "gold": {
                        "verdict": spec.verdict,
                        "scope": "matrix_identity",
                        "failure_mode": spec.failure_mode,
                        "counterexample": counterexample,
                        "repair": spec.repair,
                    },
                }
            )

    if len(public) != LINEAR_ALGEBRA_TASK_COUNT:
        raise RuntimeError("Unexpected linear algebra task count")
    return public, private


def _distribution(
    private: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    families = (
        "optimizer_transport",
        "affine_recurrence",
        "linear_algebra",
    )
    return {
        family: {
            verdict: sum(
                1
                for row in private
                if row["family"] == family
                and row["gold"]["verdict"] == verdict
            )
            for verdict in ("valid", "invalid")
        }
        for family in families
    }


def generate_extended_benchmark(
    *,
    seed: int = DEFAULT_SEED,
    base_public_path: Path = DEFAULT_BASE_PUBLIC_PATH,
    base_private_path: Path = DEFAULT_BASE_PRIVATE_PATH,
    public_path: Path = DEFAULT_PUBLIC_PATH,
    private_path: Path = DEFAULT_PRIVATE_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_paths = (public_path, private_path, manifest_path)

    if not overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing extension files: "
                + ", ".join(existing)
            )

    base_public, base_private = _verify_frozen_pilot(
        base_public_path,
        base_private_path,
    )
    linear_public, linear_private = _linear_rows()

    public = [*base_public, *linear_public]
    private = [*base_private, *linear_private]
    public_ids = [row["task_id"] for row in public]
    private_ids = [row["task_id"] for row in private]

    if len(public) != BASE_TASK_COUNT + LINEAR_ALGEBRA_TASK_COUNT:
        raise RuntimeError("Unexpected extended task count")
    if public_ids != private_ids or len(set(public_ids)) != len(public):
        raise RuntimeError("Extended public and private task IDs disagree")

    _write_jsonl(public_path, public)
    _write_jsonl(private_path, private)

    manifest = {
        "generator": "claimscope.generate_extended",
        "generator_version": GENERATOR_VERSION,
        "split": "extended_v1",
        "seed": seed,
        "task_count": len(public),
        "base_pilot_task_count": BASE_TASK_COUNT,
        "linear_algebra_task_count": LINEAR_ALGEBRA_TASK_COUNT,
        "linear_algebra_unique_specification_count": len(FORMULA_SPECS),
        "linear_algebra_surface_variants_per_spec": (
            LINEAR_ALGEBRA_SURFACE_VARIANTS
        ),
        "base_pilot_public_sha256": FROZEN_PILOT_PUBLIC_SHA256,
        "base_pilot_private_sha256": FROZEN_PILOT_PRIVATE_SHA256,
        "distribution": _distribution(private),
        "task_id_sha256": hashlib.sha256(
            "\n".join(public_ids).encode("utf-8")
        ).hexdigest(),
        "public_sha256": _sha256(public_path),
        "private_sha256": _sha256(private_path),
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the hash-bound ClaimScope extended_v1 benchmark."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-public", type=Path, default=DEFAULT_BASE_PUBLIC_PATH)
    parser.add_argument("--base-private", type=Path, default=DEFAULT_BASE_PRIVATE_PATH)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC_PATH)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_extended_benchmark(
            seed=args.seed,
            base_public_path=args.base_public,
            base_private_path=args.base_private,
            public_path=args.public,
            private_path=args.private,
            manifest_path=args.manifest,
            overwrite=args.force,
        )
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
