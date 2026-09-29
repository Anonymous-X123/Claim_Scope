"""Verify the frozen ClaimScope release and its reported scores."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimscope.pilot_score import SCORER_VERSION, score_records  # noqa: E402


PUBLIC = ROOT / "artifacts" / "benchmark" / "extended_v1_public.jsonl"
GOLD = ROOT / "artifacts" / "reference" / "extended_v1_gold.jsonl"
MANIFEST = ROOT / "artifacts" / "benchmark" / "extended_v1_manifest.json"

RUNS = {
    "direct": ROOT / "results" / "runs" / "direct.jsonl",
    "prompt_controls": ROOT / "results" / "runs" / "prompt_controls.jsonl",
    "generic_feedback": ROOT / "results" / "runs" / "generic_feedback.jsonl",
    "verifier_feedback": ROOT / "results" / "runs" / "verifier_feedback.jsonl",
}

SCORES = {
    name: ROOT / "results" / "scores" / f"{name}.json"
    for name in RUNS
}

EXPECTED_PROTOCOL_COUNTS = {
    "direct": Counter({"direct": 72}),
    "prompt_controls": Counter(
        {"self_reflect": 72, "counterexample_guided": 72}
    ),
    "generic_feedback": Counter({"generic_feedback": 72}),
    "verifier_feedback": Counter({"verifier_feedback": 72}),
}

FORBIDDEN_PUBLIC_FIELDS = {
    "source",
    "gold",
    "answer",
    "answer_key",
    "expected_verdict",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AssertionError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def oracle_prediction(row: dict[str, Any]) -> dict[str, Any]:
    prediction = {"task_id": row["task_id"], **row["gold"]}
    return {
        "task_id": row["task_id"],
        "model_id": "reference-oracle",
        "protocol": "reference-oracle",
        "prediction": prediction,
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    public = read_jsonl(PUBLIC)
    gold = read_jsonl(GOLD)

    assert sha256(PUBLIC) == manifest["public_sha256"]
    assert sha256(GOLD) == manifest["private_sha256"]
    assert len(public) == len(gold) == manifest["task_count"] == 72

    public_ids = [row["task_id"] for row in public]
    gold_ids = [row["task_id"] for row in gold]
    assert public_ids == gold_ids
    assert len(set(public_ids)) == 72
    assert all(not FORBIDDEN_PUBLIC_FIELDS.intersection(row) for row in public)

    distribution: dict[str, Counter[str]] = {}
    for row in gold:
        distribution.setdefault(row["family"], Counter())[row["gold"]["verdict"]] += 1
    assert {
        family: dict(counts)
        for family, counts in distribution.items()
    } == manifest["distribution"]

    oracle_report = score_records(
        gold,
        [oracle_prediction(row) for row in gold],
    )
    oracle_metrics = oracle_report["summary"]["overall"]
    assert oracle_metrics["safe_completion"] == 1.0
    assert oracle_metrics["strict_completion"] == 1.0
    assert oracle_metrics["counterexample_valid"] == 1.0

    verified_runs: dict[str, dict[str, int]] = {}
    for name, run_path in RUNS.items():
        run_rows = read_jsonl(run_path)
        protocol_counts = Counter(str(row.get("protocol")) for row in run_rows)
        assert protocol_counts == EXPECTED_PROTOCOL_COUNTS[name]

        recomputed = score_records(gold, run_rows)
        stored = json.loads(SCORES[name].read_text(encoding="utf-8"))
        assert stored["scorer_version"] == SCORER_VERSION
        assert recomputed["rows"] == stored["rows"]
        assert recomputed["summary"] == stored["summary"]
        verified_runs[name] = dict(protocol_counts)

    result = {
        "status": "ok",
        "task_count": len(public),
        "public_sha256": sha256(PUBLIC),
        "reference_sha256": sha256(GOLD),
        "oracle_safe_completion": oracle_metrics["safe_completion"],
        "verified_runs": verified_runs,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
