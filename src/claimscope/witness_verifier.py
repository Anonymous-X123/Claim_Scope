"""Label-blind exact witness verifier for tool-augmented ClaimScope runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pilot_score import validate_counterexample


VERIFIER_VERSION = "0.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


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


@dataclass(frozen=True)
class VerifierEntry:
    family: str
    source: dict[str, Any]


class WitnessVerifier:
    """Checks a submitted witness without exposing benchmark gold labels."""

    def __init__(
        self,
        entries: dict[str, VerifierEntry],
        private_key_sha256: str,
    ) -> None:
        self._entries = dict(entries)
        self._private_key_sha256 = private_key_sha256

    def metadata(self) -> dict[str, object]:
        return {
            "version": VERIFIER_VERSION,
            "private_key_sha256": self._private_key_sha256,
            "task_count": len(self._entries),
            "feedback_contract": (
                "witness_validity_only_no_gold_verdict_repair_or_answer_key"
            ),
        }

    def feedback(
        self,
        task: dict[str, Any],
        prediction: dict[str, Any] | None,
        parse_error: str | None,
    ) -> dict[str, str]:
        task_id = str(task.get("task_id", ""))
        entry = self._entries.get(task_id)

        if entry is None:
            raise ValueError(
                f"No verifier entry is available for task: {task_id}"
            )

        if task.get("family") != entry.family:
            raise ValueError(
                f"Public family disagrees with verifier entry: {task_id}"
            )

        if prediction is None:
            detail = (
                "The draft contained no machine-readable JSON object."
                if parse_error is None
                else parse_error
            )
            return {
                "status": "unavailable",
                "message": (
                    "No counterexample was checked because the draft could "
                    f"not be parsed. {detail} The verifier does not reveal "
                    "a correct verdict, repair, or answer key."
                ),
            }

        verdict = prediction.get("verdict")
        if not isinstance(verdict, str) or verdict.strip().lower() != "invalid":
            return {
                "status": "not_applicable",
                "message": (
                    "The verifier checks only a counterexample submitted "
                    "with an invalid verdict. No witness was checked, and "
                    "the verifier does not reveal a correct verdict, repair, "
                    "or answer key."
                ),
            }

        counterexample = prediction.get("counterexample")
        if counterexample is None:
            return {
                "status": "rejected",
                "message": (
                    "An invalid verdict was submitted without a "
                    "counterexample, so no witness was verified. The "
                    "verifier does not reveal a correct verdict, repair, "
                    "or answer key."
                ),
            }

        valid, reason = validate_counterexample(
            entry.family,
            entry.source,
            counterexample,
        )

        if valid:
            return {
                "status": "verified",
                "message": (
                    "The submitted counterexample was verified to refute "
                    "the stated claim. This checks the witness only; it "
                    "does not assess the remaining response."
                ),
            }

        return {
            "status": "rejected",
            "message": (
                "The submitted counterexample was not verified. "
                f"{reason} The verifier does not reveal a correct verdict, "
                "repair, or answer key."
            ),
        }


def load_witness_verifier(
    public_tasks: list[dict[str, Any]],
    private_key_path: Path,
    manifest_path: Path,
) -> WitnessVerifier:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha256 = manifest.get("private_sha256")
    actual_sha256 = _sha256(private_key_path)

    if expected_sha256 != actual_sha256:
        raise ValueError(
            "Private verifier key SHA-256 does not match the manifest"
        )

    private_rows = _read_jsonl(private_key_path)
    private_by_id: dict[str, dict[str, Any]] = {}

    for row in private_rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Private verifier key has an invalid task ID")
        if task_id in private_by_id:
            raise ValueError(
                f"Duplicate private verifier task ID: {task_id}"
            )
        private_by_id[task_id] = row

    entries: dict[str, VerifierEntry] = {}

    for task in public_tasks:
        task_id = str(task.get("task_id", ""))
        row = private_by_id.get(task_id)

        if row is None:
            raise ValueError(
                f"Private verifier key is missing task: {task_id}"
            )

        family = row.get("family")
        source = row.get("source")

        if family != task.get("family"):
            raise ValueError(
                f"Private verifier family mismatch: {task_id}"
            )
        if not isinstance(source, dict):
            raise ValueError(
                f"Private verifier source is invalid: {task_id}"
            )

        # Deliberately retain source only. Gold labels are discarded here.
        entries[task_id] = VerifierEntry(
            family=family,
            source=dict(source),
        )

    return WitnessVerifier(entries, actual_sha256)
