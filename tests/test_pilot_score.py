import copy
import json
import tempfile
import unittest
from pathlib import Path

from claimscope.generate_benchmark import generate_benchmark
from claimscope.pilot_score import (
    read_jsonl,
    score_files,
    score_prediction,
    score_records,
    validate_counterexample,
)


class PilotScorerTests(unittest.TestCase):
    def benchmark_in(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path, list[dict[str, object]]]:
        public = root / "public.jsonl"
        private = root / "private.jsonl"
        manifest = root / "manifest.json"

        generate_benchmark(
            public_path=public,
            private_path=private,
            manifest_path=manifest,
            overwrite=True,
        )
        return public, private, manifest, read_jsonl(private)

    def oracle_prediction(
        self,
        gold_record: dict[str, object],
        *,
        failure_mode: str | None = None,
    ) -> dict[str, object]:
        gold = gold_record["gold"]

        prediction = {
            "task_id": gold_record["task_id"],
            "verdict": gold["verdict"],
            "scope": gold["scope"],
            "failure_mode": (
                gold["failure_mode"]
                if failure_mode is None
                else failure_mode
            ),
            "counterexample": gold["counterexample"],
            "repair": gold["repair"],
        }

        return {
            "task_id": gold_record["task_id"],
            "model_id": "oracle",
            "protocol": "oracle",
            "prediction": prediction,
        }

    def test_oracle_predictions_reach_safe_completion_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, gold = self.benchmark_in(Path(directory))
            predictions = [
                self.oracle_prediction(record)
                for record in gold
            ]

            report = score_records(gold, predictions)
            overall = report["summary"]["overall"]

            self.assertEqual(overall["safe_completion"], 1.0)
            self.assertEqual(overall["strict_completion"], 1.0)
            self.assertEqual(
                overall["counterexample_valid"],
                1.0,
            )
            self.assertEqual(
                report["summary"]["by_model_protocol"][
                    "oracle|oracle"
                ]["coverage"],
                1.0,
            )

    def test_failure_label_is_diagnostic_not_primary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, gold = self.benchmark_in(Path(directory))
            invalid = next(
                record
                for record in gold
                if record["gold"]["verdict"] == "invalid"
            )

            prediction = self.oracle_prediction(
                invalid,
                failure_mode="plausible_alternative_label",
            )
            scored = score_prediction(invalid, prediction)

            self.assertEqual(scored["safe_completion"], 1)
            self.assertEqual(
                scored["failure_mode_correct"],
                0,
            )
            self.assertEqual(scored["strict_completion"], 0)

    def test_invalid_or_non_null_witness_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, gold = self.benchmark_in(Path(directory))

            invalid = next(
                record
                for record in gold
                if record["gold"]["verdict"] == "invalid"
            )
            bad_invalid = self.oracle_prediction(invalid)
            bad_invalid["prediction"]["counterexample"] = {
                "state": {}
            }

            invalid_score = score_prediction(
                invalid,
                bad_invalid,
            )
            self.assertEqual(
                invalid_score["counterexample_valid"],
                0,
            )
            self.assertEqual(
                invalid_score["safe_completion"],
                0,
            )

            valid = next(
                record
                for record in gold
                if record["gold"]["verdict"] == "valid"
            )
            bad_valid = self.oracle_prediction(valid)
            bad_valid["prediction"]["counterexample"] = {
                "state": {}
            }

            valid_score = score_prediction(valid, bad_valid)
            self.assertEqual(
                valid_score["counterexample_valid"],
                0,
            )
            self.assertEqual(
                valid_score["safe_completion"],
                0,
            )

    def test_linear_algebra_counterexample_is_verified(self) -> None:
        source = {
            "formula_id": "transpose_same_order",
            "condition": "all",
        }
        counterexample = {
            "state": {
                "A": [["1", "1"], ["0", "1"]],
                "B": [["1", "0"], ["1", "1"]],
            }
        }

        valid, reason = validate_counterexample(
            "linear_algebra",
            source,
            counterexample,
        )

        self.assertTrue(valid)
        self.assertEqual(
            reason,
            "Counterexample verified by the linear algebra oracle.",
        )

    def test_runner_envelope_routes_misspelled_reported_task_id(self) -> None:
        gold = {
            "task_id": "T001",
            "family": "affine_recurrence",
            "source": {},
            "gold": {
                "verdict": "valid",
                "scope": "finite_horizon_value",
                "failure_mode": "none",
                "counterexample": None,
                "repair": "none",
            },
        }
        prediction = {
            "task_id": "T001",
            "model_id": "fake/model",
            "protocol": "generic_feedback",
            "prediction": {
                "task_id": "TOO1",
                "verdict": "valid",
                "scope": "finite_horizon_value",
                "failure_mode": "none",
                "counterexample": None,
                "repair": "none",
            },
        }

        report = score_records([gold], [prediction])
        row = report["rows"][0]

        self.assertEqual(row["task_id"], "T001")
        self.assertEqual(row["task_id_correct"], 0)
        self.assertEqual(row["json_valid"], 0)
        self.assertEqual(row["safe_completion"], 0)
        self.assertEqual(row["verdict_correct"], 1)

    def test_bare_unknown_task_id_is_still_rejected(self) -> None:
        gold = {
            "task_id": "T001",
            "family": "affine_recurrence",
            "source": {},
            "gold": {
                "verdict": "valid",
                "scope": "finite_horizon_value",
                "failure_mode": "none",
                "counterexample": None,
                "repair": "none",
            },
        }

        with self.assertRaisesRegex(ValueError, "Unknown prediction task_id"):
            score_records(
                [gold],
                [
                    {
                        "task_id": "UNKNOWN",
                        "verdict": "valid",
                        "scope": "finite_horizon_value",
                        "failure_mode": "none",
                        "counterexample": None,
                        "repair": "none",
                    }
                ],
            )

    def test_linear_algebra_rejects_non_matrix_witness(self) -> None:
        valid, reason = validate_counterexample(
            "linear_algebra",
            {
                "formula_id": "transpose_same_order",
                "condition": "all",
            },
            {"state": {"A": [[1]], "B": [[1, 0], [0, 1]]}},
        )

        self.assertFalse(valid)
        self.assertIn("A must be a 2x2 JSON list", reason)

    def test_file_scoring_verifies_hash_and_omits_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, private, manifest, gold = self.benchmark_in(root)

            predictions = root / "predictions.jsonl"
            output = root / "report.json"

            rows = [
                self.oracle_prediction(record)
                for record in gold
            ]
            predictions.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

            report = score_files(
                predictions_path=predictions,
                gold_path=private,
                manifest_path=manifest,
                output_path=output,
                require_complete=True,
            )

            self.assertEqual(
                report["summary"]["overall"][
                    "safe_completion"
                ],
                1.0,
            )
            self.assertTrue(output.exists())
            self.assertNotIn(b"\r\n", output.read_bytes())

            stored = json.loads(
                output.read_text(encoding="utf-8")
            )
            self.assertEqual(
                stored["benchmark"]["private_sha256"],
                json.loads(
                    manifest.read_text(encoding="utf-8")
                )["private_sha256"],
            )

            for row in stored["rows"]:
                self.assertNotIn("gold", row)
                self.assertNotIn("source", row)


if __name__ == "__main__":
    unittest.main()
