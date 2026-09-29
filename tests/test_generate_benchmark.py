import json
import tempfile
import unittest
from pathlib import Path

from claimscope.core import audit_candidate
from claimscope.generate_benchmark import (
    DEFAULT_SEED,
    generate_benchmark,
)
from claimscope.recurrence import audit_recurrence
from claimscope.recurrence import TASKS as RECURRENCE_TASKS
from claimscope.tasks import TASKS as OPTIMIZER_TASKS


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class BenchmarkGenerationTests(unittest.TestCase):
    def generate_in(
        self,
        root: Path,
    ) -> tuple[
        Path,
        Path,
        Path,
        dict[str, object],
    ]:
        public = root / "public.jsonl"
        private = root / "private.jsonl"
        manifest = root / "manifest.json"

        result = generate_benchmark(
            seed=DEFAULT_SEED,
            public_path=public,
            private_path=private,
            manifest_path=manifest,
            overwrite=True,
        )
        return public, private, manifest, result

    def test_balanced_48_task_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            public_path, private_path, manifest_path, manifest = (
                self.generate_in(Path(directory))
            )

            public = read_jsonl(public_path)
            private = read_jsonl(private_path)

            self.assertEqual(len(public), 48)
            self.assertEqual(len(private), 48)
            self.assertEqual(manifest["task_count"], 48)

            public_ids = [row["task_id"] for row in public]
            private_ids = [row["task_id"] for row in private]

            self.assertEqual(public_ids, private_ids)
            self.assertEqual(len(set(public_ids)), 48)

            expected_distribution = {
                "optimizer_transport": {
                    "valid": 12,
                    "invalid": 12,
                },
                "affine_recurrence": {
                    "valid": 12,
                    "invalid": 12,
                },
            }
            self.assertEqual(
                manifest["distribution"],
                expected_distribution,
            )

            stored_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(stored_manifest, manifest)

    def test_public_split_contains_no_answer_key(self) -> None:
        forbidden = {
            "source",
            "gold",
            "formula_id",
            "protocol",
            "quantity",
            "condition",
            "expected_verdict",
            "failure_mode",
            "repair",
            "counterexample",
            "cases_checked",
        }

        with tempfile.TemporaryDirectory() as directory:
            public_path, private_path, _, _ = self.generate_in(
                Path(directory)
            )
            public = read_jsonl(public_path)
            private = read_jsonl(private_path)

            self.assertTrue(private)
            self.assertTrue(all("gold" in row for row in private))

            for row in public:
                self.assertTrue(forbidden.isdisjoint(row))
                self.assertEqual(row["split"], "pilot")
                self.assertIn("prompt", row)
                self.assertIn("response_schema", row)

    def test_generation_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.generate_in(root / "first")
            second = self.generate_in(root / "second")

            for first_path, second_path in zip(
                first[:3],
                second[:3],
            ):
                self.assertEqual(
                    first_path.read_bytes(),
                    second_path.read_bytes(),
                )

            self.assertEqual(first[3], second[3])

    def test_gold_labels_recompute_and_development_is_excluded(
        self,
    ) -> None:
        optimizer_development = {
            (
                task.formula_id,
                task.protocol,
                task.quantity,
                task.condition,
            )
            for task in OPTIMIZER_TASKS
        }
        recurrence_development = {
            (task.formula_id, task.condition)
            for task in RECURRENCE_TASKS
        }

        with tempfile.TemporaryDirectory() as directory:
            _, private_path, _, _ = self.generate_in(Path(directory))
            private = read_jsonl(private_path)

            for row in private:
                source = row["source"]
                gold = row["gold"]

                if row["family"] == "optimizer_transport":
                    signature = (
                        source["formula_id"],
                        source["protocol"],
                        source["quantity"],
                        source["condition"],
                    )
                    self.assertNotIn(
                        signature,
                        optimizer_development,
                    )

                    audit = audit_candidate(
                        source["formula_id"],
                        source["protocol"],
                        source["quantity"],
                        condition=source["condition"],
                    )
                else:
                    signature = (
                        source["formula_id"],
                        source["condition"],
                    )
                    self.assertNotIn(
                        signature,
                        recurrence_development,
                    )

                    audit = audit_recurrence(
                        source["formula_id"],
                        condition=source["condition"],
                    )

                recomputed_verdict = (
                    "valid" if audit.valid_on_grid else "invalid"
                )
                self.assertEqual(
                    gold["verdict"],
                    recomputed_verdict,
                )
                self.assertEqual(
                    gold["cases_checked"],
                    audit.cases_checked,
                )

                if recomputed_verdict == "valid":
                    self.assertIsNone(gold["counterexample"])
                    self.assertEqual(gold["repair"], "none")
                else:
                    self.assertIsNotNone(gold["counterexample"])
                    self.assertNotEqual(gold["repair"], "none")


    def test_prompts_have_one_consistent_response_schema(self) -> None:
        required_instruction = (
            "Return a JSON object with exactly these fields: task_id, verdict, "
            "scope, failure_mode, counterexample, and repair."
        )
        obsolete_instruction = (
            "Return JSON with exactly these keys: task_id, verdict, scope, "
            "failure_mode, and repair."
        )

        with tempfile.TemporaryDirectory() as directory:
            public_path, _, _, _ = self.generate_in(Path(directory))
            public = read_jsonl(public_path)

            for row in public:
                prompt = row["prompt"]
                self.assertEqual(prompt.count(required_instruction), 1)
                self.assertNotIn(obsolete_instruction, prompt)
                self.assertIn("task_id", row["response_schema"])


    def test_prompts_publish_machine_checkable_witness_schema(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            public_path, _, _, _ = self.generate_in(Path(directory))
            public = read_jsonl(public_path)

            for row in public:
                prompt = row["prompt"]

                if row["family"] == "optimizer_transport":
                    self.assertIn('"w":number', prompt)
                    self.assertIn('"epsilon":number', prompt)
                    self.assertIn('"step":integer', prompt)
                else:
                    self.assertIn('"rho":"p/q"', prompt)
                    self.assertIn('"inputs":["p/q",...]', prompt)
                    self.assertIn('"horizon":integer', prompt)


    def test_recurrence_prompts_publish_label_vocabulary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            public_path, _, _, _ = self.generate_in(Path(directory))
            public = read_jsonl(public_path)

            recurrence = [
                row
                for row in public
                if row["family"] == "affine_recurrence"
            ]
            self.assertEqual(len(recurrence), 24)

            for row in recurrence:
                prompt = row["prompt"]
                self.assertIn(
                    "scope=finite_horizon_value",
                    prompt,
                )
                self.assertIn(
                    "Allowed failure modes",
                    prompt,
                )
                self.assertIn(
                    "unroll_exact",
                    prompt,
                )
                self.assertIn(
                    "constant_piecewise",
                    prompt,
                )
                self.assertIn(
                    "Use repair=none if and only if",
                    prompt,
                )


if __name__ == "__main__":
    unittest.main()
