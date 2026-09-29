import json
import tempfile
import unittest
from pathlib import Path

from claimscope.generate_benchmark import generate_benchmark
from claimscope.generate_extended import generate_extended_benchmark
from claimscope.linear_algebra import (
    SPEC_BY_ID,
    find_counterexample as find_linear_algebra_counterexample,
)
from claimscope.pilot_score import score_records


class ExtendedBenchmarkGenerationTests(unittest.TestCase):
    def generate_in(
        self,
        root: Path,
    ) -> tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        base_public = root / "base_public.jsonl"
        base_private = root / "base_private.jsonl"
        base_manifest = root / "base_manifest.json"

        generate_benchmark(
            public_path=base_public,
            private_path=base_private,
            manifest_path=base_manifest,
            overwrite=True,
        )

        public = root / "extended_public.jsonl"
        private = root / "extended_private.jsonl"
        manifest = root / "extended_manifest.json"

        generate_extended_benchmark(
            base_public_path=base_public,
            base_private_path=base_private,
            public_path=public,
            private_path=private,
            manifest_path=manifest,
            overwrite=True,
        )

        public_rows = [
            json.loads(line)
            for line in public.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        private_rows = [
            json.loads(line)
            for line in private.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        return (
            base_public,
            base_private,
            public,
            private,
            manifest,
            public_rows,
            private_rows,
        )

    def test_extended_split_has_balanced_three_family_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                _,
                _,
                _,
                _,
                manifest_path,
                public_rows,
                private_rows,
            ) = self.generate_in(Path(directory))

            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            self.assertEqual(len(public_rows), 72)
            self.assertEqual(len(private_rows), 72)
            self.assertEqual(
                manifest["linear_algebra_unique_specification_count"],
                8,
            )
            self.assertEqual(
                manifest["linear_algebra_surface_variants_per_spec"],
                3,
            )

            for family in (
                "optimizer_transport",
                "affine_recurrence",
                "linear_algebra",
            ):
                self.assertEqual(
                    manifest["distribution"][family]["valid"],
                    12,
                )
                self.assertEqual(
                    manifest["distribution"][family]["invalid"],
                    12,
                )

            for row in public_rows:
                self.assertNotIn("source", row)
                self.assertNotIn("gold", row)

    def test_generation_preserves_base_and_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                base_public,
                base_private,
                public_one,
                private_one,
                manifest_one,
                _,
                _,
            ) = self.generate_in(root / "one")

            base_public_before = base_public.read_bytes()
            base_private_before = base_private.read_bytes()

            public_two = root / "two" / "extended_public.jsonl"
            private_two = root / "two" / "extended_private.jsonl"
            manifest_two = root / "two" / "extended_manifest.json"

            generate_extended_benchmark(
                base_public_path=base_public,
                base_private_path=base_private,
                public_path=public_two,
                private_path=private_two,
                manifest_path=manifest_two,
                overwrite=True,
            )

            self.assertEqual(base_public.read_bytes(), base_public_before)
            self.assertEqual(base_private.read_bytes(), base_private_before)
            self.assertEqual(
                public_one.read_bytes(),
                public_two.read_bytes(),
            )
            self.assertEqual(
                private_one.read_bytes(),
                private_two.read_bytes(),
            )
            self.assertEqual(
                manifest_one.read_bytes(),
                manifest_two.read_bytes(),
            )

    def test_linear_algebra_gold_recomputes_and_scores_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                _,
                _,
                _,
                _,
                _,
                _,
                private_rows,
            ) = self.generate_in(Path(directory))

            linear_rows = [
                row
                for row in private_rows
                if row["family"] == "linear_algebra"
            ]
            self.assertEqual(len(linear_rows), 24)

            for row in linear_rows:
                source = row["source"]
                gold = row["gold"]
                spec = SPEC_BY_ID[source["formula_id"]]

                self.assertEqual(gold["verdict"], spec.verdict)
                self.assertEqual(
                    gold["failure_mode"],
                    spec.failure_mode,
                )
                self.assertEqual(gold["repair"], spec.repair)

                witness = find_linear_algebra_counterexample(
                    source["formula_id"],
                    condition=source["condition"],
                )
                if spec.verdict == "valid":
                    self.assertIsNone(witness)
                    self.assertIsNone(gold["counterexample"])
                else:
                    self.assertIsNotNone(witness)
                    self.assertEqual(
                        gold["counterexample"],
                        witness.to_dict(),
                    )

            predictions = []
            for row in private_rows:
                prediction = {
                    "task_id": row["task_id"],
                    **row["gold"],
                }
                predictions.append(
                    {
                        "task_id": row["task_id"],
                        "model_id": "oracle",
                        "protocol": "oracle",
                        "prediction": prediction,
                    }
                )

            report = score_records(private_rows, predictions)
            self.assertEqual(
                report["summary"]["overall"]["safe_completion"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
