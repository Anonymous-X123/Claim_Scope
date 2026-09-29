import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from claimscope.witness_verifier import load_witness_verifier


def public_task() -> dict[str, object]:
    return {
        "task_id": "EXT-LA-002",
        "family": "linear_algebra",
        "split": "extended_v1",
        "prompt": "Public mathematical claim only.",
        "response_schema": {},
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class WitnessVerifierTests(unittest.TestCase):
    def verifier_in(self, root: Path):
        private_path = root / "private.jsonl"
        manifest_path = root / "manifest.json"

        private_row = {
            "task_id": "EXT-LA-002",
            "family": "linear_algebra",
            "source": {
                "formula_id": "transpose_same_order",
                "condition": "all",
            },
            "gold": {
                "verdict": "GOLD_SENTINEL_DO_NOT_DISCLOSE",
                "repair": "secret_repair",
            },
        }
        write_jsonl(private_path, [private_row])

        manifest_path.write_text(
            json.dumps(
                {
                    "private_sha256": hashlib.sha256(
                        private_path.read_bytes()
                    ).hexdigest()
                }
            ),
            encoding="utf-8",
        )

        verifier = load_witness_verifier(
            [public_task()],
            private_path,
            manifest_path,
        )
        return verifier, private_path, manifest_path

    def test_verified_feedback_does_not_expose_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier, _, _ = self.verifier_in(Path(directory))

            feedback = verifier.feedback(
                public_task(),
                {
                    "verdict": "invalid",
                    "counterexample": {
                        "state": {
                            "A": [["1", "1"], ["0", "1"]],
                            "B": [["1", "0"], ["1", "1"]],
                        }
                    },
                },
                None,
            )

            self.assertEqual(feedback["status"], "verified")
            serialized = json.dumps(feedback, sort_keys=True)
            self.assertNotIn("GOLD_SENTINEL_DO_NOT_DISCLOSE", serialized)
            self.assertNotIn("secret_repair", serialized)

            metadata = verifier.metadata()
            self.assertEqual(metadata["task_count"], 1)
            self.assertIn(
                "witness_validity_only",
                metadata["feedback_contract"],
            )

    def test_nonrefuting_witness_is_rejected_without_gold_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verifier, _, _ = self.verifier_in(Path(directory))

            feedback = verifier.feedback(
                public_task(),
                {
                    "verdict": "invalid",
                    "counterexample": {
                        "state": {
                            "A": [["1", "0"], ["0", "1"]],
                            "B": [["1", "0"], ["0", "1"]],
                        }
                    },
                },
                None,
            )

            self.assertEqual(feedback["status"], "rejected")
            serialized = json.dumps(feedback, sort_keys=True)
            self.assertNotIn("GOLD_SENTINEL_DO_NOT_DISCLOSE", serialized)
            self.assertNotIn("secret_repair", serialized)
            self.assertIn("does not reveal a correct verdict", serialized)

    def test_private_key_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, manifest_path = self.verifier_in(root)

            manifest_path.write_text(
                json.dumps({"private_sha256": "0" * 64}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_witness_verifier(
                    [public_task()],
                    root / "private.jsonl",
                    manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
