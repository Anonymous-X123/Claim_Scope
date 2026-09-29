import json
import tempfile
import unittest
from pathlib import Path

from claimscope.local_runner import (
    Generation,
    completed_keys,
    execute_protocol,
    load_public_tasks,
    parse_prediction,
    run_experiment,
    select_tasks,
)


FINAL_JSON = {
    "task_id": "PILOT-OPT-001",
    "verdict": "valid",
    "scope": "optimizer_direction",
    "failure_mode": "none",
    "counterexample": None,
    "repair": "none",
}


class FakeBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[list[dict[str, str]]] = []
        self.prefills: list[str | None] = []

    def generate(
        self,
        messages,
        *,
        max_new_tokens,
        temperature,
        prefill: str | None = None,
    ) -> Generation:
        self.calls.append(messages)
        self.prefills.append(prefill)
        text = next(self.outputs)
        if prefill is not None:
            text = prefill + text

        return Generation(
            text=text,
            input_tokens=10,
            output_tokens=20,
            seconds=0.25,
        )


def public_task() -> dict[str, object]:
    return {
        "task_id": "PILOT-OPT-001",
        "family": "optimizer_transport",
        "split": "pilot",
        "prompt": "Audit this claim and return JSON.",
        "response_schema": {},
    }


class LocalRunnerTests(unittest.TestCase):
    def test_parser_extracts_final_json_after_thinking(self) -> None:
        text = (
            "<think>Internal work with {braces}.</think>\n"
            "```json\n"
            + json.dumps(FINAL_JSON)
            + "\n```"
        )

        parsed, error = parse_prediction(text)

        self.assertIsNone(error)
        self.assertEqual(parsed, FINAL_JSON)

    def test_two_stage_protocol_uses_review_turn(self) -> None:
        backend = FakeBackend(
            [
                "Initial draft",
                json.dumps(FINAL_JSON),
            ]
        )

        result = execute_protocol(
            public_task(),
            "counterexample_guided",
            backend,
            max_new_tokens=256,
            temperature=0.0,
        )

        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result["prediction"], FINAL_JSON)
        self.assertIsNone(result["parse_error"])
        self.assertEqual(result["input_tokens"], 20)
        self.assertEqual(result["output_tokens"], 40)
        self.assertEqual(result["seconds"], 0.5)
        self.assertIsNotNone(result["draft_parse_error"])

        second_call = backend.calls[1]
        self.assertEqual(second_call[-2]["role"], "assistant")
        self.assertEqual(second_call[-1]["role"], "user")
        self.assertIn(
            "counterexample search",
            second_call[-1]["content"],
        )

    def test_verifier_feedback_uses_only_checker_message(self) -> None:
        class StubVerifier:
            def __init__(self) -> None:
                self.calls = []

            def feedback(self, task, prediction, parse_error):
                self.calls.append((task, prediction, parse_error))
                return {
                    "status": "rejected",
                    "message": (
                        "The submitted counterexample was not verified. "
                        "The verifier does not reveal a correct verdict, "
                        "repair, or answer key."
                    ),
                }

        backend = FakeBackend(
            [
                json.dumps(FINAL_JSON),
                json.dumps(FINAL_JSON),
            ]
        )
        verifier = StubVerifier()

        result = execute_protocol(
            public_task(),
            "verifier_feedback",
            backend,
            max_new_tokens=256,
            temperature=0.0,
            verifier=verifier,
        )

        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(result["draft_prediction"], FINAL_JSON)
        self.assertIsNone(result["draft_parse_error"])
        self.assertEqual(
            result["verifier_feedback"]["status"],
            "rejected",
        )

        review = backend.calls[1][-1]["content"]
        self.assertIn("external witness verifier", review.lower())
        self.assertIn("never reveals a gold verdict", review.lower())
        self.assertNotIn("GOLD_SENTINEL", review)

    def test_generic_feedback_is_checker_free_and_two_stage(self) -> None:
        backend = FakeBackend(
            [
                json.dumps(FINAL_JSON),
                json.dumps(FINAL_JSON),
            ]
        )

        result = execute_protocol(
            public_task(),
            "generic_feedback",
            backend,
            max_new_tokens=256,
            temperature=0.0,
        )

        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result["prediction"], FINAL_JSON)
        self.assertIsNone(result["parse_error"])
        self.assertEqual(result["draft_prediction"], FINAL_JSON)
        self.assertIsNone(result["draft_parse_error"])
        self.assertIsNone(result["verifier_feedback"])

        review = " ".join(
            backend.calls[1][-1]["content"].lower().split()
        )
        self.assertIn("generic-feedback control", review)
        self.assertIn("no external checker", review)
        self.assertIn("no task-specific correctness signal", review)
        self.assertNotIn("gold", review)

    def test_json_prefill_protocol_completes_one_json_object(
        self,
    ) -> None:
        prefill = (
            '{"task_id": "PILOT-OPT-001", "verdict": '
        )
        payload = json.dumps(FINAL_JSON)
        self.assertTrue(payload.startswith(prefill))

        backend = FakeBackend([payload[len(prefill):]])
        result = execute_protocol(
            public_task(),
            "json_prefill",
            backend,
            max_new_tokens=256,
            temperature=0.0,
        )

        self.assertEqual(result["prediction"], FINAL_JSON)
        self.assertIsNone(result["parse_error"])
        self.assertEqual(backend.prefills, [prefill])
        self.assertEqual(len(backend.calls), 1)


    def test_public_loader_rejects_answer_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"

            leaked = public_task()
            leaked["gold"] = {"verdict": "valid"}
            path.write_text(
                json.dumps(leaked) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_public_tasks(path)

    def test_runner_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "responses.jsonl"
            backend = FakeBackend([json.dumps(FINAL_JSON)])

            first = run_experiment(
                tasks=[public_task()],
                protocols=("direct",),
                backend=backend,
                model_id="fake/model",
                output_path=output,
                max_new_tokens=256,
                temperature=0.0,
                seed=7,
                resume=False,
                limit=None,
            )

            self.assertEqual(first["written"], 1)
            self.assertIn(
                (
                    "PILOT-OPT-001",
                    "direct",
                    "fake/model",
                ),
                completed_keys(output),
            )

            second = run_experiment(
                tasks=[public_task()],
                protocols=("direct",),
                backend=backend,
                model_id="fake/model",
                output_path=output,
                max_new_tokens=256,
                temperature=0.0,
                seed=7,
                resume=True,
                limit=None,
            )

            self.assertEqual(second["written"], 0)
            self.assertEqual(second["skipped"], 1)


    def test_failed_generation_is_not_marked_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "responses.jsonl"
            failure = {
                "task_id": "PILOT-OPT-001",
                "protocol": "direct",
                "model_id": "fake/model",
                "runner_error": "RuntimeError: simulated",
                "prediction": None,
            }
            output.write_text(
                json.dumps(failure) + "\n",
                encoding="utf-8",
            )

            self.assertNotIn(
                (
                    "PILOT-OPT-001",
                    "direct",
                    "fake/model",
                ),
                completed_keys(output),
            )


    def test_partial_parser_prefers_outer_prediction(self) -> None:
        partial = {
            "task_id": "PILOT-OPT-001",
            "verdict": "invalid",
            "scope": "full_transition",
            "failure_mode": "none",
            "counterexample": {
                "state": {"w": 1},
                "repair": "none",
            },
        }

        parsed, error = parse_prediction(
            "```json\n"
            + json.dumps(partial)
            + "\n```"
        )

        self.assertIsNotNone(error)
        self.assertEqual(parsed, partial)
        self.assertNotEqual(parsed, partial["counterexample"])
        self.assertNotEqual(
            parsed,
            partial["counterexample"]["state"],
        )


    def test_runtime_metadata_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "responses.jsonl"
            backend = FakeBackend([json.dumps(FINAL_JSON)])
            metadata = {
                "runner_version": "test",
                "revision": "immutable-revision",
                "thinking": "off",
                "quantization": "4bit",
            }

            run_experiment(
                tasks=[public_task()],
                protocols=("direct",),
                backend=backend,
                model_id="fake/model",
                output_path=output,
                max_new_tokens=256,
                temperature=0.0,
                seed=7,
                resume=False,
                limit=None,
                run_metadata=metadata,
            )

            row = json.loads(
                output.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(row["run_metadata"], metadata)


    def test_explicit_task_selection_preserves_requested_order(
        self,
    ) -> None:
        optimizer = public_task()
        recurrence = {
            **public_task(),
            "task_id": "PILOT-REC-001",
            "family": "affine_recurrence",
        }

        selected = select_tasks(
            [optimizer, recurrence],
            ["PILOT-REC-001", "PILOT-OPT-001"],
        )

        self.assertEqual(
            [task["task_id"] for task in selected],
            ["PILOT-REC-001", "PILOT-OPT-001"],
        )

        with self.assertRaises(ValueError):
            select_tasks(
                [optimizer, recurrence],
                ["PILOT-UNKNOWN-999"],
            )


if __name__ == "__main__":
    unittest.main()
