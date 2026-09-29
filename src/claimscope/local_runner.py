"""Resumable local-model runner for ClaimScope benchmark tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .witness_verifier import WitnessVerifier, load_witness_verifier


RUNNER_VERSION = "0.7.0"
PROTOCOLS = (
    "direct",
    "self_reflect",
    "counterexample_guided",
    "json_prefill",
    "generic_feedback",
    "verifier_feedback",
)

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT

DEFAULT_INPUT_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "extended_v1_public.jsonl"
)
DEFAULT_MANIFEST_PATH = (
    SOURCE_ROOT / "artifacts" / "benchmark" / "extended_v1_manifest.json"
)
DEFAULT_CACHE_PATH = PROJECT_ROOT / "models" / "huggingface"

FORBIDDEN_PUBLIC_FIELDS = {
    "source",
    "gold",
    "expected_verdict",
    "failure_mode",
    "repair",
    "counterexample",
    "cases_checked",
}

PREDICTION_FIELDS = {
    "verdict",
    "scope",
    "failure_mode",
    "counterexample",
    "repair",
}

SYSTEM_PROMPT = """You are an exacting mathematical research auditor.
Check identities under precisely the stated assumptions. A numerical
counterexample must satisfy every stated condition. Return the final response
as one JSON object matching the task schema. Do not use external answer keys."""

SELF_REFLECT_PROMPT = """Review your draft critically before answering again.
Check each algebraic factor, the claimed scope, all boundary or singular cases,
and whether any proposed counterexample satisfies the assumptions and truly
creates unequal values. Replace the draft with one final JSON object only."""

COUNTEREXAMPLE_GUIDED_PROMPT = """Audit the draft using an explicit
counterexample search. For a potentially invalid claim, try small exact
integers or rational values, verify all assumptions, and recompute both sides.
For a potentially valid claim, explain to yourself why no admissible witness
can exist and identify the exact identity or restriction. Then return one
final JSON object only, including a machine-checkable state when invalid."""


GENERIC_FEEDBACK_PROMPT = """This is a matched generic-feedback control.
No external checker has examined your draft, and no task-specific correctness
signal is available. Your submitted counterexample may fail an assumption or
may not refute the claim. Independently recheck the mathematics, both sides of
the proposed witness, and all stated conditions. Then replace the draft with
one final JSON object only. Preserve a counterexample only when it satisfies
every assumption and actually refutes the claim."""


VERIFIER_FEEDBACK_PROMPT = """An external witness verifier has checked only
the counterexample submitted in your draft. It never reveals a gold verdict,
answer key, repair, or proof. Treat rejected feedback as a reason to recheck
your mathematics, not as proof that the claim is valid.

Verifier feedback:
{feedback}

Now re-audit the claim and return one final JSON object only. Preserve a
counterexample only when it satisfies every stated assumption and actually
refutes the claim."""


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(row)

    return rows


def load_public_tasks(
    path: Path,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    resolved = path.resolve()
    reference_root = (
        PROJECT_ROOT / "artifacts" / "reference"
    ).resolve()

    try:
        resolved.relative_to(reference_root)
    except ValueError:
        pass
    else:
        raise ValueError("The model runner cannot read reference artifacts")

    if manifest_path is not None:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        expected_hash = manifest.get("public_sha256")
        actual_hash = _sha256(path)

        if expected_hash != actual_hash:
            raise ValueError(
                "Public benchmark SHA-256 does not match the manifest"
            )

    tasks = read_jsonl(path)
    seen: set[str] = set()

    for task in tasks:
        if FORBIDDEN_PUBLIC_FIELDS.intersection(task):
            raise ValueError(
                f"Public task exposes forbidden fields: {task.get('task_id')}"
            )

        required = {
            "task_id",
            "family",
            "split",
            "prompt",
            "response_schema",
        }
        if not required.issubset(task):
            raise ValueError(
                f"Public task is missing required fields: {task}"
            )

        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every public task needs a non-empty task_id")
        if task_id in seen:
            raise ValueError(f"Duplicate public task_id: {task_id}")

        seen.add(task_id)

    return tasks


def select_tasks(
    tasks: list[dict[str, Any]],
    task_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not task_ids:
        return list(tasks)

    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Requested task IDs must be unique")

    by_id = {task["task_id"]: task for task in tasks}
    missing = [
        task_id
        for task_id in task_ids
        if task_id not in by_id
    ]
    if missing:
        raise ValueError(
            "Unknown requested task IDs: " + ", ".join(missing)
        )

    return [by_id[task_id] for task_id in task_ids]


def strip_thinking(text: str) -> str:
    result = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if "</think>" in result.lower():
        index = result.lower().rfind("</think>")
        result = result[index + len("</think>"):]

    return result.strip()


def parse_prediction(text: str) -> tuple[dict[str, Any] | None, str | None]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            candidates.append(value)

    if not candidates:
        return None, "No valid JSON object was found in the model output"

    for candidate in reversed(candidates):
        if PREDICTION_FIELDS.issubset(candidate):
            return candidate, None

    partial = max(
        candidates,
        key=lambda candidate: (
            len(PREDICTION_FIELDS.intersection(candidate)),
            len(candidate),
        ),
    )

    return (
        partial,
        "A JSON object was found, but required prediction fields are missing",
    )


def _messages(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]


def _json_prefill(task: dict[str, Any]) -> str:
    return (
        '{"task_id": '
        + json.dumps(task["task_id"])
        + ', "verdict": '
    )


def execute_protocol(
    task: dict[str, Any],
    protocol: str,
    backend: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    verifier: WitnessVerifier | None = None,
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")

    messages = _messages(task)
    draft: Generation | None = None
    feedback: dict[str, str] | None = None
    draft_prediction: dict[str, Any] | None = None
    draft_parse_error: str | None = None

    if protocol == "direct":
        final = backend.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif protocol == "json_prefill":
        final = backend.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prefill=_json_prefill(task),
        )
    else:
        draft = backend.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        draft_prediction, draft_parse_error = parse_prediction(draft.text)

        if protocol == "self_reflect":
            review_prompt = SELF_REFLECT_PROMPT
        elif protocol == "counterexample_guided":
            review_prompt = COUNTEREXAMPLE_GUIDED_PROMPT
        elif protocol == "generic_feedback":
            review_prompt = GENERIC_FEEDBACK_PROMPT
        elif protocol == "verifier_feedback":
            if verifier is None:
                raise ValueError(
                    "verifier_feedback requires an exact witness verifier"
                )

            feedback = verifier.feedback(
                task,
                draft_prediction,
                draft_parse_error,
            )
            review_prompt = VERIFIER_FEEDBACK_PROMPT.format(
                feedback=feedback["message"]
            )
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        messages = [
            *messages,
            {
                "role": "assistant",
                "content": strip_thinking(draft.text),
            },
            {"role": "user", "content": review_prompt},
        ]

        final = backend.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    prediction, parse_error = parse_prediction(final.text)

    return {
        "prediction": prediction,
        "parse_error": parse_error,
        "raw_output": final.text,
        "draft_output": None if draft is None else draft.text,
        "draft_prediction": draft_prediction,
        "draft_parse_error": draft_parse_error,
        "verifier_feedback": feedback,
        "input_tokens": final.input_tokens + (
            0 if draft is None else draft.input_tokens
        ),
        "output_tokens": final.output_tokens + (
            0 if draft is None else draft.output_tokens
        ),
        "seconds": final.seconds + (
            0.0 if draft is None else draft.seconds
        ),
    }


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str, str]] = set()

    for row in read_jsonl(path):
        if row.get("runner_error"):
            continue

        keys.add(
            (
                str(row.get("task_id", "")),
                str(row.get("protocol", "")),
                str(row.get("model_id", "")),
            )
        )

    return keys


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class TransformersBackend:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: Path,
        quantization: str,
        thinking: str,
        seed: int,
        max_input_tokens: int,
    ) -> None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip() != "1":
            raise RuntimeError(
                "Set CUDA_VISIBLE_DEVICES=1 so only physical GPU 1 is visible"
            )

        try:
            import torch
            import transformers
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install the GPU runner dependencies before loading a model"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Exactly one logical CUDA device must be visible"
            )

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.torch = torch
        self.transformers = transformers
        self.model_id = model_id
        self.revision = revision
        self.quantization = quantization
        self.thinking = thinking
        self.max_input_tokens = max_input_tokens

        cache_dir.mkdir(parents=True, exist_ok=True)

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": str(cache_dir),
            "device_map": {"": 0},
            "dtype": compute_dtype,
            "low_cpu_mem_usage": True,
        }

        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif quantization != "none":
            raise ValueError(f"Unknown quantization: {quantization}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()

    def metadata(self) -> dict[str, Any]:
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(0)

        return {
            "runner_version": RUNNER_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "quantization": self.quantization,
            "thinking": self.thinking,
            "torch_version": self.torch.__version__,
            "transformers_version": self.transformers.__version__,
            "cuda_version": self.torch.version.cuda,
            "device_name": self.torch.cuda.get_device_name(0),
            "free_memory_bytes_after_load": free_bytes,
            "total_memory_bytes": total_bytes,
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        prefill: str | None = None,
    ) -> Generation:
        template_options: dict[str, Any] = {}

        if self.thinking == "on":
            template_options["enable_thinking"] = True
        elif self.thinking == "off":
            template_options["enable_thinking"] = False

        template_messages = messages
        continue_final_message = False

        if prefill is not None:
            template_messages = [
                *messages,
                {"role": "assistant", "content": prefill},
            ]
            continue_final_message = True

        inputs = self.tokenizer.apply_chat_template(
            template_messages,
            tokenize=True,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
            return_dict=True,
            return_tensors="pt",
            **template_options,
        )
        inputs = inputs.to(self.model.device)

        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                f"Prompt has {input_tokens} tokens; limit is "
                f"{self.max_input_tokens}"
            )

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature

        started = time.perf_counter()

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - started

        generated_ids = output[0, input_tokens:]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        if prefill is not None:
            text = prefill + text

        result = Generation(
            text=text,
            input_tokens=input_tokens,
            output_tokens=int(generated_ids.shape[-1]),
            seconds=seconds,
        )

        del output
        del generated_ids
        del inputs
        return result


def run_experiment(
    *,
    tasks: list[dict[str, Any]],
    protocols: tuple[str, ...],
    backend: Any,
    model_id: str,
    output_path: Path,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    resume: bool,
    limit: int | None,
    run_metadata: dict[str, Any] | None = None,
    verifier: WitnessVerifier | None = None,
) -> dict[str, int]:
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"{output_path} already exists; pass --resume to continue"
        )

    completed = completed_keys(output_path)
    selected = tasks if limit is None else tasks[:limit]
    persisted_metadata = dict(run_metadata or {})

    written = 0
    skipped = 0

    for task in selected:
        for protocol in protocols:
            key = (task["task_id"], protocol, model_id)

            if key in completed:
                skipped += 1
                continue

            print(
                f"RUN {task['task_id']} protocol={protocol}",
                flush=True,
            )

            started_at = datetime.now(timezone.utc).isoformat()

            try:
                result = execute_protocol(
                    task,
                    protocol,
                    backend,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    verifier=verifier,
                )
            except Exception as error:
                failure = {
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "model_id": model_id,
                    "protocol": protocol,
                    "started_at": started_at,
                    "prediction": None,
                    "parse_error": None,
                    "runner_error": (
                        f"{type(error).__name__}: {error}"
                    ),
                    "seed": seed,
                    "run_metadata": persisted_metadata,
                }
                append_jsonl(output_path, failure)
                raise

            record = {
                "task_id": task["task_id"],
                "family": task["family"],
                "model_id": model_id,
                "protocol": protocol,
                "started_at": started_at,
                "seed": seed,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "run_metadata": persisted_metadata,
                **result,
            }

            append_jsonl(output_path, record)
            completed.add(key)
            written += 1

    return {
        "written": written,
        "skipped": skipped,
        "selected_tasks": len(selected),
        "protocol_count": len(protocols),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Hugging Face models on ClaimScope."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_PATH,
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=PROTOCOLS,
        dest="protocols",
    )
    parser.add_argument(
        "--verifier-key",
        type=Path,
        help=(
            "Private, manifest-hash-checked witness verifier key. Required "
            "only by the verifier_feedback tool condition."
        ),
    )
    parser.add_argument(
        "--quantization",
        choices=("4bit", "none"),
        default="4bit",
    )
    parser.add_argument(
        "--thinking",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Run only this task ID; repeat for multiple tasks.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature cannot be negative")

    protocols = tuple(args.protocols or PROTOCOLS)
    tasks = load_public_tasks(args.input, args.manifest)
    tasks = select_tasks(tasks, args.task_ids)

    verifier: WitnessVerifier | None = None
    if "verifier_feedback" in protocols:
        if args.verifier_key is None:
            raise SystemExit(
                "--verifier-key is required for verifier_feedback"
            )
        verifier = load_witness_verifier(
            tasks,
            args.verifier_key,
            args.manifest,
        )
    elif args.verifier_key is not None:
        raise SystemExit(
            "--verifier-key is valid only with verifier_feedback"
        )

    backend = TransformersBackend(
        model_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        quantization=args.quantization,
        thinking=args.thinking,
        seed=args.seed,
        max_input_tokens=args.max_input_tokens,
    )

    backend_metadata = backend.metadata()
    benchmark_manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    run_metadata = {
        **backend_metadata,
        "benchmark_public_sha256": benchmark_manifest.get(
            "public_sha256"
        ),
        "input_path": str(args.input.resolve()),
        "requested_task_ids": args.task_ids,
        "limit": args.limit,
    }
    if verifier is not None:
        run_metadata["verifier"] = verifier.metadata()

    print(
        json.dumps(
            run_metadata,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    summary = run_experiment(
        tasks=tasks,
        protocols=protocols,
        backend=backend,
        model_id=args.model,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        resume=args.resume,
        limit=args.limit,
        run_metadata=run_metadata,
        verifier=verifier,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
