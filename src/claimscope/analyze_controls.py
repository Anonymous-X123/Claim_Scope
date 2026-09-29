"""Paired, template-aware, and cost analysis for ClaimScope conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


ANALYZER_VERSION = "0.1.0"
METRICS = (
    "json_valid",
    "verdict_correct",
    "scope_correct",
    "repair_correct",
    "counterexample_valid",
    "safe_completion",
    "strict_completion",
)
FOCUS_METRICS = (
    "verdict_correct",
    "counterexample_valid",
    "safe_completion",
    "strict_completion",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


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


def _index(
    rows: list[dict[str, Any]],
    *,
    protocol: str,
    source: Path,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("protocol") != protocol:
            continue
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{source}: row has no task_id")
        if task_id in indexed:
            raise ValueError(
                f"{source}: duplicate {protocol} record for {task_id}"
            )
        indexed[task_id] = row
    if not indexed:
        raise ValueError(f"{source}: no records for protocol {protocol}")
    return indexed


def _metric_count(
    rows: dict[str, dict[str, Any]], metric: str
) -> int:
    return sum(int(row.get(metric, 0)) for row in rows.values())


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _metric_summary(
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    result: dict[str, Any] = {"task_count": total}
    for metric in METRICS:
        count = _metric_count(rows, metric)
        result[metric] = {"count": count, "rate": _rate(count, total)}
    return result


def _cost_summary(
    run_rows: dict[str, dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_tokens = sum(int(row.get("input_tokens", 0)) for row in run_rows.values())
    output_tokens = sum(
        int(row.get("output_tokens", 0)) for row in run_rows.values()
    )
    seconds_values = [
        float(row.get("seconds", 0.0)) for row in run_rows.values()
    ]
    total_seconds = sum(seconds_values)
    generation_count = sum(
        2 if row.get("draft_output") is not None else 1
        for row in run_rows.values()
    )
    safe = _metric_count(score_rows, "safe_completion")

    return {
        "generation_count": generation_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_seconds": total_seconds,
        "median_seconds_per_task": statistics.median(seconds_values),
        "safe_per_1000_output_tokens": (
            1000.0 * safe / output_tokens if output_tokens else 0.0
        ),
        "safe_per_gpu_minute": (
            safe / (total_seconds / 60.0) if total_seconds else 0.0
        ),
    }


def _template_key(gold_row: dict[str, Any]) -> str:
    source = dict(gold_row.get("source", {}))
    source.pop("surface_variant", None)
    value = {
        "family": gold_row["family"],
        "source": source,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _template_label(gold_row: dict[str, Any]) -> str:
    source = gold_row.get("source", {})
    parts = [
        str(
            source.get("semantic_id")
            or source.get("formula_id")
            or "unknown"
        )
    ]
    for field in ("condition", "protocol", "quantity"):
        value = source.get(field)
        if value is not None and value not in parts:
            parts.append(str(value))
    return " | ".join(parts)


def _template_groups(
    gold_rows: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id, row in gold_rows.items():
        groups[_template_key(row)].append(task_id)
    return dict(groups)


def _template_macro(
    score_rows: dict[str, dict[str, Any]],
    groups: dict[str, list[str]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        template_rates = [
            statistics.mean(
                int(score_rows[task_id].get(metric, 0))
                for task_id in task_ids
            )
            for task_ids in groups.values()
        ]
        result[metric] = statistics.mean(template_rates)
    return result


def _transition_counts(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    metric: str,
    task_ids: list[str],
) -> dict[str, int]:
    counts = {"0_to_0": 0, "0_to_1": 0, "1_to_0": 0, "1_to_1": 0}
    for task_id in task_ids:
        before = int(baseline[task_id].get(metric, 0))
        after = int(treatment[task_id].get(metric, 0))
        counts[f"{before}_to_{after}"] += 1
    return counts


def _bootstrap_interval(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    generator = random.Random(seed)
    size = len(deltas)
    means = sorted(
        statistics.mean(generator.choice(deltas) for _ in range(size))
        for _ in range(samples)
    )

    def percentile(probability: float) -> float:
        index = round(probability * (len(means) - 1))
        return means[index]

    return [percentile(0.025), percentile(0.975)]


def _template_comparison(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    groups: dict[str, list[str]],
    *,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas: list[float] = []
    improved = unchanged = worsened = 0
    for task_ids in groups.values():
        before = statistics.mean(
            int(baseline[task_id].get(metric, 0)) for task_id in task_ids
        )
        after = statistics.mean(
            int(treatment[task_id].get(metric, 0)) for task_id in task_ids
        )
        delta = after - before
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta < 0:
            worsened += 1
        else:
            unchanged += 1

    return {
        "mean_delta": statistics.mean(deltas),
        "bootstrap_95_interval": _bootstrap_interval(
            deltas,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "templates_improved": improved,
        "templates_unchanged": unchanged,
        "templates_worsened": worsened,
    }


def _family_summary(
    score_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    families = sorted({str(row["family"]) for row in score_rows.values()})
    result: dict[str, dict[str, Any]] = {}
    for family in families:
        selected = {
            task_id: row
            for task_id, row in score_rows.items()
            if row["family"] == family
        }
        result[family] = _metric_summary(selected)
    return result


def analyze_files(
    *,
    gold_path: Path,
    conditions: list[tuple[str, Path, Path]],
    baseline_name: str,
    require_complete: bool,
    bootstrap_samples: int,
    seed: int = 20260904,
    comparison_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    gold_list = _read_jsonl(gold_path)
    gold_rows = {
        str(row["task_id"]): row
        for row in gold_list
    }
    if len(gold_rows) != len(gold_list):
        raise ValueError(f"{gold_path}: duplicate task_id")
    gold_ids = set(gold_rows)
    groups = _template_groups(gold_rows)

    loaded: dict[str, dict[str, Any]] = {}
    for name, run_path, score_path in conditions:
        if name in loaded:
            raise ValueError(f"duplicate condition: {name}")
        run_rows = _index(
            _read_jsonl(run_path), protocol=name, source=run_path
        )
        score_payload = _read_json(score_path)
        score_value = score_payload.get("rows")
        if not isinstance(score_value, list):
            raise ValueError(f"{score_path}: rows must be a list")
        score_rows = _index(score_value, protocol=name, source=score_path)

        if set(run_rows) != set(score_rows):
            raise ValueError(
                f"{name}: run and score task sets do not match"
            )
        unknown = set(score_rows) - gold_ids
        if unknown:
            raise ValueError(f"{name}: unknown task IDs: {sorted(unknown)}")
        if require_complete and set(score_rows) != gold_ids:
            missing = sorted(gold_ids - set(score_rows))
            raise ValueError(f"{name}: incomplete condition; missing {missing}")

        active_gold = {
            task_id: gold_rows[task_id] for task_id in score_rows
        }
        active_groups = _template_groups(active_gold)
        models = sorted(
            {str(row.get("model_id", "")) for row in run_rows.values()}
        )
        if len(models) != 1:
            raise ValueError(f"{name}: expected exactly one model, got {models}")

        loaded[name] = {
            "model_id": models[0],
            "run_path": str(run_path),
            "run_sha256": _sha256(run_path),
            "score_path": str(score_path),
            "score_sha256": _sha256(score_path),
            "metrics": _metric_summary(score_rows),
            "families": _family_summary(score_rows),
            "cost": _cost_summary(run_rows, score_rows),
            "template_count": len(active_groups),
            "template_macro": _template_macro(score_rows, active_groups),
            "_scores": score_rows,
        }

    if baseline_name not in loaded:
        raise ValueError(f"baseline condition is missing: {baseline_name}")

    requested_pairs = [
        (baseline_name, name) for name in loaded if name != baseline_name
    ]
    requested_pairs.extend(comparison_pairs or [])
    pairs = list(dict.fromkeys(requested_pairs))

    comparisons: dict[str, Any] = {}
    for before_name, after_name in pairs:
        if before_name not in loaded or after_name not in loaded:
            raise ValueError(
                f"comparison condition is missing: {before_name}, {after_name}"
            )
        if before_name == after_name:
            raise ValueError("comparison conditions must be distinct")

        baseline_scores = loaded[before_name]["_scores"]
        treatment_scores = loaded[after_name]["_scores"]
        shared = sorted(set(baseline_scores) & set(treatment_scores))
        if require_complete and len(shared) != len(gold_rows):
            raise ValueError(
                f"{before_name} and {after_name} are not fully paired"
            )

        paired: dict[str, Any] = {"task_count": len(shared), "metrics": {}}
        for metric in FOCUS_METRICS:
            transitions = _transition_counts(
                baseline_scores, treatment_scores, metric, shared
            )
            paired["metrics"][metric] = {
                "transitions": transitions,
                "delta": transitions["0_to_1"] - transitions["1_to_0"],
                "gained_task_ids": [
                    task_id
                    for task_id in shared
                    if int(baseline_scores[task_id].get(metric, 0)) == 0
                    and int(treatment_scores[task_id].get(metric, 0)) == 1
                ],
                "lost_task_ids": [
                    task_id
                    for task_id in shared
                    if int(baseline_scores[task_id].get(metric, 0)) == 1
                    and int(treatment_scores[task_id].get(metric, 0)) == 0
                ],
            }

        shared_gold = {task_id: gold_rows[task_id] for task_id in shared}
        shared_groups = _template_groups(shared_gold)
        paired["template_comparisons"] = {
            metric: _template_comparison(
                baseline_scores,
                treatment_scores,
                shared_groups,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index,
            )
            for index, metric in enumerate(FOCUS_METRICS)
        }
        comparisons[f"{before_name}_to_{after_name}"] = paired

    for condition in loaded.values():
        condition.pop("_scores")

    return {
        "analyzer": "claimscope.analyze_controls",
        "analyzer_version": ANALYZER_VERSION,
        "baseline": baseline_name,
        "gold_path": str(gold_path),
        "gold_sha256": _sha256(gold_path),
        "gold_task_count": len(gold_rows),
        "unique_template_count": len(groups),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "conditions": loaded,
        "paired_comparisons": comparisons,
    }


def _count_cell(condition: dict[str, Any], metric: str) -> str:
    value = condition["metrics"][metric]
    return f"{value['count']}/{condition['metrics']['task_count']}"


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# ClaimScope matched-control analysis",
        "",
        (
            f"Analyzer `{analysis['analyzer_version']}`; "
            f"{analysis['gold_task_count']} tasks; "
            f"{analysis['unique_template_count']} unique templates."
        ),
        "",
        "## Overall results and inference cost",
        "",
        "| Condition | JSON | Verdict | Witness | Safe | Strict | Generations | Output tokens | GPU seconds | Safe / 1k output tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, condition in analysis["conditions"].items():
        cost = condition["cost"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _count_cell(condition, "json_valid"),
                    _count_cell(condition, "verdict_correct"),
                    _count_cell(condition, "counterexample_valid"),
                    _count_cell(condition, "safe_completion"),
                    _count_cell(condition, "strict_completion"),
                    str(cost["generation_count"]),
                    str(cost["output_tokens"]),
                    f"{cost['total_seconds']:.1f}",
                    f"{cost['safe_per_1000_output_tokens']:.3f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Template-weighted results",
            "",
            "Each unique mathematical source specification receives equal weight; `surface_variant` is excluded from the template key.",
            "",
            "| Condition | Templates | Verdict | Witness | Safe | Strict |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, condition in analysis["conditions"].items():
        macro = condition["template_macro"]
        lines.append(
            f"| {name} | {condition['template_count']} | "
            f"{macro['verdict_correct']:.3f} | "
            f"{macro['counterexample_valid']:.3f} | "
            f"{macro['safe_completion']:.3f} | "
            f"{macro['strict_completion']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Paired changes",
            "",
            "| Comparison | Witness delta | Safe delta | Strict delta | Safe gains | Safe losses | Template-safe improved / unchanged / worsened | Template-safe mean delta (95% cluster bootstrap interval) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, comparison in analysis["paired_comparisons"].items():
        metrics = comparison["metrics"]
        template = comparison["template_comparisons"]["safe_completion"]
        interval = template["bootstrap_95_interval"]
        lines.append(
            f"| {name} | {metrics['counterexample_valid']['delta']:+d} | "
            f"{metrics['safe_completion']['delta']:+d} | "
            f"{metrics['strict_completion']['delta']:+d} | "
            f"{len(metrics['safe_completion']['gained_task_ids'])} | "
            f"{len(metrics['safe_completion']['lost_task_ids'])} | "
            f"{template['templates_improved']} / "
            f"{template['templates_unchanged']} / "
            f"{template['templates_worsened']} | "
            f"{template['mean_delta']:+.3f} "
            f"([{interval[0]:+.3f}, {interval[1]:+.3f}]) |"
        )

    lines.extend(["", "## Provenance", ""])
    for name, condition in analysis["conditions"].items():
        lines.extend(
            [
                f"- `{name}` run SHA-256: `{condition['run_sha256']}`",
                f"- `{name}` score SHA-256: `{condition['score_sha256']}`",
            ]
        )
    lines.append(f"- Gold SHA-256: `{analysis['gold_sha256']}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze matched ClaimScope controls with paired, template-aware, "
            "and inference-cost summaries."
        )
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--condition",
        nargs=3,
        action="append",
        metavar=("NAME", "RUN_JSONL", "SCORE_JSON"),
        required=True,
    )
    parser.add_argument("--baseline", default="direct")
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("FROM", "TO"),
        help="Add a paired comparison beyond baseline-to-condition pairs.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conditions = [
        (name, Path(run_path), Path(score_path))
        for name, run_path, score_path in args.condition
    ]
    analysis = analyze_files(
        gold_path=args.gold,
        conditions=conditions,
        baseline_name=args.baseline,
        require_complete=args.require_complete,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        comparison_pairs=[tuple(pair) for pair in (args.pair or [])],
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.output_markdown.write_text(
        render_markdown(analysis),
        encoding="utf-8",
        newline="\n",
    )
    print(render_markdown(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
