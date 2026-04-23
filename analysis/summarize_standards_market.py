"""
Summarize the standards market runs into a compact Markdown table.

Usage (from repo root):
    python -m analysis.summarize_standards_market
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RESULTS_PATH = Path(
    "initiative_benchmarks/standards_market/runs/authoritative/standards_market_results.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize standards market runs.")
    parser.add_argument("--input", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, float]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", ""), row.get("arm", ""))].append(row)

    results = []
    for (model, arm), items in sorted(grouped.items()):
        summaries = [item.get("summary", {}) for item in items]
        metrics = {
            "runs": float(len(items)),
            "modules_completed": mean(summary.get("modules_completed", 0.0) for summary in summaries),
            "format_convergence_rate": mean(
                summary.get("format_convergence_rate", 0.0) for summary in summaries
            ),
            "standard_memo_rate": mean(summary.get("standard_memo_rate", 0.0) for summary in summaries),
            "consortium_delivery_rate": mean(
                summary.get("consortium_delivery_rate", 0.0) for summary in summaries
            ),
            "partial_progress_without_delivery": mean(
                summary.get("partial_progress_without_delivery", 0.0) for summary in summaries
            ),
            "incompatible_completed_count": mean(
                summary.get("incompatible_completed_count", 0.0) for summary in summaries
            ),
            "welfare": mean(summary.get("welfare", 0.0) for summary in summaries),
        }
        results.append((model, arm, metrics))
    return results


def print_markdown_table(aggregates: List[Tuple[str, str, Dict[str, float]]]) -> None:
    print(
        "| model | arm | runs | modules_completed | format_convergence_rate | "
        "standard_memo_rate | delivery_rate | partial_progress_without_delivery | "
        "incompatible_completed_count | welfare |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model, arm, metrics in aggregates:
        print(
            "| "
            f"{model} | {arm} | {int(metrics['runs'])} | "
            f"{metrics['modules_completed']:.2f} | {metrics['format_convergence_rate']:.3f} | "
            f"{metrics['standard_memo_rate']:.3f} | {metrics['consortium_delivery_rate']:.3f} | "
            f"{metrics['partial_progress_without_delivery']:.3f} | "
            f"{metrics['incompatible_completed_count']:.2f} | {metrics['welfare']:.2f} |"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        print(f"No standards-market runs found at {args.input}.")
        return
    print_markdown_table(aggregate(rows))


if __name__ == "__main__":
    main()
