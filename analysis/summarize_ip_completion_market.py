"""
Summarize the customer-order IP initiative runs into a compact Markdown table.

Usage (from repo root):
    python -m analysis.summarize_ip_completion_market
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RESULTS_PATH = Path("runs_ip_completion_market/ip_completion_market_results.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize IP completion market runs.")
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
            "fulfillment_rate": mean(summary.get("fulfillment_rate", 0.0) for summary in summaries),
            "started_not_delivered_count": mean(
                summary.get("started_not_delivered_count", 0.0) for summary in summaries
            ),
            "self_initiated_trade_rate": mean(
                summary.get("self_initiated_trade_rate", 0.0) for summary in summaries
            ),
            "deal_volume": mean(summary.get("deal_volume", 0.0) for summary in summaries),
            "substitute_builds_started": mean(
                summary.get("substitute_builds_started", 0.0) for summary in summaries
            ),
            "substitute_builds_completed": mean(
                summary.get("substitute_builds_completed", 0.0) for summary in summaries
            ),
            "welfare": mean(summary.get("welfare", 0.0) for summary in summaries),
            "customer_value_captured": mean(
                summary.get("customer_value_captured", 0.0) for summary in summaries
            ),
        }
        results.append((model, arm, metrics))
    return results


def print_markdown_table(aggregates: List[Tuple[str, str, Dict[str, float]]]) -> None:
    print(
        "| model | arm | runs | fulfillment_rate | started_not_delivered | "
        "self_initiated_trade_rate | deal_volume | substitute_builds_started | "
        "substitute_builds_completed | welfare | customer_value_captured |"
    )
    print(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for model, arm, metrics in aggregates:
        print(
            "| "
            f"{model} | {arm} | {int(metrics['runs'])} | {metrics['fulfillment_rate']:.3f} | "
            f"{metrics['started_not_delivered_count']:.2f} | "
            f"{metrics['self_initiated_trade_rate']:.3f} | "
            f"{metrics['deal_volume']:.2f} | {metrics['substitute_builds_started']:.2f} | "
            f"{metrics['substitute_builds_completed']:.2f} | {metrics['welfare']:.2f} | "
            f"{metrics['customer_value_captured']:.2f} |"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        print(f"No completion-market runs found at {args.input}.")
        return
    print_markdown_table(aggregate(rows))


if __name__ == "__main__":
    main()
