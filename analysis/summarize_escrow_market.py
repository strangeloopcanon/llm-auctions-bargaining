"""
Summarize the escrow market runs into a compact Markdown table.

Usage (from repo root):
    python -m analysis.summarize_escrow_market
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RESULTS_PATH = Path("runs_escrow_market/escrow_market_results.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize escrow market runs.")
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
            "institution_notice_rate": mean(
                summary.get("institution_notice_rate", 0.0) for summary in summaries
            ),
            "institution_activation_rate": mean(
                summary.get("institution_activation_rate", 0.0) for summary in summaries
            ),
            "reply_rate": mean(summary.get("reply_rate", 0.0) for summary in summaries),
            "board_use_rate": mean(summary.get("board_use_rate", 0.0) for summary in summaries),
            "deal_volume": mean(summary.get("deal_volume", 0.0) for summary in summaries),
            "safe_deal_volume": mean(summary.get("safe_deal_volume", 0.0) for summary in summaries),
            "failed_deal_volume": mean(
                summary.get("failed_deal_volume", 0.0) for summary in summaries
            ),
            "orders_fulfilled": mean(summary.get("orders_fulfilled", 0.0) for summary in summaries),
            "fulfillment_rate": mean(summary.get("fulfillment_rate", 0.0) for summary in summaries),
            "started_not_delivered_count": mean(
                summary.get("started_not_delivered_count", 0.0) for summary in summaries
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
        "| model | arm | runs | institution_notice_rate | institution_activation_rate | "
        "reply_rate | board_use_rate | deal_volume | safe_deal_volume | failed_deal_volume | "
        "orders_fulfilled | fulfillment_rate | started_not_delivered | welfare | customer_value_captured |"
    )
    print(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for model, arm, metrics in aggregates:
        print(
            "| "
            f"{model} | {arm} | {int(metrics['runs'])} | "
            f"{metrics['institution_notice_rate']:.3f} | {metrics['institution_activation_rate']:.3f} | "
            f"{metrics['reply_rate']:.3f} | {metrics['board_use_rate']:.3f} | "
            f"{metrics['deal_volume']:.2f} | {metrics['safe_deal_volume']:.2f} | "
            f"{metrics['failed_deal_volume']:.2f} | {metrics['orders_fulfilled']:.2f} | "
            f"{metrics['fulfillment_rate']:.3f} | {metrics['started_not_delivered_count']:.2f} | "
            f"{metrics['welfare']:.2f} | {metrics['customer_value_captured']:.2f} |"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        print(f"No escrow-market runs found at {args.input}.")
        return
    print_markdown_table(aggregate(rows))


if __name__ == "__main__":
    main()
