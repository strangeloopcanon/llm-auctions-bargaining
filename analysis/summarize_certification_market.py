"""
Summarize the certification market runs into a compact Markdown table.

Usage (from repo root):
    python -m analysis.summarize_certification_market
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RESULTS_PATH = Path("runs_certification_market/certification_market_results.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize certification market runs.")
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
            "products_built": mean(summary.get("products_built", 0.0) for summary in summaries),
            "certification_successes": mean(
                summary.get("certification_successes", 0.0) for summary in summaries
            ),
            "certification_schedule_notice_rate": mean(
                summary.get("certification_schedule_notice_rate", 0.0) for summary in summaries
            ),
            "shipping_schedule_notice_rate": mean(
                summary.get("shipping_schedule_notice_rate", 0.0) for summary in summaries
            ),
            "certification_schedule_activation_rate": mean(
                summary.get("certification_schedule_activation_rate", 0.0) for summary in summaries
            ),
            "shipping_schedule_activation_rate": mean(
                summary.get("shipping_schedule_activation_rate", 0.0) for summary in summaries
            ),
            "certification_slot_claim_rate": mean(
                summary.get("certification_slot_claim_rate", 0.0) for summary in summaries
            ),
            "shipping_slot_claim_rate": mean(
                summary.get("shipping_slot_claim_rate", 0.0) for summary in summaries
            ),
            "orders_fulfilled": mean(summary.get("orders_fulfilled", 0.0) for summary in summaries),
            "fulfillment_rate": mean(summary.get("fulfillment_rate", 0.0) for summary in summaries),
            "certified_but_undelivered_count": mean(
                summary.get("certified_but_undelivered_count", 0.0) for summary in summaries
            ),
            "ready_but_undelivered_count": mean(
                summary.get("ready_but_undelivered_count", 0.0) for summary in summaries
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
        "| model | arm | runs | products_built | certification_successes | "
        "cert_notice_rate | ship_notice_rate | cert_activation | ship_activation | "
        "cert_claim_rate | ship_claim_rate | orders_fulfilled | fulfillment_rate | "
        "certified_but_undelivered | ready_but_undelivered | welfare | customer_value_captured |"
    )
    print(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for model, arm, metrics in aggregates:
        print(
            "| "
            f"{model} | {arm} | {int(metrics['runs'])} | "
            f"{metrics['products_built']:.2f} | {metrics['certification_successes']:.2f} | "
            f"{metrics['certification_schedule_notice_rate']:.3f} | "
            f"{metrics['shipping_schedule_notice_rate']:.3f} | "
            f"{metrics['certification_schedule_activation_rate']:.3f} | "
            f"{metrics['shipping_schedule_activation_rate']:.3f} | "
            f"{metrics['certification_slot_claim_rate']:.3f} | "
            f"{metrics['shipping_slot_claim_rate']:.3f} | "
            f"{metrics['orders_fulfilled']:.2f} | {metrics['fulfillment_rate']:.3f} | "
            f"{metrics['certified_but_undelivered_count']:.2f} | "
            f"{metrics['ready_but_undelivered_count']:.2f} | "
            f"{metrics['welfare']:.2f} | {metrics['customer_value_captured']:.2f} |"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        print(f"No certification-market runs found at {args.input}.")
        return
    print_markdown_table(aggregate(rows))


if __name__ == "__main__":
    main()
